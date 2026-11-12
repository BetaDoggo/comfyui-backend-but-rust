//! Weight residency and streaming ("system memory offloading").
//!
//! A model is split into units (transformer blocks). Each unit is packed into one contiguous
//! image in the exact device layout the kernels read (fused QKV/gate-up weights, f32 norms, ...).
//! Units [0, R) live in VRAM for the whole run; the rest stay in page-locked host memory and are
//! streamed into a small ring of VRAM slots on a dedicated copy stream. Copies for upcoming units
//! are issued as soon as a slot is released, so PCIe transfers overlap the compute of earlier
//! blocks and the compute stream only waits on an event when a copy is genuinely late.
//!
//! A lazy set ([`WeightSet::lazy`]) keeps no pinned copy of the model: units are read from the file's
//! mapping when they are about to be used, by reader threads that fill a few pinned staging buffers
//! ahead of compute. Clean file pages can be dropped without touching the pagefile, so a model larger
//! than free RAM costs reads, not paging. Units rewritten by a LoRA keep a pinned host image and go
//! straight to VRAM: a copy through staging would cross the memory bus twice, and it is the slow part.
//! So do the units that turn out to be streamed, which are copied on every evaluation, while RAM allows.
//!
//! Under ComfyUI's memory management ([`comfy_memory`], the default) plans are ignored: each unit is faulted into
//! VRAM on use while ComfyUI's budget allows, or copied into a shared cast buffer, and other allocations can take it
//! back out. `--legacy-offload` plans as above.

use crate::comfy_memory;
use crate::cuda::{memcpy_d2d, memcpy_h2d, DevBuf, Event, PinnedBuf, Stream};
use crate::st::{f32_bytes, Dt, SafeTensors};
use anyhow::{bail, Context, Result};
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, RwLock};
use std::time::Instant;

pub const ALIGN: usize = 256;

#[derive(Clone)]
pub enum Src {
    /// raw bytes of one or more tensors, concatenated (int8 weights, f32 scales)
    Raw(Vec<String>),
    /// tensors converted to bf16 and concatenated
    Bf16(Vec<String>),
    /// f32 vector: bf16-rounded value (as ComfyUI holds it) + `add`
    F32(String, f32),
    /// literal bytes
    Bytes(Vec<u8>),
    /// 2D tensors as stored, each zero-padded to (rows, cols) and concatenated, for aligned GEMMs
    Padded(Vec<(String, usize, usize)>),
}

#[derive(Clone)]
struct Entry {
    src: Src,
    off: usize,
    len: usize,
}

/// How a tensor is held in a unit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Stored {
    Raw,
    Bf16,
    /// bf16-rounded value + this
    F32(f32),
}

/// Where a tensor sits in its unit.
#[derive(Clone, Copy, Debug)]
pub struct Placed {
    pub off: usize,
    pub len: usize,
    pub kind: Stored,
}

/// Device layout of one unit.
#[derive(Clone, Default)]
pub struct Layout {
    entries: Vec<Entry>,
    offsets: HashMap<String, usize>,
    pub size: usize,
}

impl Layout {
    pub fn new() -> Self {
        Self::default()
    }

    fn push(&mut self, key: &str, src: Src, len: usize) {
        self.size = (self.size + ALIGN - 1) / ALIGN * ALIGN;
        self.offsets.insert(key.to_string(), self.size);
        self.entries.push(Entry { src, off: self.size, len });
        self.size += len;
    }

    pub fn raw(&mut self, st: &SafeTensors, key: &str, names: &[String]) -> Result<()> {
        let mut len = 0;
        for n in names {
            len += st.info(n)?.len;
        }
        self.push(key, Src::Raw(names.to_vec()), len);
        Ok(())
    }

    pub fn bf16(&mut self, st: &SafeTensors, key: &str, names: &[String]) -> Result<()> {
        let mut len = 0;
        for n in names {
            len += st.packed_len(n, Dt::BF16)?;
        }
        self.push(key, Src::Bf16(names.to_vec()), len);
        Ok(())
    }

    pub fn f32(&mut self, st: &SafeTensors, key: &str, name: &str, add: f32) -> Result<()> {
        let len = st.packed_len(name, Dt::F32)?;
        self.push(key, Src::F32(name.to_string(), add), len);
        Ok(())
    }

    /// 2D tensors zero-padded to `(rows, cols)` each, one after another. Padding with zeros leaves their products as
    /// they are; it only aligns the strides.
    pub fn padded(&mut self, st: &SafeTensors, key: &str, parts: &[(String, usize, usize)]) -> Result<()> {
        let mut len = 0;
        for (n, rows, cols) in parts {
            let i = st.info(n)?;
            if i.shape.len() != 2 || i.shape[0] > *rows || i.shape[1] > *cols {
                anyhow::bail!("{n}: {:?} doesn't pad to {rows}x{cols}", i.shape);
            }
            len += rows * cols * i.dtype.size();
        }
        self.push(key, Src::Padded(parts.to_vec()), len);
        Ok(())
    }

    pub fn bytes(&mut self, key: &str, b: Vec<u8>) {
        let len = b.len();
        self.push(key, Src::Bytes(b), len);
    }

    /// Bytes of one entry.
    pub fn size_of(&self, key: &str) -> usize {
        let off = self.off(key);
        self.entries.iter().find(|e| e.off == off).map_or(0, |e| e.len)
    }

    pub fn off(&self, key: &str) -> usize {
        *self.offsets.get(key).unwrap_or_else(|| panic!("layout key {key}"))
    }

    /// Every tensor of this unit that comes from the file, with its place.
    pub fn tensors(&self, st: &SafeTensors) -> Result<Vec<(String, Placed)>> {
        let mut v = Vec::new();
        for e in &self.entries {
            let (names, dt, kind) = match &e.src {
                Src::Raw(n) => (n.clone(), None, Stored::Raw),
                Src::Bf16(n) => (n.clone(), Some(Dt::BF16), Stored::Bf16),
                Src::F32(n, add) => (vec![n.clone()], Some(Dt::F32), Stored::F32(*add)),
                // padded copies aren't the file's layout; nothing patches them in place
                Src::Bytes(_) | Src::Padded(_) => continue,
            };
            let mut off = e.off;
            for n in names {
                let len = match dt {
                    Some(dt) => st.packed_len(&n, dt)?,
                    None => st.info(&n)?.len,
                };
                v.push((n, Placed { off, len, kind }));
                off += len;
            }
        }
        Ok(v)
    }

    /// Take this unit's tensors out of the process, once they are held elsewhere.
    pub fn evict(&self, st: &SafeTensors) {
        for e in &self.entries {
            match &e.src {
                Src::Raw(names) | Src::Bf16(names) => names.iter().for_each(|n| st.evict(n)),
                Src::F32(n, _) => st.evict(n),
                Src::Padded(parts) => parts.iter().for_each(|(n, ..)| st.evict(n)),
                Src::Bytes(_) => {}
            }
        }
    }

    pub fn pack(&self, st: &SafeTensors, dst: &mut [u8]) -> Result<()> {
        for e in &self.entries {
            Self::pack_entry(e, st, &mut dst[e.off..e.off + e.len])?;
        }
        Ok(())
    }

    /// This unit as it can go to VRAM from the file's mapping: its large raw tensors where they lie, page-locked, and
    /// the rest packed in a small buffer. None when much of it isn't stored as the unit reads it, or the driver won't
    /// lock the mapping.
    fn map(&self, st: &SafeTensors) -> Result<Option<Image>> {
        let lies = |e: &Entry| matches!(&e.src, Src::Raw(names) if names.iter().all(|n| st.info(n).is_ok_and(|i| i.len >= MAPPED_MIN)));
        let rest_len: usize = self.entries.iter().filter(|e| !lies(e)).map(|e| e.len).sum();
        if rest_len * 8 > self.size {
            return Ok(None);
        }
        let mut rest = PinnedBuf::alloc(rest_len)?;
        let mut parts = Vec::new();
        let mut o = 0;
        for e in &self.entries {
            match &e.src {
                Src::Raw(names) if lies(e) => {
                    let mut off = e.off;
                    for n in names {
                        let Some(runs) = st.pin(n) else { return Ok(None) };
                        for (addr, len) in runs {
                            parts.push((off, addr, len));
                            off += len;
                        }
                    }
                }
                _ => {
                    Self::pack_entry(e, st, &mut rest.as_mut_slice()[o..o + e.len])?;
                    parts.push((e.off, rest.ptr() as usize + o, e.len));
                    o += e.len;
                }
            }
        }
        Ok(Some(Image::Mapped { rest, parts }))
    }

    fn pack_entry(e: &Entry, st: &SafeTensors, d: &mut [u8]) -> Result<()> {
        match &e.src {
            Src::Raw(names) => {
                let mut o = 0;
                for n in names {
                    let b = st.bytes(n)?;
                    d[o..o + b.len()].copy_from_slice(b);
                    o += b.len();
                }
            }
            Src::Bf16(names) => {
                let mut o = 0;
                for n in names {
                    let len = st.packed_len(n, Dt::BF16)?;
                    st.bf16_into(n, &mut d[o..o + len])?;
                    o += len;
                }
            }
            Src::F32(n, add) => {
                let v: Vec<f32> = st.f32_vec(n, true)?.into_iter().map(|x| x + add).collect();
                d.copy_from_slice(&f32_bytes(&v));
            }
            Src::Bytes(b) => d.copy_from_slice(b),
            Src::Padded(parts) => {
                d.fill(0);
                let mut o = 0;
                for (n, rows, cols) in parts {
                    let i = st.info(n)?;
                    let (es, (r, c)) = (i.dtype.size(), (i.shape[0], i.shape[1]));
                    let b = st.bytes(n)?;
                    for y in 0..r {
                        d[o + y * cols * es..][..c * es].copy_from_slice(&b[y * c * es..][..c * es]);
                    }
                    o += rows * cols * es;
                }
            }
        }
        Ok(())
    }
}

/// A unit lives in VRAM (`dev`), in page-locked RAM (`host`), or both. Units that started out
/// resident get their host copy packed lazily the first time they are demoted.
#[derive(Default)]
struct Copies {
    dev: Option<DevBuf>,
    host: Option<PinnedBuf>,
}

struct Slot {
    buf: DevBuf,
    unit: Option<usize>,
    ready: Event,
    free: Event,
    released: bool,
}

/// Where a model's weights live, to rewrite them in place.
pub enum Store<'a> {
    Set(&'a mut WeightSet),
    Buf(&'a DevBuf, &'a Layout),
}

impl Store<'_> {
    pub fn layouts(&self) -> &[Layout] {
        match self {
            Store::Set(w) => &w.layouts,
            Store::Buf(_, l) => std::slice::from_ref(*l),
        }
    }

    /// Copy `len` bytes of `src` over `unit` at `off`. Issued on `st`.
    pub fn write(&mut self, unit: usize, off: usize, src: &DevBuf, len: usize, st: &Stream) -> Result<()> {
        match self {
            Store::Set(w) => w.write(unit, off, src, len, st),
            Store::Buf(b, _) => memcpy_d2d(b.at(off), src.ptr(), len, st),
        }
    }
}

/// Units staged per reader thread, ahead of compute.
const STAGING: usize = 4;

/// In place of a unit, tells a reader thread to stop.
const STOP: usize = usize::MAX;

/// RAM left available after a streamed unit of a lazy set is given a pinned image.
const KEEP_RAM_FREE: u64 = 8 << 30;

#[repr(C)]
struct MemoryStatusEx {
    length: u32,
    load: u32,
    total_phys: u64,
    avail_phys: u64,
    rest: [u64; 5],
}

#[link(name = "kernel32")]
extern "system" {
    fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
}

/// Physical memory the system can hand out now, file cache included.
fn available_ram() -> Option<u64> {
    let mut m = MemoryStatusEx { length: std::mem::size_of::<MemoryStatusEx>() as u32, load: 0, total_phys: 0, avail_phys: 0, rest: [0; 5] };
    (unsafe { GlobalMemoryStatusEx(&mut m) } != 0).then_some(m.avail_phys)
}

/// A unit a reader is done with: in its staging buffer, or None when it goes to VRAM from its image.
type Staged = (usize, u64, Result<Option<PinnedBuf>>);

/// Leave to fill a staging buffer, with the buffer if one is free and what it must wait for.
type Ticket = (Option<PinnedBuf>, Option<Event>);

/// The staging buffers of a lazy set. They are taken from the driver only when a unit has to pass
/// through one, so a set that is idle, or holds its images, keeps none.
struct Staging {
    tickets: Mutex<Vec<Ticket>>,
    ready: Condvar,
    generation: AtomicU64,
}

impl Staging {
    fn new() -> Self {
        Staging { tickets: Mutex::new((0..STAGING).map(|_| (None, None)).collect()), ready: Condvar::new(), generation: AtomicU64::new(0) }
    }

    fn current(&self, generation: u64) -> bool {
        self.generation.load(Ordering::SeqCst) == generation
    }

    /// The next ticket, or None once `generation` is over.
    fn take(&self, generation: u64) -> Option<Ticket> {
        let mut q = self.tickets.lock().unwrap();
        loop {
            if !self.current(generation) {
                return None;
            }
            if !q.is_empty() {
                return Some(q.remove(0));
            }
            q = self.ready.wait(q).unwrap();
        }
    }

    fn put(&self, t: Ticket) {
        self.tickets.lock().unwrap().push(t);
        self.ready.notify_one();
    }

    fn start(&self, generation: u64) {
        let _q = self.tickets.lock().unwrap();
        self.generation.store(generation, Ordering::SeqCst);
        self.ready.notify_all();
    }

    /// Give the waiting buffers back. Their copies must be finished.
    fn empty(&self) {
        self.tickets.lock().unwrap().iter_mut().for_each(|t| *t = (None, None));
    }
}

/// Raw tensors smaller than this are copied next to a mapped unit's other small parts.
const MAPPED_MIN: usize = 1 << 20;

/// A unit held in RAM, pinned unless there is no room for that.
enum Image {
    Pinned(PinnedBuf),
    Plain(Vec<u8>),
    /// The file's own pages, page-locked in its mapping, and what the file doesn't hold as the unit reads it in `rest`.
    /// `parts` are the unit's pieces, each (offset in the unit, address, bytes).
    Mapped { rest: PinnedBuf, parts: Vec<(usize, usize, usize)> },
}

impl Image {
    fn new(size: usize) -> Self {
        PinnedBuf::alloc(size).map_or_else(|_| Image::Plain(vec![0; size]), Image::Pinned)
    }

    /// The unit's bytes, unless they lie in the file's mapping.
    fn bytes(&mut self) -> Option<&mut [u8]> {
        match self {
            Image::Pinned(p) => Some(p.as_mut_slice()),
            Image::Plain(v) => Some(v),
            Image::Mapped { .. } => None,
        }
    }

    /// Read a mapped unit's pages from the file, ahead of the copies that page-lock them.
    fn read_in(&self) {
        if let Image::Mapped { rest, parts } = self {
            let own = rest.ptr() as usize..rest.ptr() as usize + rest.size();
            parts.iter().filter(|(_, p, _)| !own.contains(p)).for_each(|&(_, p, len)| crate::st::read_in(p as *const u8, len));
        }
    }
}

/// What a lazy set adds: the readers and what they have staged.
struct Lazy {
    st: Arc<SafeTensors>,
    images: Arc<Vec<RwLock<Option<Image>>>>,
    keep: Arc<AtomicBool>,
    /// Units that went through a cast buffer or a ring slot: they are copied on every evaluation, so the readers
    /// give them a pinned image when they next read them.
    streams: Arc<Vec<AtomicBool>>,
    want: mpsc::Sender<(usize, u64)>,
    done: mpsc::Receiver<Staged>,
    staging: Arc<Staging>,
    staged: HashMap<usize, Option<PinnedBuf>>,
    asked: HashSet<usize>,
    /// Resident units whose VRAM holds their data.
    filled: Vec<bool>,
    generation: u64,
    readers: Vec<std::thread::JoinHandle<()>>,
}

impl Drop for Lazy {
    fn drop(&mut self) {
        // The readers share the images and the staging buffers, and would free the last of them as they stop. Page-locked
        // memory and events released on another thread while the process exits hang or crash in the driver, so the
        // readers are stopped first and everything is released here.
        self.staging.start(u64::MAX);
        for _ in &self.readers {
            let _ = self.want.send((STOP, 0));
        }
        for r in self.readers.drain(..) {
            let _ = r.join();
        }
    }
}

pub struct WeightSet {
    pub name: String,
    pub layouts: Vec<Layout>,
    stores: Vec<Copies>,
    /// Units rewritten since loading: the file no longer holds their contents.
    rewritten: Vec<bool>,
    streamed: Vec<usize>,           // unit ids streamed, in execution order
    slot_of: HashMap<usize, usize>, // unit -> slot currently holding it
    next_rank: usize,               // next index of `streamed` to copy
    rr: usize,                      // next slot to fill
    slots: Vec<Slot>,
    copy: Stream,
    lazy: Option<Lazy>,
    comfy: Option<Comfy>,
    pub copied_bytes: usize,
}

/// A set under ComfyUI's memory management: its model in [`comfy_memory`], and the units acquired and not yet released,
/// each pinned in VRAM (None) or in a cast buffer (its index).
struct Comfy {
    id: u64,
    held: HashMap<usize, Option<usize>>,
}

impl Comfy {
    fn new(name: &str, layouts: &[Layout]) -> Result<Option<Self>> {
        if !comfy_memory::enabled() {
            return Ok(None);
        }
        let id = comfy_memory::register(name, layouts.iter().map(|l| l.size).collect())?;
        Ok(Some(Comfy { id, held: HashMap::new() }))
    }

    /// Give back what `unit` holds.
    fn drop_hold(&self, unit: usize, hold: Option<usize>, compute: Option<&Stream>) -> Result<()> {
        match (hold, compute) {
            (None, _) => comfy_memory::unpin(self.id, unit),
            (Some(i), Some(c)) => comfy_memory::cast_release(i, c),
            (Some(i), None) => comfy_memory::cast_abandon(i),
        }
    }
}

impl Drop for WeightSet {
    fn drop(&mut self) {
        if let Some(c) = &mut self.comfy {
            for (u, h) in std::mem::take(&mut c.held) {
                let _ = c.drop_hold(u, h, None);
            }
            comfy_memory::unregister(c.id);
        }
    }
}

pub struct Plan {
    pub resident: usize,
    pub slots: usize,
}

/// Decide how many units (in order) stay resident within `budget` bytes of VRAM, reserving
/// `slots` streaming slots if anything has to be streamed.
pub fn plan(layouts: &[Layout], budget: usize, slots: usize) -> Plan {
    let total: usize = layouts.iter().map(|l| l.size).sum();
    if total <= budget {
        return Plan { resident: layouts.len(), slots: 0 };
    }
    let slot_size = layouts.iter().map(|l| l.size).max().unwrap_or(0);
    let mut avail = budget.saturating_sub(slots * slot_size);
    let mut r = 0;
    while r < layouts.len() && layouts[r].size <= avail {
        avail -= layouts[r].size;
        r += 1;
    }
    // too few streamed units for a ring: give their slots back as residency
    let streamed = layouts.len() - r;
    Plan { resident: r, slots: slots.min(streamed).max(1) }
}

impl WeightSet {
    /// Pack all units from `st` (multi-threaded; disk/page-cache bound), uploading resident ones.
    pub fn load(name: &str, st: &SafeTensors, layouts: Vec<Layout>, plan: Plan, threads: usize) -> Result<Self> {
        let t0 = Instant::now();
        // ComfyUI's policy decides residency as it runs, from a host copy of every unit
        let plan = if comfy_memory::enabled() { Plan { resident: 0, slots: 0 } } else { plan };
        let comfy = Comfy::new(name, &layouts)?;
        let n = layouts.len();
        let copy = Stream::new()?;
        let max_size = layouts.iter().map(|l| l.size).max().unwrap_or(0);
        let resident: Vec<bool> = (0..n).map(|i| i < plan.resident).collect();


        let mut dev: Vec<Option<DevBuf>> = Vec::with_capacity(n);
        for i in 0..n {
            dev.push(if resident[i] { Some(DevBuf::alloc(layouts[i].size)?) } else { None });
        }

        let nstage = threads.min(plan.resident).max(if plan.resident > 0 { 1 } else { 0 });
        let (free_tx, free_rx) = mpsc::channel::<PinnedBuf>();
        for _ in 0..nstage {
            free_tx.send(PinnedBuf::alloc(max_size)?)?;
        }
        let free_rx = Mutex::new(free_rx);
        let (done_tx, done_rx) = mpsc::channel::<(usize, Result<PinnedBuf>)>();
        let next = AtomicUsize::new(0);
        let host_out: Mutex<Vec<Option<PinnedBuf>>> = Mutex::new((0..n).map(|_| None).collect());
        let err: Mutex<Option<anyhow::Error>> = Mutex::new(None);

        std::thread::scope(|s| -> Result<()> {
            for _ in 0..threads.max(1) {
                let done_tx = done_tx.clone();
                let (layouts, resident, next, free_rx, host_out, err) =
                    (&layouts, &resident, &next, &free_rx, &host_out, &err);
                s.spawn(move || loop {
                    crate::st::read_in_background();
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    if i >= n {
                        break;
                    }
                    if resident[i] {
                        let buf = free_rx.lock().unwrap().recv();
                        let Ok(mut buf) = buf else { break };
                        let r = layouts[i].pack(st, &mut buf.as_mut_slice()[..layouts[i].size]).map(|_| buf);
                        layouts[i].evict(st);
                        if done_tx.send((i, r)).is_err() {
                            break;
                        }
                    } else {
                        let r = PinnedBuf::alloc(layouts[i].size)
                            .and_then(|mut b| layouts[i].pack(st, b.as_mut_slice()).map(|_| b));
                        layouts[i].evict(st);
                        match r {
                            Ok(b) => host_out.lock().unwrap()[i] = Some(b),
                            Err(e) => *err.lock().unwrap() = Some(e),
                        }
                    }
                });
            }
            drop(done_tx);
            // upload resident units as workers finish packing them
            for (i, r) in done_rx.iter() {
                let buf = r?;
                let d = dev[i].as_ref().unwrap();
                memcpy_h2d(d.ptr(), buf.ptr(), layouts[i].size, &copy)?;
                copy.sync()?;
                let _ = free_tx.send(buf);
            }
            drop(free_tx);
            Ok(())
        })?;
        if let Some(e) = err.into_inner().unwrap() {
            return Err(e);
        }
        let mut host = host_out.into_inner().unwrap();
        let stores: Vec<Copies> = (0..n).map(|i| Copies { dev: dev[i].take(), host: host[i].take() }).collect();

        let streamed: Vec<usize> = (plan.resident..n).collect();
        let mut slots = Vec::new();
        if !streamed.is_empty() {
            for _ in 0..plan.slots {
                slots.push(Slot { buf: DevBuf::alloc(max_size)?, unit: None, ready: Event::new()?, free: Event::new()?, released: true });
            }
        }
        let resident_bytes: usize = layouts[..plan.resident].iter().map(|l| l.size).sum();
        let streamed_bytes: usize = layouts[plan.resident..].iter().map(|l| l.size).sum();
        eprintln!(
            "[{name}] {} units: {} resident ({:.2} GB VRAM), {} streamed from pinned RAM ({:.2} GB) via {} slots; packed in {:.1}s",
            n, plan.resident, resident_bytes as f64 / 1e9, streamed.len(), streamed_bytes as f64 / 1e9,
            slots.len(), t0.elapsed().as_secs_f64()
        );
        let mut ws = WeightSet {
            name: name.to_string(), layouts, stores, rewritten: vec![false; n], streamed, slot_of: HashMap::new(), next_rank: 0, rr: 0, slots,
            copy,
            lazy: None,
            comfy,
            copied_bytes: 0,
        };
        ws.prefetch_initial()?;
        Ok(ws)
    }

    /// A set that reads nothing until it is used. VRAM for the plan is taken now.
    pub fn lazy(name: &str, st: Arc<SafeTensors>, layouts: Vec<Layout>, plan: Plan, threads: usize) -> Result<Self> {
        let plan = if comfy_memory::enabled() { Plan { resident: 0, slots: 0 } } else { plan };
        let comfy = Comfy::new(name, &layouts)?;
        let n = layouts.len();
        let max_size = layouts.iter().map(|l| l.size).max().unwrap_or(0);
        let stores = (0..n).map(|i| Ok(Copies { dev: if i < plan.resident { Some(DevBuf::alloc(layouts[i].size)?) } else { None }, host: None })).collect::<Result<Vec<_>>>()?;
        let streamed: Vec<usize> = (plan.resident..n).collect();
        let mut slots = Vec::new();
        if !streamed.is_empty() {
            for _ in 0..plan.slots {
                slots.push(Slot { buf: DevBuf::alloc(max_size)?, unit: None, ready: Event::new()?, free: Event::new()?, released: true });
            }
        }
        let (want, want_rx) = mpsc::channel::<(usize, u64)>();
        let (done_tx, done) = mpsc::channel::<Staged>();
        let staging = Arc::new(Staging::new());
        let images: Arc<Vec<RwLock<Option<Image>>>> = Arc::new((0..n).map(|_| RwLock::new(None)).collect());
        let want_rx = Arc::new(Mutex::new(want_rx));
        let keep = Arc::new(AtomicBool::new(false));
        let streams: Arc<Vec<AtomicBool>> = Arc::new((0..n).map(|_| AtomicBool::new(false)).collect());
        let mut readers = Vec::new();
        // one reader: more of them read no faster from a drive, and leave it less room for anyone else
        let _ = threads;
        for _ in 0..1 {
            let (want_rx, staging, done_tx, st, layouts, images, keep, streams) =
                (want_rx.clone(), staging.clone(), done_tx.clone(), st.clone(), layouts.clone(), images.clone(), keep.clone(), streams.clone());
            readers.push(std::thread::spawn(move || loop {
                crate::st::read_in_background();
                let Ok((unit, generation)) = want_rx.lock().unwrap().recv() else { break };
                if unit == STOP {
                    break;
                }
                // a streamed unit goes to VRAM from the file's mapping when it can: that takes no second copy of it in RAM
                // and no staging buffer, and leaves the file cache holding it when the process ends
                let mut mapped = matches!(&*images[unit].read().unwrap(), Some(Image::Mapped { .. }));
                let size = layouts[unit].size;
                if !mapped && images[unit].read().unwrap().is_none() && !keep.load(Ordering::Relaxed) && streams[unit].load(Ordering::Relaxed)
                    && available_ram().is_some_and(|a| a >= size as u64 + KEEP_RAM_FREE)
                {
                    match layouts[unit].map(&st) {
                        Ok(Some(img)) => {
                            img.read_in();
                            let mut held = images[unit].write().unwrap();
                            if held.is_none() {
                                *held = Some(img);
                                mapped = true;
                            }
                        }
                        Ok(None) => {}
                        Err(e) => {
                            if done_tx.send((unit, generation, Err(e))).is_err() {
                                break;
                            }
                            continue;
                        }
                    }
                }
                if mapped {
                    if staging.current(generation) && done_tx.send((unit, generation, Ok(None))).is_err() {
                        break;
                    }
                    continue;
                }
                let Some((buf, event)) = staging.take(generation) else { continue };
                let r = (|| -> Result<Option<PinnedBuf>> {
                    if let Some(e) = event {
                        e.sync()?;
                    }
                    let mut buf = match buf {
                        Some(b) => b,
                        None => PinnedBuf::alloc(max_size)?,
                    };
                    let mut held = images[unit].write().unwrap();
                    match held.as_mut().map(|img| img.bytes()) {
                        Some(Some(b)) => buf.as_mut_slice()[..size].copy_from_slice(&b[..size]),
                        Some(None) => layouts[unit].pack(&st, &mut buf.as_mut_slice()[..size])?,
                        None => {
                            layouts[unit].pack(&st, &mut buf.as_mut_slice()[..size])?;
                            // a streamed unit is only worth holding pinned: from pageable memory it would pass through here again
                            let img = if keep.load(Ordering::Relaxed) {
                                Some(Image::new(size))
                            } else if streams[unit].load(Ordering::Relaxed) && available_ram().is_some_and(|a| a >= size as u64 + KEEP_RAM_FREE) {
                                PinnedBuf::alloc(size).ok().map(Image::Pinned)
                            } else {
                                None
                            };
                            if let Some(mut img) = img {
                                img.bytes().unwrap()[..size].copy_from_slice(&buf.as_mut_slice()[..size]);
                                *held = Some(img);
                                layouts[unit].evict(&st);
                            }
                        }
                    }
                    Ok(Some(buf))
                })();
                // nobody waits for a stale unit, and a parked set wants its memory back
                if r.is_err() || !staging.current(generation) {
                    staging.put((None, None));
                    if r.is_ok() {
                        continue;
                    }
                }
                if done_tx.send((unit, generation, r)).is_err() {
                    break;
                }
            }));
        }
        let total: usize = layouts.iter().map(|l| l.size).sum();
        eprintln!("[{name}] {n} units, {:.2} GB: {} resident, {} streamed via {} slots, read on use", total as f64 / 1e9, plan.resident, streamed.len(), slots.len());
        let lazy = Lazy { st, images, keep, streams, want, done, staging, staged: HashMap::new(), asked: HashSet::new(), filled: vec![false; n], generation: 0, readers };
        Ok(WeightSet { name: name.to_string(), layouts, stores, rewritten: vec![false; n], streamed, slot_of: HashMap::new(), next_rank: 0, rr: 0, slots, copy: Stream::new()?, lazy: Some(lazy), comfy, copied_bytes: 0 })
    }

    /// Move a lazy set to `p`, freeing VRAM before taking any. What leaves VRAM is read again on use,
    /// LoRA bakes included. The compute stream must be idle. Keeps everything if `p` is the current plan.
    pub fn replan(&mut self, p: Plan) -> Result<()> {
        if self.comfy.is_some() {
            // a plan to run is ComfyUI's load_models_gpu; a plan to park leaves it to the budget
            return if p.resident > 0 || p.slots > 0 { self.prioritize() } else { Ok(()) };
        }
        let n = self.len();
        if self.resident() == p.resident.min(n) && self.slots.len() == p.slots {
            return Ok(());
        }
        self.copy.sync()?;
        for s in &mut self.stores {
            s.dev = None;
        }
        self.slots.clear();
        // streamed units spread between resident ones, so each copy overlaps compute
        let out = n - p.resident.min(n);
        let streamed: Vec<usize> = (0..n).filter(|i| (i + 1) * out / n > i * out / n).collect();
        for i in (0..n).filter(|i| !streamed.contains(i)) {
            self.stores[i].dev = Some(DevBuf::alloc(self.layouts[i].size)?);
        }
        let max_size = self.layouts.iter().map(|l| l.size).max().unwrap_or(0);
        for _ in 0..p.slots {
            self.slots.push(Slot { buf: DevBuf::alloc(max_size)?, unit: None, ready: Event::new()?, free: Event::new()?, released: true });
        }
        self.streamed = streamed;
        if let Some(l) = &mut self.lazy {
            l.filled.iter_mut().for_each(|f| *f = false);
        }
        eprintln!("[{}] {} units in VRAM, {} streamed via {} slots", self.name, self.resident(), self.streamed.len(), self.slots.len());
        self.restart_ring()?;
        if let (true, Some(l)) = (self.stores.iter().all(|s| s.dev.is_none()) && self.slots.is_empty(), &self.lazy) {
            while let Ok((_, _, r)) = l.done.try_recv() {
                if matches!(r, Ok(Some(_))) {
                    l.staging.put((None, None));
                }
            }
            l.staging.empty();
        }
        Ok(())
    }

    /// Take a set that will not run for a while out of the process.
    pub fn evict(&self) {
        if let Some(l) = &self.lazy {
            self.layouts.iter().for_each(|u| u.evict(&l.st));
        }
    }

    /// As [`Self::evict`], for the units held in host memory since a LoRA rewrote them.
    pub fn evict_rewritten(&self) {
        if let Some(l) = &self.lazy {
            self.layouts.iter().zip(&self.rewritten).filter(|(_, r)| **r).for_each(|(u, _)| u.evict(&l.st));
        }
    }

    /// Keep what is read in host memory, so a lazy set that returns needs no second trip to the file.
    pub fn keep_images(&self) {
        if let Some(l) = &self.lazy {
            l.keep.store(true, Ordering::Relaxed);
        }
    }

    /// For a set that runs once per prompt. Its units are held in pinned RAM from their first read, while RAM allows:
    /// the file cache doesn't reliably keep what was read through the mapping, and reading it again costs a trip to
    /// the disk. Under ComfyUI's memory management it is streamed rather than faulted into VRAM while the models
    /// don't fit there together ([`comfy_memory::set_transient`]).
    pub fn once_per_prompt(&self) {
        if let Some(l) = &self.lazy {
            l.streams.iter().for_each(|s| s.store(true, Ordering::Relaxed));
        }
        if let Some(c) = &self.comfy {
            comfy_memory::set_transient(c.id);
        }
    }

    /// True when `unit`'s image can go to VRAM as it is.
    fn has_direct(&self, unit: usize) -> bool {
        self.lazy.as_ref().is_some_and(|l| matches!(&*l.images[unit].read().unwrap(), Some(Image::Pinned(_) | Image::Mapped { .. })))
    }

    /// `unit`'s image as the pieces that go to VRAM as they are, each (offset in the unit, address, bytes).
    fn direct(&self, unit: usize) -> Option<Vec<(usize, usize, usize)>> {
        match &*self.lazy.as_ref()?.images[unit].read().unwrap() {
            Some(Image::Pinned(p)) => Some(vec![(0, p.ptr() as usize, self.layouts[unit].size)]),
            Some(Image::Mapped { parts, .. }) => Some(parts.clone()),
            _ => None,
        }
    }

    /// Copy `unit`'s pieces to `dst` on the copy stream.
    fn copy_direct(&mut self, unit: usize, parts: &[(usize, usize, usize)], dst: *mut c_void) -> Result<Event> {
        for &(off, src, len) in parts {
            memcpy_h2d(unsafe { (dst as *mut u8).add(off) }.cast(), src as *const u8, len, &self.copy)?;
        }
        let done = Event::new()?;
        done.record(&self.copy)?;
        self.copied_bytes += self.layouts[unit].size;
        Ok(done)
    }

    /// True when `unit`'s data will have to come from the host.
    fn needs_data(&self, unit: usize) -> bool {
        let Some(l) = &self.lazy else { return false };
        if let Some(c) = &self.comfy {
            return !comfy_memory::holds(c.id, unit);
        }
        if self.stores[unit].dev.is_some() { !l.filled[unit] } else { !self.slot_of.contains_key(&unit) }
    }

    /// Ask the readers for the units after `unit`, as far as the staging buffers go.
    fn read_ahead(&mut self, unit: usize) -> Result<()> {
        let n = self.len();
        let mut ahead = 0;
        for j in 0..n {
            let v = (unit + j) % n;
            if ahead == STAGING - 1 {
                break;
            }
            if self.needs_data(v) && !self.has_direct(v) {
                ahead += 1;
                let l = self.lazy.as_mut().unwrap();
                if l.asked.insert(v) {
                    l.want.send((v, l.generation))?;
                }
            }
        }
        Ok(())
    }

    /// Copy `unit` from its staging buffer to `dst` on the copy stream. Blocks until it is read.
    fn fill(&mut self, unit: usize, dst: *mut c_void) -> Result<Event> {
        let size = self.layouts[unit].size;
        // what the readers were asked for is on its way through staging, whatever became of its image
        let direct = if self.lazy.as_ref().unwrap().asked.contains(&unit) { None } else { self.direct(unit) };
        if let Some(parts) = direct {
            return self.copy_direct(unit, &parts, dst);
        }
        let l = self.lazy.as_mut().unwrap();
        if l.asked.insert(unit) {
            l.want.send((unit, l.generation))?;
        }
        while !l.staged.contains_key(&unit) {
            let (u, generation, r) = l.done.recv()?;
            let buf = r?;
            if generation == l.generation {
                l.staged.insert(u, buf);
            } else if buf.is_some() {
                l.staging.put((buf, None));
            }
        }
        let buf = l.staged.remove(&unit).unwrap();
        l.asked.remove(&unit);
        let Some(buf) = buf else {
            let parts = self.direct(unit).context("a unit left its mapping while it was read")?;
            return self.copy_direct(unit, &parts, dst);
        };
        memcpy_h2d(dst, buf.ptr(), size, &self.copy)?;
        let done = Event::new()?;
        done.record(&self.copy)?;
        l.staging.put((Some(buf), Some(Event::new_like(&done, &self.copy)?)));
        self.copied_bytes += size;
        Ok(done)
    }

    pub fn len(&self) -> usize {
        self.layouts.len()
    }

    /// Number of units held in VRAM.
    pub fn resident(&self) -> usize {
        if let Some(c) = &self.comfy {
            return comfy_memory::resident(c.id).0;
        }
        self.stores.iter().filter(|s| s.dev.is_some()).count()
    }

    /// VRAM held by this set (resident units + streaming slots).
    pub fn vram_bytes(&self) -> usize {
        if let Some(c) = &self.comfy {
            return comfy_memory::resident(c.id).1;
        }
        let res: usize = self.stores.iter().filter_map(|s| s.dev.as_ref()).map(|d| d.size()).sum();
        res + self.slots.iter().map(|s| s.buf.size()).sum::<usize>()
    }

    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Move the residency boundary to `r` units (the first `r` in VRAM, the rest streamed through
    /// `nslots` ring slots). Frees VRAM before allocating it. The compute stream must be idle.
    pub fn set_resident(&mut self, r: usize, nslots: usize, st: &SafeTensors) -> Result<()> {
        if self.comfy.is_some() {
            return Ok(());
        }
        let t0 = Instant::now();
        let n = self.len();
        let r = r.min(n);
        let before = self.resident();
        self.copy.sync()?;
        // demote: make sure a host copy exists, then release the VRAM
        for i in r..n {
            if self.stores[i].dev.is_some() {
                if self.stores[i].host.is_none() {
                    let mut b = PinnedBuf::alloc(self.layouts[i].size)?;
                    if self.rewritten[i] {
                        self.stores[i].dev.as_ref().unwrap().download(0, b.as_mut_slice(), &self.copy)?;
                    } else {
                        self.layouts[i].pack(st, b.as_mut_slice())?;
                    }
                    self.stores[i].host = Some(b);
                }
                self.stores[i].dev = None;
            }
        }
        let want_slots = if r < n { nslots.min(n - r).max(1) } else { 0 };
        if self.slots.len() > want_slots {
            self.slots.clear();
        }
        // promote from the pinned copy
        for i in 0..r {
            if self.stores[i].dev.is_none() {
                let d = DevBuf::alloc(self.layouts[i].size)?;
                let h = self.stores[i].host.as_ref().unwrap();
                memcpy_h2d(d.ptr(), h.ptr(), h.size(), &self.copy)?;
                self.copied_bytes += h.size();
                self.stores[i].dev = Some(d);
            }
        }
        self.copy.sync()?;
        let max_size = self.layouts.iter().map(|l| l.size).max().unwrap_or(0);
        while self.slots.len() < want_slots {
            self.slots.push(Slot { buf: DevBuf::alloc(max_size)?, unit: None, ready: Event::new()?, free: Event::new()?, released: true });
        }
        self.streamed = (r..n).collect();
        self.restart_ring()?;
        if before != r {
            eprintln!("[{}] residency {before} -> {r} units in VRAM ({} streamed via {} slots) in {:.0} ms",
                self.name, n - r, self.slots.len(), t0.elapsed().as_secs_f64() * 1e3);
        }
        Ok(())
    }

    /// Forget what the ring holds. Compute and copy streams must be idle.
    pub fn restart_ring(&mut self) -> Result<()> {
        if let Some(l) = &mut self.lazy {
            l.generation += 1;
            l.staging.start(l.generation);
            l.asked.clear();
            for (_, buf) in l.staged.drain() {
                if buf.is_some() {
                    l.staging.put((buf, None));
                }
            }
        }
        self.slot_of.clear();
        for s in &mut self.slots {
            s.unit = None;
            s.released = true;
        }
        self.rr = 0;
        self.next_rank = 0;
        self.prefetch_initial()
    }

    /// Wait for the copies in flight, so units can be rewritten.
    pub fn quiesce(&self) -> Result<()> {
        self.copy.sync()
    }

    /// Copy `len` bytes of `src` over every copy of `unit`. Call [`Self::restart_ring`] when done.
    fn write(&mut self, unit: usize, off: usize, src: &DevBuf, len: usize, st: &Stream) -> Result<()> {
        self.rewritten[unit] = true;
        if let Some(c) = &self.comfy {
            comfy_memory::write(c.id, unit, off, src, len, st)?;
        }
        if let Some(l) = &self.lazy {
            let mut image = l.images[unit].write().unwrap();
            // a rewritten unit is no longer what the file holds
            if !matches!(&*image, Some(Image::Pinned(_) | Image::Plain(_))) {
                let mut b = Image::new(self.layouts[unit].size);
                self.layouts[unit].pack(&l.st, &mut b.bytes().unwrap()[..self.layouts[unit].size])?;
                *image = Some(b);
            }
            src.download(0, &mut image.as_mut().unwrap().bytes().unwrap()[off..off + len], st)?;
            if let (Some(d), true) = (&self.stores[unit].dev, l.filled[unit]) {
                memcpy_d2d(d.at(off), src.ptr(), len, st)?;
            }
            return Ok(());
        }
        let store = &mut self.stores[unit];
        if let Some(d) = &store.dev {
            memcpy_d2d(d.at(off), src.ptr(), len, st)?;
        }
        if let Some(h) = &mut store.host {
            src.download(0, &mut h.as_mut_slice()[off..off + len], st)?;
        }
        Ok(())
    }

    /// Copy the next streamed unit (cyclic execution order) into the next ring slot, if that
    /// slot has been released. Returns false when the ring is full.
    fn issue_next(&mut self) -> Result<bool> {
        let ns = self.slots.len();
        let si = self.rr;
        if !self.slots[si].released {
            return Ok(false);
        }
        let unit = self.streamed[self.next_rank];
        // the slot's previous occupant must be finished on the compute stream
        self.copy.wait(&self.slots[si].free)?;
        if let Some(l) = &self.lazy {
            l.streams[unit].store(true, Ordering::Relaxed);
            let dst = self.slots[si].buf.ptr();
            self.fill(unit, dst)?;
        } else {
            let h = self.stores[unit].host.as_ref().expect("streamed unit without host copy");
            memcpy_h2d(self.slots[si].buf.ptr(), h.ptr(), h.size(), &self.copy)?;
            self.copied_bytes += h.size();
        }
        let slot = &mut self.slots[si];
        slot.ready.record(&self.copy)?;
        if let Some(old) = slot.unit.replace(unit) {
            self.slot_of.remove(&old);
        }
        slot.released = false;
        self.slot_of.insert(unit, si);
        self.rr = (self.rr + 1) % ns;
        self.next_rank = (self.next_rank + 1) % self.streamed.len();
        Ok(true)
    }

    fn prefetch_initial(&mut self) -> Result<()> {
        if self.lazy.is_some() {
            return Ok(());
        }
        while !self.slots.is_empty() && self.issue_next()? {
            if self.slot_of.len() == self.streamed.len() {
                break;
            }
        }
        Ok(())
    }

    /// Device base pointer of `unit`, making `compute` wait for its copy if it is streamed.
    pub fn acquire(&mut self, unit: usize, compute: &Stream) -> Result<*mut c_void> {
        if self.comfy.is_some() {
            return self.comfy_acquire(unit, compute);
        }
        if self.lazy.is_some() {
            self.read_ahead(unit)?;
            if self.stores[unit].dev.is_some() && self.needs_data(unit) {
                let dst = self.stores[unit].dev.as_ref().unwrap().ptr();
                let done = self.fill(unit, dst)?;
                compute.wait(&done)?;
                self.lazy.as_mut().unwrap().filled[unit] = true;
            }
        }
        if let Some(d) = &self.stores[unit].dev {
            return Ok(d.ptr());
        }
        while !self.slot_of.contains_key(&unit) {
            if !self.issue_next()? {
                bail!("{}: unit {unit} requested out of order while all slots are busy", self.name);
            }
        }
        let si = self.slot_of[&unit];
        compute.wait(&self.slots[si].ready)?;
        Ok(self.slots[si].buf.ptr())
    }

    /// Mark `unit` finished on `compute`; freed slots are refilled with upcoming units.
    pub fn release(&mut self, unit: usize, compute: &Stream) -> Result<()> {
        if let Some(c) = &mut self.comfy {
            if let Some(h) = c.held.remove(&unit) {
                c.drop_hold(unit, h, Some(compute))?;
            }
            return Ok(());
        }
        if let Some(&si) = self.slot_of.get(&unit) {
            self.slots[si].free.record(compute)?;
            self.slots[si].released = true;
            if self.streamed.len() > self.slots.len() {
                if self.lazy.is_none() {
                    while self.issue_next()? {}
                }
            }
        }
        Ok(())
    }

    /// The model is loaded for a node, as ComfyUI's `load_models_gpu`: top priority, and every unit may come back into
    /// VRAM. Anything still held from a failed run is let go. Does nothing under `--legacy-offload`.
    pub fn prioritize(&mut self) -> Result<()> {
        let Some(c) = &mut self.comfy else { return Ok(()) };
        for (u, h) in std::mem::take(&mut c.held) {
            c.drop_hold(u, h, None)?;
        }
        comfy_memory::prioritize(c.id)
    }

    /// Copy `unit` from the host to `dst` on the copy stream; returns the copy's event.
    fn copy_unit(&mut self, unit: usize, dst: *mut c_void) -> Result<Event> {
        if self.lazy.is_some() {
            return self.fill(unit, dst);
        }
        let size = self.layouts[unit].size;
        let h = self.stores[unit].host.as_ref().context("streamed unit without host copy")?;
        memcpy_h2d(dst, h.ptr(), size, &self.copy)?;
        let done = Event::new()?;
        done.record(&self.copy)?;
        self.copied_bytes += size;
        Ok(done)
    }

    /// `acquire` under ComfyUI's policy: fault the unit into VRAM, else copy it into a cast buffer.
    fn comfy_acquire(&mut self, unit: usize, compute: &Stream) -> Result<*mut c_void> {
        let id = self.comfy.as_ref().unwrap().id;
        comfy_memory::ensure_active(id)?;
        if self.lazy.is_some() {
            self.read_ahead(unit)?;
        }
        let (ptr, hold) = match comfy_memory::fault(id, unit)? {
            Some((ptr, valid)) => {
                if !valid {
                    let done = self.copy_unit(unit, ptr)?;
                    compute.wait(&done)?;
                    comfy_memory::mark_valid(id, unit)?;
                }
                (ptr, None)
            }
            None => {
                if let Some(l) = &self.lazy {
                    l.streams[unit].store(true, Ordering::Relaxed);
                }
                let (idx, ptr) = comfy_memory::cast_take(self.layouts[unit].size, &self.copy)?;
                let done = self.copy_unit(unit, ptr)?;
                compute.wait(&done)?;
                (ptr, Some(idx))
            }
        };
        let c = self.comfy.as_mut().unwrap();
        if let Some(old) = c.held.insert(unit, hold) {
            // acquired again without a release: the earlier use is over on this stream
            c.drop_hold(unit, old, Some(compute))?;
        }
        Ok(ptr)
    }
}
