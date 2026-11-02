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

use crate::cuda::{memcpy_d2d, memcpy_h2d, DevBuf, Event, PinnedBuf, Stream};
use crate::st::{f32_bytes, Dt, SafeTensors};
use anyhow::{bail, Result};
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex, RwLock};
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
                Src::Bytes(_) => continue,
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

    pub fn pack(&self, st: &SafeTensors, dst: &mut [u8]) -> Result<()> {
        for e in &self.entries {
            let d = &mut dst[e.off..e.off + e.len];
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
                        let b = st.bf16_bytes(n)?;
                        d[o..o + b.len()].copy_from_slice(&b);
                        o += b.len();
                    }
                }
                Src::F32(n, add) => {
                    let v: Vec<f32> = st.f32_vec(n, true)?.into_iter().map(|x| x + add).collect();
                    d.copy_from_slice(&f32_bytes(&v));
                }
                Src::Bytes(b) => d.copy_from_slice(b),
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

type Staged = (usize, u64, Result<PinnedBuf>);

/// A unit held in RAM, pinned unless there is no room for that.
enum Image {
    Pinned(PinnedBuf),
    Plain(Vec<u8>),
}

impl Image {
    fn new(size: usize) -> Self {
        PinnedBuf::alloc(size).map_or_else(|_| Image::Plain(vec![0; size]), Image::Pinned)
    }

    fn bytes(&mut self) -> &mut [u8] {
        match self {
            Image::Pinned(p) => p.as_mut_slice(),
            Image::Plain(v) => v,
        }
    }
}

/// What a lazy set adds: the readers and what they have staged.
struct Lazy {
    st: Arc<SafeTensors>,
    images: Arc<Vec<RwLock<Option<Image>>>>,
    keep: Arc<AtomicBool>,
    want: mpsc::Sender<(usize, u64)>,
    done: mpsc::Receiver<Staged>,
    free: mpsc::Sender<(PinnedBuf, Option<Event>)>,
    staged: HashMap<usize, PinnedBuf>,
    asked: HashSet<usize>,
    /// Resident units whose VRAM holds their data.
    filled: Vec<bool>,
    generation: u64,
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
    pub copied_bytes: usize,
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
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    if i >= n {
                        break;
                    }
                    if resident[i] {
                        let buf = free_rx.lock().unwrap().recv();
                        let Ok(mut buf) = buf else { break };
                        let r = layouts[i].pack(st, &mut buf.as_mut_slice()[..layouts[i].size]).map(|_| buf);
                        if done_tx.send((i, r)).is_err() {
                            break;
                        }
                    } else {
                        let r = PinnedBuf::alloc(layouts[i].size)
                            .and_then(|mut b| layouts[i].pack(st, b.as_mut_slice()).map(|_| b));
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
            copied_bytes: 0,
        };
        ws.prefetch_initial()?;
        Ok(ws)
    }

    /// A set that reads nothing until it is used. VRAM for the plan is taken now.
    pub fn lazy(name: &str, st: Arc<SafeTensors>, layouts: Vec<Layout>, plan: Plan, threads: usize) -> Result<Self> {
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
        let (free, free_rx) = mpsc::channel::<(PinnedBuf, Option<Event>)>();
        for _ in 0..STAGING {
            free.send((PinnedBuf::alloc(max_size)?, None))?;
        }
        let images: Arc<Vec<RwLock<Option<Image>>>> = Arc::new((0..n).map(|_| RwLock::new(None)).collect());
        let (want_rx, free_rx) = (Arc::new(Mutex::new(want_rx)), Arc::new(Mutex::new(free_rx)));
        let keep = Arc::new(AtomicBool::new(false));
        for _ in 0..threads.clamp(1, 3) {
            let (want_rx, free_rx, done_tx, st, layouts, images, keep) = (want_rx.clone(), free_rx.clone(), done_tx.clone(), st.clone(), layouts.clone(), images.clone(), keep.clone());
            std::thread::spawn(move || loop {
                let Ok((unit, generation)) = want_rx.lock().unwrap().recv() else { break };
                let Ok((mut buf, event)) = free_rx.lock().unwrap().recv() else { break };
                let r = (|| -> Result<()> {
                    if let Some(e) = event {
                        e.sync()?;
                    }
                    let size = layouts[unit].size;
                    let mut held = images[unit].write().unwrap();
                    match &mut *held {
                        Some(img) => buf.as_mut_slice()[..size].copy_from_slice(&img.bytes()[..size]),
                        None => {
                            layouts[unit].pack(&st, &mut buf.as_mut_slice()[..size])?;
                            if keep.load(Ordering::Relaxed) {
                                let mut img = Image::new(size);
                                img.bytes()[..size].copy_from_slice(&buf.as_mut_slice()[..size]);
                                *held = Some(img);
                            }
                        }
                    }
                    Ok(())
                })();
                if done_tx.send((unit, generation, r.map(|_| buf))).is_err() {
                    break;
                }
            });
        }
        let total: usize = layouts.iter().map(|l| l.size).sum();
        eprintln!("[{name}] {n} units, {:.2} GB: {} resident, {} streamed via {} slots, read on use", total as f64 / 1e9, plan.resident, streamed.len(), slots.len());
        let lazy = Lazy { st, images, keep, want, done, free, staged: HashMap::new(), asked: HashSet::new(), filled: vec![false; n], generation: 0 };
        Ok(WeightSet { name: name.to_string(), layouts, stores, rewritten: vec![false; n], streamed, slot_of: HashMap::new(), next_rank: 0, rr: 0, slots, copy: Stream::new()?, lazy: Some(lazy), copied_bytes: 0 })
    }

    /// Move a lazy set to `p`, freeing VRAM before taking any. What leaves VRAM is read again on use,
    /// LoRA bakes included. The compute stream must be idle. Keeps everything if `p` is the current plan.
    pub fn replan(&mut self, p: Plan) -> Result<()> {
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
        self.restart_ring()
    }

    /// Keep what is read in host memory, so a lazy set that returns needs no second trip to the file.
    pub fn keep_images(&self) {
        if let Some(l) = &self.lazy {
            l.keep.store(true, Ordering::Relaxed);
        }
    }

    /// `unit`'s image, if it can go to VRAM as it is.
    fn pinned(&self, unit: usize) -> Option<*mut u8> {
        match &*self.lazy.as_ref()?.images[unit].read().unwrap() {
            Some(Image::Pinned(p)) => Some(p.ptr()),
            _ => None,
        }
    }

    /// True when `unit`'s data will have to come from the host.
    fn needs_data(&self, unit: usize) -> bool {
        let Some(l) = &self.lazy else { return false };
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
            if self.needs_data(v) && self.pinned(v).is_none() {
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
        let direct = if self.lazy.as_ref().unwrap().asked.contains(&unit) { None } else { self.pinned(unit) };
        if let Some(p) = direct {
            memcpy_h2d(dst, p, size, &self.copy)?;
            let done = Event::new()?;
            done.record(&self.copy)?;
            self.copied_bytes += size;
            return Ok(done);
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
            } else {
                l.free.send((buf, None))?;
            }
        }
        let buf = l.staged.remove(&unit).unwrap();
        l.asked.remove(&unit);
        memcpy_h2d(dst, buf.ptr(), size, &self.copy)?;
        let done = Event::new()?;
        done.record(&self.copy)?;
        l.free.send((buf, Some(Event::new_like(&done, &self.copy)?)))?;
        self.copied_bytes += size;
        Ok(done)
    }

    pub fn len(&self) -> usize {
        self.layouts.len()
    }

    /// Number of units held in VRAM.
    pub fn resident(&self) -> usize {
        self.stores.iter().filter(|s| s.dev.is_some()).count()
    }

    /// VRAM held by this set (resident units + streaming slots).
    pub fn vram_bytes(&self) -> usize {
        let res: usize = self.stores.iter().filter_map(|s| s.dev.as_ref()).map(|d| d.size()).sum();
        res + self.slots.iter().map(|s| s.buf.size()).sum::<usize>()
    }

    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Move the residency boundary to `r` units (the first `r` in VRAM, the rest streamed through
    /// `nslots` ring slots). Frees VRAM before allocating it. The compute stream must be idle.
    pub fn set_resident(&mut self, r: usize, nslots: usize, st: &SafeTensors) -> Result<()> {
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
            l.asked.clear();
            for (_, buf) in l.staged.drain() {
                l.free.send((buf, None))?;
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
        if let Some(l) = &self.lazy {
            let mut image = l.images[unit].write().unwrap();
            if image.is_none() {
                let mut b = Image::new(self.layouts[unit].size);
                self.layouts[unit].pack(&l.st, &mut b.bytes()[..self.layouts[unit].size])?;
                *image = Some(b);
            }
            src.download(0, &mut image.as_mut().unwrap().bytes()[off..off + len], st)?;
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
        if self.lazy.is_some() {
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
}
