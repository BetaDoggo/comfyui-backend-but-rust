//! The prompt queue and history (ComfyUI's `PromptQueue`).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value as Json};

#[derive(Clone, Debug)]
pub struct QueueItem {
    pub number: i64,
    pub prompt_id: String,
    pub prompt: Json,
    pub extra_data: Json,
    pub outputs: Vec<String>,
    pub client_id: Option<String>,
}

impl QueueItem {
    /// ComfyUI's tuple form: `[number, prompt_id, prompt, extra_data, outputs_to_execute]`.
    pub fn to_json(&self) -> Json {
        json!([self.number, self.prompt_id, self.prompt, self.extra_data, self.outputs])
    }
}

#[derive(Clone, Debug)]
pub struct HistoryEntry {
    pub item: QueueItem,
    pub outputs: Map<String, Json>,
    pub status_str: String,
    pub completed: bool,
    pub messages: Vec<(String, Json)>,
    pub meta: Map<String, Json>,
}

impl HistoryEntry {
    pub fn to_json(&self) -> Json {
        json!({
            "prompt": self.item.to_json(),
            "outputs": self.outputs,
            "status": {
                "status_str": self.status_str,
                "completed": self.completed,
                "messages": self.messages.iter().map(|(k, v)| json!([k, v])).collect::<Vec<_>>(),
            },
            "meta": self.meta,
        })
    }
}

/// Work the worker thread should do besides running prompts.
#[derive(Clone, Copy, Debug, Default)]
pub struct FreeRequest {
    pub unload_models: bool,
    pub free_memory: bool,
}

pub enum Work {
    Prompt(QueueItem),
    Free(FreeRequest),
}

struct Inner {
    pending: Vec<QueueItem>,
    running: Option<QueueItem>,
    history: VecDeque<HistoryEntry>,
    next_number: i64,
    free: Option<FreeRequest>,
}

pub struct PromptQueue {
    inner: Mutex<Inner>,
    cv: Condvar,
    max_history: usize,
    interrupt: AtomicBool,
}

impl PromptQueue {
    pub fn new(max_history: usize) -> Self {
        Self {
            inner: Mutex::new(Inner { pending: Vec::new(), running: None, history: VecDeque::new(), next_number: 0, free: None }),
            cv: Condvar::new(),
            max_history,
            interrupt: AtomicBool::new(false),
        }
    }

    pub fn next_number(&self, front: bool) -> i64 {
        let mut g = self.inner.lock().unwrap();
        let n = g.next_number;
        g.next_number += 1;
        if front {
            -n
        } else {
            n
        }
    }

    pub fn push(&self, item: QueueItem) {
        let mut g = self.inner.lock().unwrap();
        let pos = g.pending.partition_point(|p| p.number <= item.number);
        g.pending.insert(pos, item);
        self.cv.notify_all();
    }

    /// Blocks until there is something to do.
    /// The next work, waiting up to `wait` for it (for ever without one).
    pub fn next_work(&self, wait: Option<Duration>) -> Option<Work> {
        let deadline = wait.map(|w| Instant::now() + w);
        let mut g = self.inner.lock().unwrap();
        loop {
            if let Some(f) = g.free.take() {
                return Some(Work::Free(f));
            }
            if !g.pending.is_empty() {
                let item = g.pending.remove(0);
                g.running = Some(item.clone());
                self.interrupt.store(false, Ordering::SeqCst);
                return Some(Work::Prompt(item));
            }
            g = match deadline {
                Some(d) => {
                    let left = d.saturating_duration_since(Instant::now());
                    if left.is_zero() {
                        return None;
                    }
                    self.cv.wait_timeout(g, left).unwrap().0
                }
                None => self.cv.wait(g).unwrap(),
            };
        }
    }

    pub fn finish(&self, entry: HistoryEntry) {
        let mut g = self.inner.lock().unwrap();
        g.running = None;
        if let Some(pos) = g.history.iter().position(|h| h.item.prompt_id == entry.item.prompt_id) {
            g.history.remove(pos);
        }
        g.history.push_back(entry);
        while g.history.len() > self.max_history {
            g.history.pop_front();
        }
    }

    pub fn request_free(&self, req: FreeRequest) {
        let mut g = self.inner.lock().unwrap();
        let cur = g.free.unwrap_or_default();
        g.free = Some(FreeRequest { unload_models: cur.unload_models || req.unload_models, free_memory: cur.free_memory || req.free_memory });
        self.cv.notify_all();
    }

    /// `POST /interrupt`: with a prompt id, only interrupts if that prompt is the one running.
    pub fn interrupt(&self, prompt_id: Option<&str>) {
        let g = self.inner.lock().unwrap();
        match (&g.running, prompt_id) {
            (Some(r), Some(id)) if r.prompt_id == id => self.interrupt.store(true, Ordering::SeqCst),
            (Some(_), None) => self.interrupt.store(true, Ordering::SeqCst),
            _ => {}
        }
    }

    pub fn is_interrupted(&self) -> bool {
        self.interrupt.load(Ordering::SeqCst)
    }

    pub fn delete_pending(&self, ids: &[String]) {
        self.inner.lock().unwrap().pending.retain(|p| !ids.contains(&p.prompt_id));
    }

    pub fn clear_pending(&self) {
        self.inner.lock().unwrap().pending.clear();
    }

    pub fn remaining(&self) -> usize {
        let g = self.inner.lock().unwrap();
        g.pending.len() + g.running.is_some() as usize
    }

    pub fn queue_json(&self) -> Json {
        let g = self.inner.lock().unwrap();
        json!({
            "queue_running": g.running.iter().map(QueueItem::to_json).collect::<Vec<_>>(),
            "queue_pending": g.pending.iter().map(QueueItem::to_json).collect::<Vec<_>>(),
        })
    }

    pub fn history_json(&self, prompt_id: Option<&str>, max_items: Option<usize>, offset: Option<usize>) -> Json {
        let g = self.inner.lock().unwrap();
        let mut out = Map::new();
        match prompt_id {
            Some(id) => {
                if let Some(h) = g.history.iter().find(|h| h.item.prompt_id == id) {
                    out.insert(id.to_string(), h.to_json());
                }
            }
            None => {
                let total = g.history.len();
                let max = max_items.unwrap_or(total);
                let start = offset.unwrap_or_else(|| total.saturating_sub(max));
                for h in g.history.iter().skip(start).take(max) {
                    out.insert(h.item.prompt_id.clone(), h.to_json());
                }
            }
        }
        Json::Object(out)
    }

    pub fn delete_history(&self, ids: &[String]) {
        self.inner.lock().unwrap().history.retain(|h| !ids.contains(&h.item.prompt_id));
    }

    pub fn clear_history(&self) {
        self.inner.lock().unwrap().history.clear();
    }
}
