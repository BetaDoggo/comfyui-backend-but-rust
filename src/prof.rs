//! Per-stage timing, on with `PROFILE=1`. Stages are timed by events on the compute stream, so the run
//! stays asynchronous and copies still overlap compute; a stage that waited on a copy shows that wait.

use crate::cuda::{Event, Stream};
use anyhow::Result;
use std::time::Instant;

#[derive(Default)]
pub struct Prof {
    on: bool,
    marks: Vec<(&'static str, Event)>,
    host: Vec<(&'static str, f64)>,
    start: Option<Instant>,
}

impl Prof {
    pub fn new() -> Self {
        Prof { on: std::env::var_os("PROFILE").is_some(), ..Self::default() }
    }

    pub fn on(&self) -> bool {
        self.on
    }

    /// The compute stream's work since the last mark belongs to `stage`.
    pub fn mark(&mut self, stage: &'static str, st: &Stream) -> Result<()> {
        if self.on {
            let e = Event::timing()?;
            e.record(st)?;
            self.marks.push((stage, e));
            self.start.get_or_insert_with(Instant::now);
        }
        Ok(())
    }

    /// Host time since `since`, added to `stage`.
    pub fn host(&mut self, stage: &'static str, since: Instant) {
        if self.on {
            self.host.push((stage, since.elapsed().as_secs_f64() * 1e3));
        }
    }

    /// Print what was marked since the last report.
    pub fn report(&mut self, label: &str, st: &Stream) -> Result<()> {
        if !self.on || self.marks.is_empty() {
            return Ok(());
        }
        st.sync()?;
        let mut gpu: Vec<(&str, f64)> = Vec::new();
        for w in self.marks.windows(2) {
            let ms = w[1].1.elapsed_ms(&w[0].1)? as f64;
            match gpu.iter_mut().find(|g| g.0 == w[1].0) {
                Some(g) => g.1 += ms,
                None => gpu.push((w[1].0, ms)),
            }
        }
        let total: f64 = gpu.iter().map(|g| g.1).sum();
        gpu.sort_by(|a, b| b.1.total_cmp(&a.1));
        let wall = self.start.take().map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e3);
        let parts: Vec<String> = gpu.iter().map(|(n, ms)| format!("{n} {ms:.0}")).collect();
        let host: Vec<String> = self.host.iter().map(|(n, ms)| format!("{n} {ms:.0}")).collect();
        eprintln!("[prof] {label}: gpu {total:.0} ms, wall {wall:.0} ms: {} | host: {}", parts.join(", "), host.join(", "));
        self.marks.clear();
        self.host.clear();
        Ok(())
    }
}
