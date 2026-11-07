//! Context windows on MiniMax H3 (`MiniMaxH3ContextWindows`, SwarmUI's Video Context Window):
//! each model evaluation runs on overlapping stretches of the clip and blends their denoised
//! outputs, as ComfyUI's `IndexListContextHandler` does with the static standard schedule.
//! Windows hold whole video latents; the audio rows that share their time span go with them.
//! FreeNoise shuffles the initial noise so the windows start from related noise.

use crate::h3::{self, Shape, FRAME_PER_TOKEN};
use crate::sampling::{torch_randperm, Mt19937};
use anyhow::{bail, Result};
use std::ops::Range;

const VCH: usize = 24;
const ACH: usize = 32;

/// `ContextFuseMethods`: how the windows' outputs are blended where they overlap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fuse {
    Pyramid,
    Relative,
    Flat,
    OverlapLinear,
}

impl Fuse {
    pub const NAMES: [&'static str; 4] = ["pyramid", "relative", "flat", "overlap-linear"];

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "pyramid" => Fuse::Pyramid,
            "relative" => Fuse::Relative,
            "flat" => Fuse::Flat,
            "overlap-linear" => Fuse::OverlapLinear,
            _ => return None,
        })
    }
}

/// The settings, in video latents.
#[derive(Clone, Debug, PartialEq)]
pub struct ContextWindows {
    pub length: usize,
    pub overlap: usize,
    pub fuse: Fuse,
    pub freenoise: bool,
}

/// One window: its video latents and the audio latents of the same time span.
#[derive(Clone, Debug, PartialEq)]
pub struct Window {
    pub video: Range<usize>,
    pub audio: Range<usize>,
}

impl Window {
    /// The shape the model sees for this window.
    pub fn shape(&self, sh: &Shape) -> Shape {
        Shape { t: self.video.len(), h: sh.h, w: sh.w, audio_t: self.audio.len() }
    }
}

/// The pixel frame each video latent starts at, plus the end of the last one.
fn pixel_starts(t: usize) -> Vec<usize> {
    let mut s = vec![0];
    for k in 0..t {
        s.push(s[k] + FRAME_PER_TOKEN[k % 5]);
    }
    s
}

/// Python's `round`, half to even.
fn round(x: f64) -> usize {
    x.round_ties_even().max(0.0) as usize
}

impl ContextWindows {
    /// From the node's lengths in frames, as `MiniMaxH3ContextWindows.execute` converts them: a
    /// length of 5n + 2 latents and an overlap at most 5 below it start every window on the
    /// model's 5-latent chunks. `length` must be at least 22 frames (the node's minimum).
    pub fn from_frames(length: usize, overlap: usize, fuse: Fuse, freenoise: bool) -> Result<Self> {
        if length < 22 {
            bail!("a context length of {length} frames (at least 22)");
        }
        let length = h3::video_latent_t(length);
        let overlap = h3::video_latent_t(overlap).min(length - 5);
        Ok(ContextWindows { length, overlap, fuse, freenoise })
    }

    /// True when the clip is longer than one window. Otherwise sampling runs as without windows.
    pub fn applies(&self, sh: &Shape) -> bool {
        sh.t > self.length
    }

    /// `create_windows_static_standard` over the video latents, each with its audio as
    /// `map_context_window_to_modalities` splits it: at the video latents' pixel frame boundaries.
    pub fn windows(&self, sh: &Shape) -> Vec<Window> {
        let t = sh.t;
        let video: Vec<Range<usize>> = if t <= self.length {
            vec![0..t]
        } else {
            let delta = self.length - self.overlap;
            let mut v = Vec::new();
            let mut start = 0;
            while start < t {
                let end = start + self.length;
                if end >= t {
                    // the last window moves back to end with the clip
                    v.push(t - self.length..t);
                    break;
                }
                v.push(start..end);
                start += delta;
            }
            v
        };
        let starts = pixel_starts(t);
        let total = starts[t] as f64;
        let edge = |k: usize| round((starts[k] * sh.audio_t) as f64 / total);
        video.into_iter().map(|v| Window { audio: edge(v.start)..edge(v.end), video: v }).collect()
    }

    /// The audio's length and overlap: the video's, scaled by the streams' latent ratio.
    fn audio_settings(&self, sh: &Shape) -> (usize, usize) {
        let ratio = sh.audio_t as f64 / sh.t as f64;
        (round(self.length as f64 * ratio).max(1), round(self.overlap as f64 * ratio))
    }

    /// FreeNoise (`_apply_freenoise`): after each window's span, the next stretch of noise is a
    /// shuffle of the stretch one window back. The video and audio are shuffled separately, each
    /// from a CPU generator seeded with `seed`. `noise` is the pack, video then audio.
    pub fn freenoise(&self, noise: &mut [f32], sh: &Shape, seed: u64) {
        if !self.freenoise {
            return;
        }
        let (video, audio) = noise.split_at_mut(sh.video_len());
        // [C, T, H*W]: a time index moves C runs of H*W values
        shuffle(video, VCH, sh.t, sh.h * sh.w, self.length, self.overlap, seed);
        let (al, ao) = self.audio_settings(sh);
        // [C*2, A]
        shuffle(audio, ACH * 2, sh.audio_t, 1, al, ao, seed);
    }

    /// The weight of each position of a window (`get_context_weights`). `span` is the window's
    /// range in a stream of `full` latents, and `overlap` that stream's overlap.
    fn weights(&self, span: &Range<usize>, full: usize, overlap: usize) -> Vec<f32> {
        let len = span.len();
        match self.fuse {
            Fuse::Flat => vec![1.0; len],
            Fuse::Pyramid | Fuse::Relative => {
                let half = len.div_ceil(2);
                (0..len).map(|i| (i + 1).min(len - i).min(half) as f32).collect()
            }
            Fuse::OverlapLinear => {
                let mut w = vec![1.0f32; len];
                let ov = overlap.min(len);
                if span.start > 0 {
                    w[..ov].copy_from_slice(&linspace(1e-37, 1.0, ov));
                }
                if span.end < full {
                    w[len - ov..].copy_from_slice(&linspace(1.0, 1e-37, ov));
                }
                w
            }
        }
    }

    /// One model evaluation over windows: `predict` takes a window's pack and shape and returns
    /// its denoised pack. Returns the fused denoised pack of the whole clip.
    pub fn run(&self, x: &[f32], sh: &Shape, mut predict: impl FnMut(&[f32], &Shape) -> Result<Vec<f32>>) -> Result<Vec<f32>> {
        let (vn, an) = (sh.video_len(), sh.audio_len());
        if x.len() != vn + an {
            bail!("a pack of {} values for {sh:?}", x.len());
        }
        let hw = sh.h * sh.w;
        let windows = self.windows(sh);
        let (_, audio_overlap) = self.audio_settings(sh);
        let mut fused = vec![0f32; vn + an];
        // per latent of each stream: total weight, or the relative method's running bias
        let (mut vcount, mut acount) = (vec![0f32; sh.t], vec![0f32; sh.audio_t]);
        let (mut vbias, mut abias) = (vec![0f64; sh.t], vec![0f64; sh.audio_t]);
        for w in &windows {
            let ws = w.shape(sh);
            let mut pack = Vec::with_capacity(ws.video_len() + ws.audio_len());
            for c in 0..VCH {
                pack.extend_from_slice(&x[(c * sh.t + w.video.start) * hw..(c * sh.t + w.video.end) * hw]);
            }
            for r in 0..ACH * 2 {
                pack.extend_from_slice(&x[vn + r * sh.audio_t + w.audio.start..vn + r * sh.audio_t + w.audio.end]);
            }
            let out = predict(&pack, &ws)?;
            if out.len() != pack.len() {
                bail!("a window's prediction has {} values, not {}", out.len(), pack.len());
            }
            let (vo, ao) = out.split_at(ws.video_len());
            // (rows, run, stream offset, stream length, window, window output, counts, biases, overlap)
            let streams = [
                (VCH, hw, 0, sh.t, &w.video, vo, &mut vcount, &mut vbias, self.overlap),
                (ACH * 2, 1, vn, sh.audio_t, &w.audio, ao, &mut acount, &mut abias, audio_overlap),
            ];
            for (rows, run, base, full, span, out, count, bias, overlap) in streams {
                let len = span.len();
                if self.fuse == Fuse::Relative {
                    // a running average weighted by each position's closeness to the window's centre
                    let (first, last) = (span.start as f64, (span.end - 1) as f64);
                    for (pos, idx) in span.clone().enumerate() {
                        let b = (1.0 - (idx as f64 - (first + last) / 2.0).abs() / ((last - first + 1e-2) / 2.0)).max(1e-2);
                        let total = bias[idx];
                        let (prev, new) = ((total / (total + b)) as f32, (b / (total + b)) as f32);
                        for r in 0..rows {
                            let dst = &mut fused[base + (r * full + idx) * run..][..run];
                            let src = &out[(r * len + pos) * run..][..run];
                            dst.iter_mut().zip(src).for_each(|(d, s)| *d = *d * prev + s * new);
                        }
                        bias[idx] = total + b;
                    }
                } else {
                    let wt = self.weights(span, full, overlap);
                    for (pos, idx) in span.clone().enumerate() {
                        for r in 0..rows {
                            let dst = &mut fused[base + (r * full + idx) * run..][..run];
                            let src = &out[(r * len + pos) * run..][..run];
                            dst.iter_mut().zip(src).for_each(|(d, s)| *d += s * wt[pos]);
                        }
                        count[idx] += wt[pos];
                    }
                }
            }
        }
        if self.fuse != Fuse::Relative {
            for (rows, run, base, full, count) in [(VCH, hw, 0, sh.t, &vcount), (ACH * 2, 1, vn, sh.audio_t, &acount)] {
                for r in 0..rows {
                    for (idx, c) in count.iter().enumerate() {
                        fused[base + (r * full + idx) * run..][..run].iter_mut().for_each(|v| *v /= c);
                    }
                }
            }
        }
        Ok(fused)
    }
}

/// `torch.linspace` in float32.
fn linspace(start: f32, end: f32, steps: usize) -> Vec<f32> {
    if steps == 1 {
        return vec![start];
    }
    let step = (end - start) / (steps - 1) as f32;
    (0..steps).map(|i| if i < steps / 2 { start + step * i as f32 } else { end - step * (steps - 1 - i) as f32 }).collect()
}

/// `apply_freenoise` along the time axis of `x`, laid out [rows, t, run].
fn shuffle(x: &mut [f32], rows: usize, t: usize, run: usize, length: usize, overlap: usize, seed: u64) {
    let mut g = Mt19937::new(seed);
    let delta = length.saturating_sub(overlap);
    if delta == 0 || t <= length {
        return;
    }
    let mut start = 0;
    while start < t - length {
        let place = start + length;
        let n = delta.min(t - place);
        if n == 0 {
            break;
        }
        let perm = torch_randperm(&mut g, n);
        for r in 0..rows {
            let row = &mut x[r * t * run..(r + 1) * t * run];
            let src: Vec<f32> = perm.iter().flat_map(|&p| row[(start + p) * run..(start + p + 1) * run].to_vec()).collect();
            row[place * run..(place + n) * run].copy_from_slice(&src);
        }
        start += delta;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cw(length: usize, overlap: usize, fuse: Fuse) -> ContextWindows {
        ContextWindows::from_frames(length, overlap, fuse, true).unwrap()
    }

    #[test]
    fn frames_to_latents() {
        // the node's defaults, and SwarmUI's
        assert_eq!((cw(124, 39, Fuse::Pyramid).length, cw(124, 39, Fuse::Pyramid).overlap), (37, 12));
        assert_eq!((cw(120, 24, Fuse::Pyramid).length, cw(120, 24, Fuse::Pyramid).overlap), (32, 7));
        // the overlap stays 5 below the length
        assert_eq!(cw(22, 22, Fuse::Pyramid).overlap, 2);
        assert!(ContextWindows::from_frames(21, 5, Fuse::Pyramid, true).is_err());
    }

    #[test]
    fn windows_start_on_chunks() {
        // 260 frames: 77 latents, 433 audio latents
        let sh = Shape::new(256, 256, 260);
        assert_eq!((sh.t, sh.audio_t), (77, 433));
        let w = cw(124, 39, Fuse::Pyramid).windows(&sh);
        let video: Vec<_> = w.iter().map(|w| w.video.clone()).collect();
        assert_eq!(video, vec![0..37, 25..62, 40..77]);
        assert!(w.iter().all(|w| w.video.start % 5 == 0));
        // audio at the video latents' pixel frame boundaries, recorded from map_context_window_to_modalities
        let audio: Vec<_> = w.iter().map(|w| w.audio.clone()).collect();
        assert_eq!(audio, vec![0..207, 142..348, 226..433]);
        // one window when the clip fits
        assert_eq!(cw(124, 39, Fuse::Pyramid).windows(&Shape::new(256, 256, 124)).len(), 1);
        assert!(!cw(124, 39, Fuse::Pyramid).applies(&Shape::new(256, 256, 124)));
    }

    #[test]
    fn weights_match_comfy() {
        let c = cw(124, 39, Fuse::Pyramid);
        assert_eq!(c.weights(&(0..4), 10, 2), vec![1.0, 2.0, 2.0, 1.0]);
        assert_eq!(c.weights(&(0..5), 10, 2), vec![1.0, 2.0, 3.0, 2.0, 1.0]);
        let o = cw(124, 39, Fuse::OverlapLinear);
        assert_eq!(o.weights(&(0..5), 10, 2), vec![1.0, 1.0, 1.0, 1.0, 1e-37]);
        assert_eq!(o.weights(&(5..10), 10, 3), vec![1e-37, 0.5, 1.0, 1.0, 1.0]);
    }

    /// Every fuse method blends identical windows back to the input.
    #[test]
    fn identity_round_trips() {
        let sh = Shape::new(64, 64, 260);
        let x: Vec<f32> = (0..sh.video_len() + sh.audio_len()).map(|i| (i % 97) as f32 * 0.25 - 3.0).collect();
        for fuse in [Fuse::Pyramid, Fuse::Relative, Fuse::Flat, Fuse::OverlapLinear] {
            let c = cw(56, 22, fuse);
            let mut calls = 0;
            let out = c.run(&x, &sh, |p, ws| {
                calls += 1;
                assert_eq!(p.len(), ws.video_len() + ws.audio_len());
                Ok(p.to_vec())
            }).unwrap();
            assert!(calls > 2);
            for (a, b) in out.iter().zip(&x) {
                assert!((a - b).abs() <= 1e-5 * b.abs().max(1.0), "{fuse:?}: {a} vs {b}");
            }
        }
    }

    /// `apply_freenoise` on torch.arange, recorded from ComfyUI's code (torch 2.13, CPU generator).
    #[test]
    fn freenoise_matches_torch() {
        // [1, 1, 20, 1, 1] on dim 2, length 7, overlap 2, seed 42
        let mut x: Vec<f32> = (0..20).map(|i| i as f32).collect();
        shuffle(&mut x, 1, 20, 1, 7, 2, 42);
        let want = [0, 1, 2, 3, 4, 5, 6, 2, 4, 3, 0, 1, 6, 3, 4, 2, 5, 1, 6, 0];
        assert_eq!(x, want.map(|v| v as f32));
        // [1, 3, 2, 50] on dim 3, length 13, overlap 4, seed 7: every row takes the same shuffle
        let mut y: Vec<f32> = (0..300).map(|i| i as f32).collect();
        shuffle(&mut y, 6, 50, 1, 13, 4, 7);
        let want = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 5, 8, 1, 7, 2, 3, 6, 4, 0, 11, 5, 9, 1, 8, 7, 12, 10, 2, 3, 0, 4, 11, 6, 1, 5, 9, 4, 7,
            12, 2, 10, 8, 0, 3, 11, 6];
        assert_eq!(&y[..50], &want.map(|v| v as f32));
        assert_eq!(&y[250..], &want.map(|v| (v + 250) as f32));
    }
}
