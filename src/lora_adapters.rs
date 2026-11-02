//! ComfyUI's LoRA weight adapters (`comfy.weight_adapter`) beyond a plain up @ down: each returns the
//! change to a weight, on the CPU except for matrix products, which go through `Mm`. ComfyUI computes
//! them in the weight's compute dtype, so results are rounded as they go, by `Work`; OFT and BOFT, which
//! ComfyUI can only compute in float32, are not.

use anyhow::{bail, ensure, Result};
use half::bf16;

/// The compute dtype: its rounding, and the `torch.finfo().eps` that `weight_decompose` adds to norms.
#[derive(Clone, Copy)]
pub struct Work {
    round: fn(f32) -> f32,
    eps: f32,
}

impl Work {
    pub const BF16: Work = Work { round: |x| bf16::from_f32(x).to_f32(), eps: 0.0078125 };
    pub const F32: Work = Work { round: |x| x, eps: f32::EPSILON };
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// `f(i, item)` for each `per`-long item of `out`, on every core: these tensors reach 100M values.
pub fn par_items<T: Send>(out: &mut [T], per: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let block = (out.len() / per / threads + 1) * per;
    std::thread::scope(|s| {
        for (b, chunk) in out.chunks_mut(block).enumerate() {
            let f = &f;
            s.spawn(move || chunk.chunks_exact_mut(per).enumerate().for_each(|(i, item)| f(b * block / per + i, item)));
        }
    });
}

/// Matrix product of two 2D tensors.
pub type Mm<'a> = dyn FnMut(&Tensor, &Tensor) -> Result<Tensor> + 'a;

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        Tensor { shape, data }
    }

    pub fn rows(&self) -> usize {
        self.shape.first().copied().unwrap_or(1)
    }

    /// `flatten(start_dim=1)`
    pub fn flat(&self) -> Tensor {
        let rows = self.rows().max(1);
        Tensor::new(vec![rows, self.data.len() / rows], self.data.clone())
    }

    /// `transpose(0, 1)`
    fn t01(&self) -> Tensor {
        let (a, b) = (self.shape[0], self.shape[1]);
        let rest: usize = self.shape[2..].iter().product();
        let mut data = vec![0.0; self.data.len()];
        for i in 0..a {
            for j in 0..b {
                let (from, to) = ((i * b + j) * rest, (j * a + i) * rest);
                data[to..to + rest].copy_from_slice(&self.data[from..from + rest]);
            }
        }
        let mut shape = self.shape.clone();
        shape.swap(0, 1);
        Tensor::new(shape, data)
    }

    fn reshaped(self, shape: Vec<usize>) -> Result<Tensor> {
        ensure!(shape.iter().product::<usize>() == self.data.len(), "can't reshape {:?} to {shape:?}", self.shape);
        Ok(Tensor::new(shape, self.data))
    }
}

/// The product of the flattened `a` and `b`, checked.
fn mm(a: &Tensor, b: &Tensor, f: &mut Mm) -> Result<Tensor> {
    let (a, b) = (a.flat(), b.flat());
    ensure!(a.shape[1] == b.shape[0], "can't multiply {:?} by {:?}", a.shape, b.shape);
    f(&a, &b)
}

/// `einsum("i j k l, j r, i p -> p r k l", t, b, a)`, flattened after `p`.
pub fn tucker(t: &Tensor, b: &Tensor, a: &Tensor, f: &mut Mm) -> Result<Tensor> {
    let &[ti, tj, tk, tl] = t.shape.as_slice() else { bail!("a Tucker core of shape {:?}", t.shape) };
    let (at, b) = (a.flat().t01(), b.flat());
    ensure!(at.shape[1] == ti && b.shape[0] == tj, "Tucker core {:?} doesn't fit {:?} and {:?}", t.shape, a.shape, b.shape);
    let (p, r) = (at.shape[0], b.shape[1]);
    let mut out = vec![0.0; p * r * tk * tl];
    for k in 0..tk {
        for l in 0..tl {
            let slice = (0..ti * tj).map(|n| t.data[n * tk * tl + k * tl + l]).collect();
            let m = mm(&mm(&at, &Tensor::new(vec![ti, tj], slice), f)?, &b, f)?;
            for x in 0..p {
                for y in 0..r {
                    out[(x * r + y) * tk * tl + k * tl + l] = m.data[x * r + y];
                }
            }
        }
    }
    Ok(Tensor::new(vec![p, r * tk * tl], out))
}

/// LoCon: `up @ down` with `down` passed through its mid weight.
pub fn lora_mid(up: &Tensor, down: &Tensor, mid: &Tensor, f: &mut Mm) -> Result<Tensor> {
    ensure!(mid.shape.len() == 4, "a mid weight of shape {:?}", mid.shape);
    let m = mm(&down.t01(), &mid.t01(), f)?;
    let down = m.reshaped(vec![down.shape[1], down.shape[0], mid.shape[2], mid.shape[3]])?.t01();
    mm(up, &down, f)
}

/// LoHa, without alpha.
pub fn loha(w1: (&Tensor, &Tensor), w2: (&Tensor, &Tensor), t: Option<(&Tensor, &Tensor)>, work: Work, f: &mut Mm) -> Result<Tensor> {
    let (m1, m2) = match t {
        Some((t1, t2)) => (tucker(t1, w1.1, w1.0, f)?, tucker(t2, w2.1, w2.0, f)?),
        None => (mm(w1.0, w1.1, f)?, mm(w2.0, w2.1, f)?),
    };
    ensure!(m1.data.len() == m2.data.len(), "LoHa halves {:?} and {:?} differ", m1.shape, m2.shape);
    let mut data = vec![0.0; m1.data.len()];
    par_items(&mut data, 1, |i, o| o[0] = (work.round)(m1.data[i] * m2.data[i]));
    Ok(Tensor::new(m1.shape, data))
}

/// `torch.kron`; a 4D `w2` flattens to the same values, since `w1` gains 1x1 trailing dimensions.
pub fn kron(a: &Tensor, b: &Tensor, work: Work) -> Tensor {
    let (a, b) = (a.flat(), b.flat());
    let (ar, ac, br, bc) = (a.shape[0], a.shape[1], b.shape[0], b.shape[1]);
    let mut out = vec![0.0; a.data.len() * b.data.len()];
    par_items(&mut out, ac * bc, |r, row| {
        let (i, k) = (r / br, r % br);
        for j in 0..ac {
            let v = a.data[i * ac + j];
            for (o, x) in row[j * bc..][..bc].iter_mut().zip(&b.data[k * bc..][..bc]) {
                *o = (work.round)(v * x);
            }
        }
    });
    Tensor::new(vec![ar * br, ac * bc], out)
}

/// The two forms of GLoRA, and its rank: lycoris's old files read `(w a2) a1 + b2 b1`.
pub fn glora(a1: &Tensor, a2: &Tensor, b1: &Tensor, b2: &Tensor, w: &Tensor, work: Work, f: &mut Mm) -> Result<(Tensor, usize)> {
    let mut old = false;
    let mut rank = 0;
    if b2.shape[1] == b1.shape[0] && b1.shape[0] == a1.shape[0] && a1.shape[0] == a2.shape[1] {
        rank = a1.shape[0];
        old = true;
    }
    if b2.shape[0] == b1.shape[1] && b1.shape[1] == a1.shape[1] && a1.shape[1] == a2.shape[0] && !(old && a2.shape[0] == w.shape[0] && w.shape[0] == w.shape[1]) {
        old = false;
        rank = a2.shape[0];
    }
    ensure!(rank > 0, "GLoRA matrices {:?} {:?} {:?} {:?} don't fit either form", a1.shape, a2.shape, b1.shape, b2.shape);
    let (first, second) = if old { (mm(b2, b1, f)?, mm(&mm(w, a2, f)?, a1, f)?) } else { (mm(&mm(w, a1, f)?, a2, f)?, mm(b1, b2, f)?) };
    ensure!(first.data.len() == second.data.len(), "GLoRA terms {:?} and {:?} differ", first.shape, second.shape);
    let data = first.data.iter().zip(&second.data).map(|(a, b)| (work.round)(a + b)).collect();
    Ok((Tensor::new(first.shape, data), rank))
}

/// Gauss-Jordan with partial pivoting, for `(I - Q)^-1`.
fn inverse(m: &[f32], n: usize) -> Result<Vec<f32>> {
    let mut a = m.to_vec();
    let mut b = vec![0.0; n * n];
    for i in 0..n {
        b[i * n + i] = 1.0;
    }
    for c in 0..n {
        let p = (c..n).max_by(|&x, &y| a[x * n + c].abs().total_cmp(&a[y * n + c].abs())).unwrap();
        ensure!(a[p * n + c] != 0.0, "a singular OFT block");
        for j in 0..n {
            a.swap(c * n + j, p * n + j);
            b.swap(c * n + j, p * n + j);
        }
        let d = a[c * n + c];
        for j in 0..n {
            a[c * n + j] /= d;
            b[c * n + j] /= d;
        }
        for r in (0..n).filter(|&r| r != c) {
            let x = a[r * n + c];
            for j in 0..n {
                a[r * n + j] -= x * a[c * n + j];
                b[r * n + j] -= x * b[c * n + j];
            }
        }
    }
    Ok(b)
}

/// The orthogonal blocks `(I + Q)(I - Q)^-1` of `blocks` `[.., n, n]` with Q = blocks - blocks^T, whose
/// norm over all of them `alpha` limits when positive.
fn orthogonal(blocks: &Tensor, alpha: f32) -> Result<Vec<f32>> {
    let n = blocks.shape[blocks.shape.len() - 1];
    let count = blocks.data.len() / (n * n);
    let mut q = vec![0.0; blocks.data.len()];
    for c in 0..count {
        for i in 0..n {
            for j in 0..n {
                q[c * n * n + i * n + j] = blocks.data[c * n * n + i * n + j] - blocks.data[c * n * n + j * n + i];
            }
        }
    }
    if alpha > 0.0 {
        let norm = q.iter().map(|x| x * x).sum::<f32>().sqrt() + 1e-8;
        if norm > alpha {
            q.iter_mut().for_each(|x| *x *= alpha / norm);
        }
    }
    let mut out = Vec::with_capacity(q.len());
    for b in q.chunks_exact(n * n) {
        let (mut plus, mut minus) = (Vec::with_capacity(n * n), Vec::with_capacity(n * n));
        for i in 0..n * n {
            let eye = if i % (n + 1) == 0 { 1.0 } else { 0.0 };
            plus.push(eye + b[i]);
            minus.push(eye - b[i]);
        }
        let inv = inverse(&minus, n)?;
        for i in 0..n {
            for j in 0..n {
                out.push((0..n).map(|k| plus[i * n + k] * inv[k * n + j]).sum());
            }
        }
    }
    Ok(out)
}

/// OFT, with `strength` already in it as ComfyUI has it: blocks `[count, n, n]` over `w`'s rows.
pub fn oft(blocks: &Tensor, alpha: f32, strength: f32, w: &Tensor, f: &mut Mm) -> Result<Tensor> {
    let &[count, n, m] = blocks.shape.as_slice() else { bail!("OFT blocks of shape {:?}", blocks.shape) };
    ensure!(n == m && count * n == w.rows(), "OFT blocks {:?} don't fit a weight of {:?}", blocks.shape, w.shape);
    let r = orthogonal(blocks, alpha)?;
    let cols = w.shape[1];
    let mut out = Vec::with_capacity(w.data.len());
    for k in 0..count {
        // sum_n D[k, n, m] W[k, n]: the transpose of D, then W
        let mut d = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let eye = if i == j { 1.0 } else { 0.0 };
                d[j * n + i] = strength * r[k * n * n + i * n + j] - strength * eye;
            }
        }
        let block = Tensor::new(vec![n, cols], w.data[k * n * cols..(k + 1) * n * cols].to_vec());
        out.extend(mm(&Tensor::new(vec![n, n], d), &block, f)?.data);
    }
    Ok(Tensor::new(w.shape.clone(), out))
}

/// BOFT's butterfly: blocks `[stages, count, b, b]` applied to `w`'s rows. `rescale` then scales the
/// result by row or by column.
pub fn boft(blocks: &Tensor, rescale: Option<&Tensor>, alpha: f32, strength: f32, w: &Tensor) -> Result<Tensor> {
    let &[stages, count, b, _] = blocks.shape.as_slice() else { bail!("BOFT blocks of shape {:?}", blocks.shape) };
    let r = orthogonal(blocks, alpha)?;
    let (rows, cols) = (w.rows(), w.shape[1]);
    let half = b / 2;
    let mut x = w.data.clone();
    for i in 0..stages {
        let k = (1 << i) * half;
        ensure!(rows % (2 * k) == 0 && rows % b == 0, "BOFT blocks {:?} don't fit a weight of {:?}", blocks.shape, w.shape);
        // rows (a, g, k') swap to (a, k', g) before the blocks and back after
        let order: Vec<usize> = (0..rows / (2 * k)).flat_map(|a| (0..k).flat_map(move |y| (0..2).map(move |g| (a * 2 + g) * k + y))).collect();
        let mut next = vec![0.0; x.len()];
        for bn in 0..rows / b {
            let at = |c: usize| if count == 1 { 0 } else { c };
            for ri in 0..b {
                let to = order[bn * b + ri];
                let row = &mut next[to * cols..][..cols];
                for rj in 0..b {
                    let eye = if ri == rj { 1.0 } else { 0.0 };
                    let v = r[(i * count + at(bn)) * b * b + ri * b + rj];
                    let v = if strength != 1.0 { v * strength + (1.0 - strength) * eye } else { v };
                    let from = order[bn * b + rj];
                    for (o, y) in row.iter_mut().zip(&x[from * cols..][..cols]) {
                        *o += v * y;
                    }
                }
            }
        }
        x = next;
    }
    if let Some(s) = rescale {
        match s.data.len() {
            1 => x.iter_mut().for_each(|v| *v *= s.data[0]),
            n if n == rows => x.chunks_exact_mut(cols).zip(&s.data).for_each(|(row, v)| row.iter_mut().for_each(|y| *y *= v)),
            n if n == cols => x.chunks_exact_mut(cols).for_each(|row| row.iter_mut().zip(&s.data).for_each(|(y, v)| *y *= v)),
            _ => bail!("a rescale of shape {:?}", s.shape),
        }
    }
    x.iter_mut().zip(&w.data).for_each(|(a, o)| *a -= o);
    Ok(Tensor::new(w.shape.clone(), x))
}

/// DoRA (`weight_decompose`): `w` afterwards. `scale` is `dora_scale`, by row if it has one value per
/// row, else by column.
pub fn dora(scale: &Tensor, w: &Tensor, diff: &[f32], alpha: f32, strength: f32, work: Work) -> Result<Vec<f32>> {
    let rb = work.round;
    let (rows, cols) = (w.rows(), w.shape[1]);
    let calc: Vec<f32> = w.data.iter().zip(diff).map(|(a, d)| rb(a + rb(alpha * d))).collect();
    let by_row = scale.rows() == rows;
    ensure!(scale.data.len() == if by_row { rows } else { cols }, "a dora_scale of shape {:?} for a weight of {:?}", scale.shape, w.shape);
    let mut factor = vec![0.0f32; scale.data.len()];
    if by_row {
        for (i, row) in w.data.chunks_exact(cols).enumerate() {
            factor[i] = row.iter().map(|x| x * x).sum();
        }
    } else {
        for row in calc.chunks_exact(cols) {
            row.iter().zip(&mut factor).for_each(|(x, n)| *n += x * x);
        }
    }
    factor.iter_mut().zip(&scale.data).for_each(|(n, s)| *n = rb(s / rb(rb(n.sqrt()) + work.eps)));
    let scaled = calc.iter().enumerate().map(|(i, x)| rb(x * factor[if by_row { i / cols } else { i % cols }]));
    Ok(if strength != 1.0 {
        scaled.zip(&w.data).map(|(x, o)| rb(o + rb(strength * rb(x - o)))).collect()
    } else {
        scaled.collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let (r, n, c) = (a.shape[0], a.shape[1], b.shape[1]);
        let mut d = vec![0.0; r * c];
        for i in 0..r {
            for j in 0..c {
                d[i * c + j] = (0..n).map(|k| a.data[i * n + k] * b.data[k * c + j]).sum();
            }
        }
        Ok(Tensor::new(vec![r, c], d))
    }

    fn t(shape: &[usize], v: &[f32]) -> Tensor {
        Tensor::new(shape.to_vec(), v.to_vec())
    }

    #[test]
    fn loha_multiplies_two_products() {
        let (a, b) = (t(&[2, 1], &[1.0, 2.0]), t(&[1, 2], &[3.0, 4.0]));
        let (c, d) = (t(&[2, 1], &[2.0, 1.0]), t(&[1, 2], &[1.0, 1.0]));
        assert_eq!(loha((&a, &b), (&c, &d), None, Work::F32, &mut cpu).unwrap().data, [6.0, 8.0, 6.0, 8.0]);
    }

    #[test]
    fn tucker_with_identity_core_is_a_product() {
        let a = t(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = t(&[2, 2], &[5.0, 6.0, 7.0, 8.0]);
        let core = t(&[2, 2, 1, 1], &[1.0, 0.0, 0.0, 1.0]);
        // a is indexed (i, p), so it enters transposed
        assert_eq!(tucker(&core, &b, &a, &mut cpu).unwrap(), cpu(&a.t01(), &b).unwrap());
    }

    #[test]
    fn kron_matches_torch() {
        let k = kron(&t(&[1, 2], &[1.0, 2.0]), &t(&[2, 1], &[3.0, 4.0]), Work::F32);
        assert_eq!((k.shape, k.data), (vec![2, 2], vec![3.0, 6.0, 4.0, 8.0]));
    }

    #[test]
    fn mid_goes_between_down_and_up() {
        let up = t(&[2, 1], &[1.0, 2.0]);
        let down = t(&[1, 2], &[3.0, 4.0]);
        let mid = t(&[1, 1, 1, 1], &[2.0]);
        assert_eq!(lora_mid(&up, &down, &mid, &mut cpu).unwrap().data, [6.0, 8.0, 12.0, 16.0]);
    }

    #[test]
    fn glora_tells_its_forms_apart() {
        let w = t(&[2, 3], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let (a1, a2) = (t(&[3, 1], &[1.0, 0.0, 0.0]), t(&[1, 3], &[0.0, 1.0, 0.0]));
        let (b1, b2) = (t(&[2, 1], &[1.0, 1.0]), t(&[1, 3], &[0.0, 0.0, 1.0]));
        let (d, rank) = glora(&a1, &a2, &b1, &b2, &w, Work::F32, &mut cpu).unwrap();
        assert_eq!(rank, 1);
        assert_eq!(d.data, [0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn symmetric_oft_blocks_change_nothing() {
        // blocks - blocks^T is 0, so R = I
        let w = t(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = t(&[1, 2, 2], &[0.5, 0.25, 0.25, 0.5]);
        assert_eq!(oft(&b, 0.0, 1.0, &w, &mut cpu).unwrap().data, [0.0; 4]);
        let b = t(&[1, 1, 2, 2], &[0.5, 0.25, 0.25, 0.5]);
        assert_eq!(boft(&b, None, 1.0, 1.0, &w).unwrap().data, [0.0; 4]);
    }

    #[test]
    fn oft_rotates_rows() {
        // Q = [[0, 1], [-1, 0]]: R = (I + Q)(I - Q)^-1 = Q
        let w = t(&[2, 1], &[1.0, 0.0]);
        let b = t(&[1, 2, 2], &[0.0, 1.0, 0.0, 0.0]);
        let d = oft(&b, 0.0, 1.0, &w, &mut cpu).unwrap();
        assert_eq!(d.data, [-1.0, 1.0]);
    }

    #[test]
    fn dora_rescales_rows_to_the_norm() {
        let w = t(&[1, 2], &[3.0, 4.0]);
        let out = dora(&t(&[1, 1], &[10.0]), &w, &[0.0, 0.0], 1.0, 1.0, Work::F32).unwrap();
        let k = 10.0 / (5.0 + f32::EPSILON);
        assert_eq!(out, [3.0 * k, 4.0 * k]);
    }
}
