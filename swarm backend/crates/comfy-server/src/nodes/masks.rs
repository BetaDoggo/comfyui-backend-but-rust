//! MASK nodes: element-wise mask math done entirely on the server (SwarmUI's regional
//! prompting masks, inverting, mask <-> image).

use std::sync::Arc;

use swarm_engine_api::{ImageBatch, Mask};

use super::{input_err, NodeError, NodeOutput, NodeRegistry};
use crate::encode;
use crate::env::ServerEnv;
use crate::schema::*;
use crate::value::Value;

fn mask_of(count: u32, height: u32, width: u32, data: Vec<f32>) -> Mask {
    Mask { count, height, width, data: Arc::new(data) }
}

/// SwarmUI's `mask_size_match`: both masks resized to the larger height/width. (Swarm uses
/// bicubic; this uses the server's bilinear resize. SwarmUI's own region masks are all
/// 256x256, so in practice nothing is resized.)
fn size_match(a: &Mask, b: &Mask) -> Result<(Mask, Mask), NodeError> {
    let (h, w) = (a.height.max(b.height), a.width.max(b.width));
    let fit = |m: &Mask| -> Result<Mask, NodeError> { if (m.height, m.width) == (h, w) { Ok(m.clone()) } else { Ok(encode::resize_mask(m, w, h)?) } };
    Ok((fit(a)?, fit(b)?))
}

/// Element-wise op with torch-style broadcasting over the batch (a count of 1 repeats).
fn zip(a: &Mask, b: &Mask, f: impl Fn(f32, f32) -> f32) -> Result<Mask, NodeError> {
    let (a, b) = size_match(a, b)?;
    let count = match (a.count, b.count) {
        (x, y) if x == y => x,
        (1, y) => y,
        (x, 1) => x,
        (x, y) => return Err(input_err(format!("can't combine mask batches of {x} and {y}"))),
    };
    let plane = (a.height * a.width) as usize;
    let at = |m: &Mask, i: usize| m.data[if m.count == 1 { i % plane } else { i }];
    let data = (0..count as usize * plane).map(|i| f(at(&a, i), at(&b, i))).collect();
    Ok(mask_of(count, a.height, a.width, data))
}

pub fn register(r: &mut NodeRegistry, env: &ServerEnv) {
    r.add(
        env,
        |_| {
            let pct = |name: &str, what: &str| float(name, 0.0, 0.0, 1.0, 0.05).tooltip(&format!("The {what} of the mask as a percentage of the image size."));
            NodeSchema::new("SwarmSquareMaskFromPercent", "SwarmSquareMaskFromPercent", "SwarmUI/masks")
                .describe("Creates a simple square mask with the specified dimensions and position, with the specified strength (ie value of all masked pixels, leaving the rest black ie 0).")
                .input(pct("x", "x position"))
                .input(pct("y", "y position"))
                .input(pct("width", "width"))
                .input(pct("height", "height"))
                .input(float_nostep("strength", 1.0, 0.0, 1.0).tooltip("The strength of the mask, ie the value of all masked pixels, leaving the rest black ie 0."))
                .output("MASK", "MASK")
        },
        |i, _| {
            const SCALE: f64 = 256.0;
            let (x, y, w, h) = (i.float("x")?, i.float("y")?, i.float("width")?, i.float("height")?);
            let strength = i.float("strength")? as f32;
            // torch slicing: mask[int(y*S):int((y+h)*S), int(x*S):int((x+w)*S)], clamped to the mask.
            let clamp = |v: f64| ((v * SCALE) as i64).clamp(0, SCALE as i64) as usize;
            let (r0, r1, c0, c1) = (clamp(y), clamp(y + h), clamp(x), clamp(x + w));
            let s = SCALE as usize;
            let data = (0..s * s).map(|p| if (r0..r1).contains(&(p / s)) && (c0..c1).contains(&(p % s)) { strength } else { 0.0 }).collect();
            Ok(NodeOutput::one(Value::Mask(mask_of(1, s as u32, s as u32, data))))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmOverMergeMasksForOverlapFix", "SwarmOverMergeMasksForOverlapFix", "SwarmUI/masks")
                .describe("Merges two masks by simply adding them together, without any overlap handling. Intended for use with the Overlap nodes.")
                .input(link("mask_a", "MASK"))
                .input(link("mask_b", "MASK"))
                .output("MASK", "MASK")
        },
        |i, _| Ok(NodeOutput::one(Value::Mask(zip(i.mask("mask_a")?, i.mask("mask_b")?, |a, b| a + b)?))),
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmCleanOverlapMasksExceptSelf", "SwarmCleanOverlapMasksExceptSelf", "SwarmUI/masks")
                .describe("If masks have been overmerged, this takes a single mask and grabs just the CleanOverlap result for the one mask relative to the overmerge result.")
                .input(link("mask_self", "MASK"))
                .input(link("mask_merged", "MASK"))
                .output("MASK", "MASK")
        },
        |i, _| Ok(NodeOutput::one(Value::Mask(zip(i.mask("mask_self")?, i.mask("mask_merged")?, |s, m| s / m.clamp(1.0, 9999.0))?))),
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmExcludeFromMask", "SwarmExcludeFromMask", "SwarmUI/masks")
                .describe("Excludes the area of the exclude mask from the main mask, such that the main mask will be black in the area of the exclude mask. This is a simple subtract and clamp.")
                .input(link("main_mask", "MASK"))
                .input(link("exclude_mask", "MASK"))
                .output("MASK", "MASK")
        },
        |i, _| Ok(NodeOutput::one(Value::Mask(zip(i.mask("main_mask")?, i.mask("exclude_mask")?, |m, e| (m - e).clamp(0.0, 1.0))?))),
    );
    r.add(
        env,
        |_| NodeSchema::new("InvertMask", "Invert Mask", "image/mask").input(link("mask", "MASK")).output("MASK", "MASK"),
        |i, _| {
            let m = i.mask("mask")?;
            Ok(NodeOutput::one(Value::Mask(mask_of(m.count, m.height, m.width, m.data.iter().map(|v| 1.0 - v).collect()))))
        },
    );
    r.add(
        env,
        |_| NodeSchema::new("MaskToImage", "Convert Mask to Image", "image/mask").input(link("mask", "MASK")).output("IMAGE", "IMAGE"),
        |i, _| {
            let m = i.mask("mask")?;
            let data = m.data.iter().flat_map(|v| [*v, *v, *v]).collect();
            Ok(NodeOutput::one(Value::Image(ImageBatch::from_f32(m.count, m.height, m.width, 3, data))))
        },
    );

    // ------------------------------------------------------------- init image / inpainting masks
    r.add(
        env,
        |_| {
            NodeSchema::new("ImageToMask", "Convert Image to Mask", "image/mask")
                .input(link("image", "IMAGE"))
                .input(combo("channel", &["red", "green", "blue", "alpha"]))
                .output("MASK", "MASK")
        },
        |i, _| {
            let img = i.image("image")?;
            let c = ["red", "green", "blue", "alpha"].iter().position(|n| *n == i.string("channel").unwrap_or("red")).unwrap_or(0);
            if c as u32 >= img.channels {
                return Err(input_err(format!("image has no {} channel", i.string("channel")?)));
            }
            let mut data = Vec::with_capacity((img.count * img.height * img.width) as usize);
            for f in 0..img.count {
                data.extend(img.frame_f32(f).chunks(img.channels as usize).map(|px| px[c]));
            }
            Ok(NodeOutput::one(Value::Mask(mask_of(img.count, img.height, img.width, data))))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("ThresholdMask", "Threshold Mask", "image/mask")
                .input(link("mask", "MASK"))
                .input(float("value", 0.5, 0.0, 1.0, 0.01))
                .output("MASK", "MASK")
        },
        |i, _| {
            let (m, v) = (i.mask("mask")?, i.float("value")? as f32);
            Ok(NodeOutput::one(Value::Mask(map(m, |x| if x > v { 1.0 } else { 0.0 }))))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("GrowMask", "Grow Mask", "image/mask")
                .input(link("mask", "MASK"))
                .input(int_step("expand", 0, -MAX_RESOLUTION, MAX_RESOLUTION, 1))
                .input(boolean("tapered_corners", true))
                .output("MASK", "MASK")
        },
        |i, _| {
            let (m, expand, tapered) = (i.mask("mask")?, i.int("expand")?, i.boolean("tapered_corners")?);
            // scipy grey_dilation / grey_erosion with a 3x3 footprint (a cross when tapered), |expand| times.
            let mut out = m.clone();
            for _ in 0..expand.unsigned_abs() {
                out = morph3(&out, !tapered, expand > 0);
            }
            Ok(NodeOutput::one(Value::Mask(out)))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("MaskComposite", "Combine Masks", "image/mask")
                .input(link("destination", "MASK"))
                .input(link("source", "MASK"))
                .input(int_step("x", 0, 0, MAX_RESOLUTION, 1))
                .input(int_step("y", 0, 0, MAX_RESOLUTION, 1))
                .input(combo("operation", &["multiply", "add", "subtract", "and", "or", "xor"]))
                .output("MASK", "MASK")
        },
        |i, _| {
            let (dst, src) = (i.mask("destination")?, i.mask("source")?);
            let (left, top) = (i.uint("x")? as usize, i.uint("y")? as usize);
            let op = i.string("operation")?;
            let (dw, dh, sw, sh) = (dst.width as usize, dst.height as usize, src.width as usize, src.height as usize);
            let mut out = dst.data.as_ref().clone();
            for b in 0..dst.count as usize {
                let sb = if src.count == 1 { 0 } else { b.min(src.count as usize - 1) };
                for y in top..(top + sh).min(dh) {
                    for x in left..(left + sw).min(dw) {
                        let d = &mut out[b * dw * dh + y * dw + x];
                        let s = src.data[sb * sw * sh + (y - top) * sw + (x - left)];
                        let (db, sbit) = (d.round() != 0.0, s.round() != 0.0);
                        *d = match op {
                            "multiply" => *d * s,
                            "add" => *d + s,
                            "subtract" => *d - s,
                            "and" => (db && sbit) as u8 as f32,
                            "or" => (db || sbit) as u8 as f32,
                            _ => (db ^ sbit) as u8 as f32,
                        };
                    }
                }
            }
            out.iter_mut().for_each(|v| *v = v.clamp(0.0, 1.0));
            Ok(NodeOutput::one(Value::Mask(mask_of(dst.count, dst.height, dst.width, out))))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("CropMask", "Crop Mask", "image/mask")
                .input(link("mask", "MASK"))
                .input(int_step("x", 0, 0, MAX_RESOLUTION, 1))
                .input(int_step("y", 0, 0, MAX_RESOLUTION, 1))
                .input(int_step("width", 512, 1, MAX_RESOLUTION, 1))
                .input(int_step("height", 512, 1, MAX_RESOLUTION, 1))
                .output("MASK", "MASK")
        },
        |i, _| Ok(NodeOutput::one(Value::Mask(crop(i.mask("mask")?, i.uint("x")?, i.uint("y")?, i.uint("width")?, i.uint("height")?)?))),
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmMaskBlur", "SwarmMaskBlur", "SwarmUI/masks")
                .describe("Blurs a mask using a Gaussian blur.")
                .input(link("mask", "MASK"))
                .input(int_step("blur_radius", 1, 1, 64, 1).tooltip("The radius of the blur kernel."))
                .input(float("sigma", 1.0, 0.1, 10.0, 0.1).tooltip("The standard deviation of the Gaussian blur kernel."))
                .output("MASK", "MASK")
        },
        |i, _| Ok(NodeOutput::one(Value::Mask(gaussian_blur(i.mask("mask")?, i.uint("blur_radius")? as usize, i.float("sigma")? as f32)))),
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmMaskGrow", "SwarmMaskGrow", "SwarmUI/masks")
                .describe("Expands the contents of the max, such that any value greater than 0 expands outward by the given number of pixels.")
                .input(link("mask", "MASK"))
                .input(int("grow", 0, 0, 1024).tooltip("Number of pixels to grow the mask by."))
                .output("MASK", "MASK")
        },
        |i, _| {
            // max_pool2d(kernel 3, stride 1, padding 1), (grow + 1) / 2 times.
            let mut out = i.mask("mask")?.clone();
            for _ in 0..(i.uint("grow")? + 1) / 2 {
                out = morph3(&out, true, true);
            }
            Ok(NodeOutput::one(Value::Mask(out)))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmMaskThreshold", "SwarmMaskThreshold", "SwarmUI/masks")
                .describe("Thresholds a mask to a range of values.")
                .input(link("mask", "MASK"))
                .input(float("min", 0.2, 0.0, 1.0, 0.01).tooltip("The minimum value to threshold the mask to."))
                .input(float("max", 0.8, 0.0, 1.0, 0.01).tooltip("The maximum value to threshold the mask to."))
                .output("MASK", "MASK")
        },
        |i, _| {
            // torch's `clamp(min, max)` with min > max sets everything to max (so the result is all 1s);
            // min == max would be 0 / 0 there, which we turn into 0 rather than NaN.
            let (lo, hi) = (i.float("min")? as f32, i.float("max")? as f32);
            Ok(NodeOutput::one(Value::Mask(map(i.mask("mask")?, |v| if hi == lo { 0.0 } else { (v.max(lo).min(hi) - lo) / (hi - lo) }))))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmMaskBounds", "SwarmMaskBounds", "SwarmUI/masks")
                .describe("Returns the bounding box of the mask (as pixel coordinates x,y,width,height), optionally grown by the number of pixels specified in 'grow'.")
                .input(link("mask", "MASK"))
                .input(int("grow", 0, 0, 1024).tooltip("Number of pixels to grow the mask by."))
                .input(int("aspect_x", 0, 0, 4096).optional().tooltip("An X width value, used to indicate a target aspect ratio. 0 to allow any aspect."))
                .input(int("aspect_y", 0, 0, 4096).optional().tooltip("A Y height value, used to indicate a target aspect ratio. 0 to allow any aspect."))
                .output("INT", "x")
                .output("INT", "y")
                .output("INT", "width")
                .output("INT", "height")
        },
        |i, _| {
            let m = i.mask("mask")?;
            let (x, y, w, h) = mask_bounds(m, i.int("grow")?, i.int("aspect_x").unwrap_or(0), i.int("aspect_y").unwrap_or(0));
            Ok(NodeOutput::values(vec![Value::Int(x), Value::Int(y), Value::Int(w), Value::Int(h)]))
        },
    );
}

fn map(m: &Mask, f: impl Fn(f32) -> f32) -> Mask {
    mask_of(m.count, m.height, m.width, m.data.iter().map(|v| f(*v)).collect())
}

/// One 3x3 dilation (max) or erosion (min) step over in-bounds neighbours; `square` uses all 8
/// neighbours, otherwise just the 4-connected cross.
pub fn morph3(m: &Mask, square: bool, dilate: bool) -> Mask {
    let (w, h) = (m.width as i64, m.height as i64);
    let mut out = Vec::with_capacity(m.data.len());
    for b in 0..m.count as i64 {
        let base = (b * w * h) as usize;
        for y in 0..h {
            for x in 0..w {
                let mut acc = m.data[base + (y * w + x) as usize];
                for (dy, dx) in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)] {
                    if !square && dy != 0 && dx != 0 {
                        continue;
                    }
                    let (ny, nx) = (y + dy, x + dx);
                    if ny < 0 || nx < 0 || ny >= h || nx >= w {
                        continue;
                    }
                    let v = m.data[base + (ny * w + nx) as usize];
                    acc = if dilate { acc.max(v) } else { acc.min(v) };
                }
                out.push(acc);
            }
        }
    }
    mask_of(m.count, m.height, m.width, out)
}

/// `mask[:, y:y+height, x:x+width]` with Python slice clamping.
pub fn crop(m: &Mask, x: u32, y: u32, width: u32, height: u32) -> Result<Mask, NodeError> {
    let (x0, y0) = (x.min(m.width), y.min(m.height));
    let (x1, y1) = (x.saturating_add(width).min(m.width), y.saturating_add(height).min(m.height));
    if x1 <= x0 || y1 <= y0 {
        return Err(input_err(format!("crop {width}x{height} at {x},{y} is outside the {}x{} mask", m.width, m.height)));
    }
    let mut out = Vec::new();
    for b in 0..m.count {
        for yy in y0..y1 {
            let row = ((b * m.height + yy) * m.width) as usize;
            out.extend_from_slice(&m.data[row + x0 as usize..row + x1 as usize]);
        }
    }
    Ok(mask_of(m.count, y1 - y0, x1 - x0, out))
}

/// SwarmMaskBlur: Gaussian over [-1, 1] kernel coordinates (ComfyUI's ImageBlur kernel), with
/// reflect padding at the edges. Separable, since the 2D kernel is a product of 1D ones.
fn gaussian_blur(m: &Mask, radius: usize, sigma: f32) -> Mask {
    if radius == 0 {
        return m.clone();
    }
    let ks = radius * 2 + 1;
    let k: Vec<f32> = (0..ks).map(|i| -1.0 + 2.0 * i as f32 / (ks - 1) as f32).map(|x| (-(x * x) / (2.0 * sigma * sigma)).exp()).collect();
    let sum: f32 = k.iter().sum();
    let k: Vec<f32> = k.iter().map(|v| v / sum).collect();
    let (w, h) = (m.width as i64, m.height as i64);
    let reflect = |i: i64, n: i64| -> i64 {
        if n == 1 {
            return 0;
        }
        let mut i = i;
        while i < 0 || i >= n {
            i = if i < 0 { -i } else { 2 * n - 2 - i };
        }
        i
    };
    let mut out = Vec::with_capacity(m.data.len());
    for b in 0..m.count as i64 {
        let base = (b * w * h) as usize;
        let mut tmp = vec![0f32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                tmp[(y * w + x) as usize] = (0..ks as i64).map(|j| k[j as usize] * m.data[base + (y * w + reflect(x + j - radius as i64, w)) as usize]).sum();
            }
        }
        for y in 0..h {
            for x in 0..w {
                out.push((0..ks as i64).map(|j| k[j as usize] * tmp[(reflect(y + j - radius as i64, h) * w + x) as usize]).sum());
            }
        }
    }
    mask_of(m.count, m.height, m.width, out)
}

/// SwarmMaskBounds on the first mask of the batch.
fn mask_bounds(m: &Mask, grow: i64, aspect_x: i64, aspect_y: i64) -> (i64, i64, i64, i64) {
    let (w, h) = (m.width as usize, m.height as usize);
    let plane = &m.data[..w * h];
    let cols: Vec<bool> = (0..w).map(|x| (0..h).map(|y| plane[y * w + x]).sum::<f32>() != 0.0).collect();
    let rows: Vec<bool> = (0..h).map(|y| plane[y * w..(y + 1) * w].iter().sum::<f32>() != 0.0).collect();
    // torch.argmax over 0/1 ints: the first 1, or 0 if there is none.
    let first = |a: Vec<bool>| -> f64 {
        let idx = a.iter().position(|v| *v).unwrap_or(0) as i64 - grow;
        idx.clamp(0, a.len() as i64 - 1) as f64
    };
    let rev = |a: &[bool]| a.iter().rev().copied().collect::<Vec<_>>();
    let mut x_start = first(cols.clone());
    let mut x_end = w as f64 - first(rev(&cols));
    let mut y_start = first(rows.clone());
    let mut y_end = h as f64 - first(rev(&rows));
    if aspect_x > 0 && aspect_y > 0 {
        let actual = aspect_x as f64 / aspect_y as f64;
        let (width, height) = (x_end - x_start, y_end - y_start);
        if width / height > actual {
            let desired = width / actual;
            y_start = (y_start - (desired - height) / 2.0).max(0.0);
            y_end = (y_start + desired).min(h as f64);
        } else {
            let desired = height * actual;
            x_start = (x_start - (desired - width) / 2.0).max(0.0);
            x_end = (x_start + desired).min(w as f64);
        }
    }
    (x_start as i64, y_start as i64, (x_end - x_start) as i64, (y_end - y_start) as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(w: u32, h: u32, on: &[(u32, u32)]) -> Mask {
        let mut d = vec![0.0; (w * h) as usize];
        for (x, y) in on {
            d[(y * w + x) as usize] = 1.0;
        }
        mask_of(1, h, w, d)
    }

    #[test]
    fn bounds_match_swarm() {
        // A single pixel at (5, 3) in a 10x8 mask: [5, 6) x [3, 4).
        assert_eq!(mask_bounds(&m(10, 8, &[(5, 3)]), 0, 0, 0), (5, 3, 1, 1));
        // grow 2 widens each side by 2.
        assert_eq!(mask_bounds(&m(10, 8, &[(5, 3)]), 2, 0, 0), (3, 1, 5, 5));
        // Empty mask: the whole image.
        assert_eq!(mask_bounds(&m(10, 8, &[]), 0, 0, 0), (0, 0, 10, 8));
    }

    #[test]
    fn grow_and_blur() {
        let g = morph3(&m(5, 5, &[(2, 2)]), false, true);
        assert_eq!(g.data.iter().filter(|v| **v == 1.0).count(), 5);
        let sq = morph3(&m(5, 5, &[(2, 2)]), true, true);
        assert_eq!(sq.data.iter().filter(|v| **v == 1.0).count(), 9);
        let b = gaussian_blur(&mask_of(1, 4, 4, vec![1.0; 16]), 2, 1.0);
        assert!(b.data.iter().all(|v| (v - 1.0).abs() < 1e-5), "a flat mask stays flat");
    }
}
