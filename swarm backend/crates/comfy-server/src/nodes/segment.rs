//! Segmentation, background removal and model upscaling: `<segment:...>` / `<clear:...>`,
//! Remove Background, SAM2 helpers, and Refiner Upscale Method = `model-*`.

use std::sync::Arc;

use swarm_engine_api::{AddonKind, Mask, ModelFolder, Opaque, SegmentRequest, UpscaleImageRequest};

use super::loaders::resolve;
use super::{input_err, NodeOutput, NodeRegistry};
use crate::env::ServerEnv;
use crate::schema::*;
use crate::value::Value;

/// `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))`.
fn ellipse(k: usize) -> Vec<bool> {
    let r = (k / 2) as i64;
    let c = r;
    let inv_r2 = if r > 0 { 1.0 / (r * r) as f64 } else { 0.0 };
    let mut out = vec![false; k * k];
    for i in 0..k as i64 {
        let dy = i - r;
        if dy.abs() <= r {
            let dx = (c as f64 * (((r * r - dy * dy) as f64) * inv_r2).sqrt()).round() as i64;
            let (j1, j2) = ((c - dx).max(0), (c + dx + 1).min(k as i64));
            for j in j1..j2 {
                out[i as usize * k + j as usize] = true;
            }
        }
    }
    out
}

/// Grey dilation / erosion of a `w`x`h` u8 plane with a `k`x`k` footprint (out-of-bounds ignored,
/// as OpenCV's default morphology border).
fn morph(plane: &[u8], w: usize, h: usize, kernel: &[bool], k: usize, dilate: bool) -> Vec<u8> {
    let r = (k / 2) as i64;
    let mut out = vec![0u8; w * h];
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let mut acc: Option<u8> = None;
            for ky in 0..k as i64 {
                for kx in 0..k as i64 {
                    if !kernel[(ky * k as i64 + kx) as usize] {
                        continue;
                    }
                    let (yy, xx) = (y + ky - r, x + kx - r);
                    if yy < 0 || xx < 0 || yy >= h as i64 || xx >= w as i64 {
                        continue;
                    }
                    let v = plane[yy as usize * w + xx as usize];
                    acc = Some(match acc {
                        None => v,
                        Some(a) if dilate => a.max(v),
                        Some(a) => a.min(v),
                    });
                }
            }
            out[y as usize * w + x as usize] = acc.unwrap_or(0);
        }
    }
    out
}

/// SwarmUI's `fill_mask_holes`: morphological close with an ellipse, then everything not
/// 4-connected to the border through zero pixels becomes 255.
fn fill_holes(plane: &[u8], w: usize, h: usize, k: usize) -> Vec<u8> {
    let kernel = ellipse(k);
    let closed = morph(&morph(plane, w, h, &kernel, k, true), w, h, &kernel, k, false);
    // Flood fill from the zero border of a (w + 2) x (h + 2) canvas.
    let (cw, ch) = (w + 2, h + 2);
    let at = |x: usize, y: usize| if x == 0 || y == 0 || x == cw - 1 || y == ch - 1 { 0 } else { closed[(y - 1) * w + (x - 1)] };
    let mut outside = vec![false; cw * ch];
    let mut stack = vec![(0usize, 0usize)];
    outside[0] = true;
    while let Some((x, y)) = stack.pop() {
        let next = [(x.wrapping_sub(1), y), (x + 1, y), (x, y.wrapping_sub(1)), (x, y + 1)];
        for (nx, ny) in next {
            if nx < cw && ny < ch && !outside[ny * cw + nx] && at(nx, ny) == 0 {
                outside[ny * cw + nx] = true;
                stack.push((nx, ny));
            }
        }
    }
    (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).map(|(x, y)| if outside[(y + 1) * cw + x + 1] { 0 } else { 255 }).collect()
}

pub fn register(r: &mut NodeRegistry, env: &ServerEnv) {
    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmClipSeg", "SwarmClipSeg", "SwarmUI/masks")
                .describe("Segment an image using CLIPSeg, creating a mask of what part of an image appears to match the given text.")
                .input(link("images", "IMAGE"))
                .input(string("match_text", "", true).no_default().tooltip("A short description (a few words) to describe something within the image to find and mask."))
                .input(float("threshold", 0.5, 0.0, 1.0, 0.01).tooltip("Threshold to apply to the mask, higher values will make the mask more strict. Without sufficient thresholding, CLIPSeg may include random stray content around the edges."))
                .output("MASK", "MASK")
        },
        |i, ctx| {
            let req = SegmentRequest::ClipSeg { images: i.image("images")?.clone(), text: i.string("match_text")?.to_string(), threshold: i.float("threshold")? };
            Ok(NodeOutput::one(Value::Mask(ctx.engine.segment(&req)?)))
        },
    );
    r.add(
        env,
        |env| {
            NodeSchema::new("SwarmYoloDetection", "SwarmYoloDetection", "SwarmUI/masks")
                .input(link("image", "IMAGE"))
                .input(combo("model_name", &env.models.list(ModelFolder::Yolov8)))
                .input(int_step("index", 0, 0, 256, 1))
                .input(string("class_filter", "", false).optional())
                .input(combo("sort_order", &["left-right", "right-left", "top-bottom", "bottom-top", "largest-smallest", "smallest-largest"]).optional())
                .input(float("threshold", 0.25, 0.0, 1.0, 0.01).optional())
                .output("MASK", "MASK")
        },
        |i, ctx| {
            let req = SegmentRequest::Yolo {
                image: i.image("image")?.clone(),
                model: resolve(ctx.env, &[ModelFolder::Yolov8], i.string("model_name")?)?,
                index: i.uint("index")?,
                class_filter: i.opt_string("class_filter")?.unwrap_or("").to_string(),
                sort_order: i.opt_string("sort_order")?.unwrap_or("left-right").to_string(),
                threshold: i.float("threshold").unwrap_or(0.25),
            };
            Ok(NodeOutput::one(Value::Mask(ctx.engine.segment(&req)?)))
        },
    );
    r.add(
        env,
        |_| NodeSchema::new("SwarmRemBg", "SwarmRemBg", "SwarmUI/images").input(link("images", "IMAGE")).output("IMAGE", "IMAGE").output("MASK", "MASK"),
        |i, ctx| {
            let out = ctx.engine.remove_background(i.image("images")?)?;
            // SwarmUI's node returns 1 - alpha as the mask (zeros if there's no alpha).
            let mask: Vec<f32> = if out.has_alpha() { out.alpha().iter().map(|a| 1.0 - a).collect() } else { vec![0.0; (out.count * out.width * out.height) as usize] };
            let mask = Mask { count: out.count, height: out.height, width: out.width, data: Arc::new(mask) };
            Ok(NodeOutput::values(vec![Value::Image(out), Value::Mask(mask)]))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("RemoveBackground", "Remove Background", "image/background removal")
                .describe("Generates a foreground mask to remove the background from an image using a background removal model.")
                .input(link("bg_removal_model", "BACKGROUND_REMOVAL").tooltip("Background removal model used to generate the mask"))
                .input(link("image", "IMAGE").tooltip("Input image to remove the background from"))
                .output("MASK", "MASK")
        },
        |i, ctx| {
            let model = i.addon("bg_removal_model", AddonKind::BackgroundRemoval)?;
            Ok(NodeOutput::one(Value::Mask(ctx.engine.background_mask(model, i.image("image")?)?)))
        },
    );
    r.add(
        env,
        |_| NodeSchema::new("SwarmSam2BBoxFromJson", "SwarmSam2BBoxFromJson", "SAM2").input(string("bbox_json", "", false).no_default().force_input()).output("BBOX", "bboxes"),
        |i, _| {
            let coords: Vec<f64> = serde_json::from_str(i.string("bbox_json")?).map_err(|e| input_err(format!("bbox_json: {e}")))?;
            if coords.len() < 4 {
                return Err(input_err("bbox_json needs [x1, y1, x2, y2]"));
            }
            // BBOX payload: Vec<[f64; 4]> of [x1, y1, x2, y2] pixel boxes.
            let boxes: Vec<[f64; 4]> = vec![[coords[0], coords[1], coords[2], coords[3]]];
            Ok(NodeOutput::one(Value::Other { ty: "BBOX".into(), data: Opaque::new(boxes) }))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmSam2MaskPostProcess", "SwarmSam2MaskPostProcess", "SAM2")
                .input(link("mask", "MASK"))
                .input(boolean("fill_holes", true).optional())
                .input(int_step("hole_kernel_size", 5, 1, 21, 2).optional())
                .output("MASK", "mask")
        },
        |i, _| {
            let m = i.mask("mask")?;
            let (w, h) = (m.width as usize, m.height as usize);
            let fill = i.boolean("fill_holes").unwrap_or(true);
            let k = i.uint("hole_kernel_size").unwrap_or(5) as usize;
            let mut data = Vec::with_capacity(m.data.len());
            for b in 0..m.count as usize {
                let plane: Vec<u8> = m.data[b * w * h..(b + 1) * w * h].iter().map(|v| (v * 255.0) as u8).collect();
                let plane = if fill { fill_holes(&plane, w, h, k) } else { plane };
                data.extend(plane.iter().map(|v| *v as f32 / 255.0));
            }
            Ok(NodeOutput::one(Value::Mask(Mask { count: m.count, height: m.height, width: m.width, data: Arc::new(data) })))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("ImageUpscaleWithModel", "Upscale Image (using Model)", "image/upscaling")
                .input(link("upscale_model", "UPSCALE_MODEL"))
                .input(link("image", "IMAGE"))
                .output("IMAGE", "IMAGE")
        },
        |i, ctx| {
            let req = UpscaleImageRequest { model: i.addon("upscale_model", AddonKind::UpscaleModel)?.clone(), image: i.image("image")?.clone() };
            Ok(NodeOutput::one(Value::Image(ctx.engine.upscale_image(&req)?)))
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ellipse_matches_opencv() {
        // cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        let e: Vec<u8> = ellipse(5).iter().map(|b| *b as u8).collect();
        assert_eq!(e, [0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0]);
    }

    #[test]
    fn holes_are_filled() {
        // A 7x7 ring with a hole in the middle.
        let mut p = vec![0u8; 49];
        for y in 1..6 {
            for x in 1..6 {
                if !(x == 3 && y == 3) {
                    p[y * 7 + x] = 255;
                }
            }
        }
        let f = fill_holes(&p, 7, 7, 1);
        assert_eq!(f[3 * 7 + 3], 255);
        assert_eq!(f[0], 0);
    }
}
