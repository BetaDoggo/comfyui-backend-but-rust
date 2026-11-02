//! Guidance tweaks and runtime options: nodes that clone-and-patch a MODEL / CLIP / VAE.
//! SwarmUI emits them for the "Alternate Guidance" group, FreeU, Seamless Tileable,
//! CLIP Stop At Layer, EasyCache, attention/compile options, and a few model-class defaults.
//! Each one only appends a patch to the handle; the engine applies them at sample/encode time.

use swarm_engine_api::{ModelPatch, SparseAttention, SparseAttentionMethod, TextEncoderPatch, TileAxis, VaePatch};

use super::{input_err, NodeInputs, NodeOutput, NodeRegistry, NodeResult};
use crate::env::ServerEnv;
use crate::schema::*;
use crate::value::Value;

/// SwarmUI's hardcoded default for "Model Attention Backend" (ComfyUI's stock backend name).
const SWARM_DEFAULT_ATTENTION: &str = "pytorch attention";

fn patch_model(i: &NodeInputs, patch: ModelPatch) -> NodeResult {
    Ok(NodeOutput::one(Value::Model(i.model("model")?.with_patch(patch))))
}

/// A node that is just `model -> patched model` with float widgets.
fn model_node(class: &str, category: &str, output_name: &str, widgets: Vec<InputSpec>) -> NodeSchema {
    let mut s = NodeSchema::new(class, class, category).input(link("model", "MODEL"));
    for w in widgets {
        s = s.input(w);
    }
    s.output("MODEL", output_name)
}

fn tile_axis(i: &NodeInputs) -> Result<TileAxis, super::NodeError> {
    let s = i.opt_string("tile_axis")?.unwrap_or("Both");
    TileAxis::parse(s).ok_or_else(|| input_err(format!("unknown tile_axis '{s}'")))
}

pub fn register(r: &mut NodeRegistry, env: &ServerEnv) {
    // ------------------------------------------------------------- Alternate Guidance group
    r.add(
        env,
        |_| {
            model_node(
                "SelfAttentionGuidance",
                "experimental",
                "MODEL",
                vec![float("scale", 0.5, -2.0, 5.0, 0.01), float("blur_sigma", 2.0, 0.0, 10.0, 0.1).advanced()],
            )
        },
        |i, _| patch_model(i, ModelPatch::SelfAttentionGuidance { scale: i.float("scale")?, blur_sigma: i.float("blur_sigma")? }),
    );
    r.add(
        env,
        |_| model_node("PerturbedAttentionGuidance", "model/patch/unet", "MODEL", vec![float("scale", 3.0, 0.0, 100.0, 0.01)]),
        |i, _| patch_model(i, ModelPatch::PerturbedAttentionGuidance { scale: i.float("scale")? }),
    );
    r.add(
        env,
        |_| model_node("RescaleCFG", "model/patch", "MODEL", vec![float("multiplier", 0.7, 0.0, 1.0, 0.01)]),
        |i, _| patch_model(i, ModelPatch::RescaleCfg { multiplier: i.float("multiplier")? }),
    );
    r.add(
        env,
        |_| {
            model_node(
                "RenormCFG",
                "model/patch",
                "MODEL",
                vec![float("cfg_trunc", 100.0, 0.0, 100.0, 0.01).advanced(), float("renorm_cfg", 1.0, 0.0, 100.0, 0.01).advanced()],
            )
        },
        |i, _| patch_model(i, ModelPatch::RenormCfg { cfg_trunc: i.float("cfg_trunc")?, renorm_cfg: i.float("renorm_cfg")? }),
    );
    r.add(env, |_| model_node("CFGZeroStar", "advanced/guidance", "patched_model", vec![]), |i, _| patch_model(i, ModelPatch::CfgZeroStar));
    r.add(
        env,
        |_| {
            model_node("TCFG", "advanced/guidance", "patched_model", vec![])
                .describe("TCFG – Tangential Damping CFG (2503.18137)\n\nRefine the uncond (negative) to align with the cond (positive) for improving quality.")
        },
        |i, _| patch_model(i, ModelPatch::Tcfg),
    );
    r.add(
        env,
        |_| {
            model_node(
                "NAGuidance",
                "advanced/guidance",
                "MODEL",
                vec![
                    float("nag_scale", 5.0, 0.0, 50.0, 0.1).tooltip("The guidance scale factor. Higher values push further from the negative prompt."),
                    float("nag_alpha", 0.5, 0.0, 1.0, 0.01).tooltip("Blending factor for the normalized attention. 1.0 is full replacement, 0.0 is no effect."),
                    float("nag_tau", 1.5, 1.0, 10.0, 0.01),
                ],
            )
            .describe("Applies Normalized Attention Guidance to models, enabling negative prompts on distilled/schnell models.")
        },
        |i, _| {
            patch_model(i, ModelPatch::NormalizedAttentionGuidance { scale: i.float("nag_scale")?, alpha: i.float("nag_alpha")?, tau: i.float("nag_tau")? })
        },
    );

    // ------------------------------------------------------------- FreeU group
    for (class, version, b1, b2) in [("FreeU", 1u8, 1.1, 1.2), ("FreeU_V2", 2u8, 1.3, 1.4)] {
        r.add(
            env,
            move |_| {
                model_node(
                    class,
                    "model/patch/unet",
                    "MODEL",
                    vec![
                        float("b1", b1, 0.0, 10.0, 0.01).advanced(),
                        float("b2", b2, 0.0, 10.0, 0.01).advanced(),
                        float("s1", 0.9, 0.0, 10.0, 0.01).advanced(),
                        float("s2", 0.2, 0.0, 10.0, 0.01).advanced(),
                    ],
                )
            },
            move |i, _| patch_model(i, ModelPatch::FreeU { version, b1: i.float("b1")?, b2: i.float("b2")?, s1: i.float("s1")?, s2: i.float("s2")? }),
        );
    }

    r.add(
        env,
        |_| model_node("VideoLinearCFGGuidance", "model/sampling/guiders", "MODEL", vec![float("min_cfg", 1.0, 0.0, 100.0, 0.5).advanced()]),
        |i, _| patch_model(i, ModelPatch::VideoLinearCfg { min_cfg: i.float("min_cfg")? }),
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("DifferentialDiffusion", "Differential Diffusion", "experimental")
                .input(link("model", "MODEL"))
                .input(float("strength", 1.0, 0.0, 1.0, 0.01).optional())
                .output("MODEL", "MODEL")
        },
        |i, _| patch_model(i, ModelPatch::DifferentialDiffusion { strength: i.float("strength").unwrap_or(1.0) }),
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("HyperTile", "HyperTile", "model/patch/unet")
                .input(link("model", "MODEL"))
                .input(int("tile_size", 256, 1, 2048))
                .input(int("swap_size", 2, 1, 128))
                .input(int("max_depth", 0, 0, 10))
                .input(boolean("scale_depth", false))
                .output("MODEL", "MODEL")
        },
        |i, _| {
            let p = ModelPatch::HyperTile { tile_size: i.uint("tile_size")?, swap_size: i.uint("swap_size")?, max_depth: i.uint("max_depth")?, scale_depth: i.boolean("scale_depth")? };
            patch_model(i, p)
        },
    );

    // ------------------------------------------------------------- model-class defaults
    r.add(
        env,
        |_| {
            model_node("CFGNorm", "advanced/guidance", "patched_model", vec![float("strength", 1.0, 0.0, 100.0, 0.01)]).input(
                boolean("pre_cfg", false)
                    .optional()
                    .tooltip("If true, rescale the combined noise BEFORE the sampler's CFG combine, without clamping (can amplify)."),
            )
        },
        |i, _| patch_model(i, ModelPatch::CfgNorm { strength: i.float("strength")?, pre_cfg: i.boolean("pre_cfg").unwrap_or(false) }),
    );
    r.add(
        env,
        |_| {
            model_node(
                "ModelNoiseScale",
                "model/patch",
                "MODEL",
                vec![float("noise_scale", 1.0, 0.0, 64.0, 0.01).tooltip("Absolute training noise scale. For example HiDream-O1 base: 8.0, dev: 7.5.")],
            )
        },
        |i, _| patch_model(i, ModelPatch::NoiseScale { noise_scale: i.float("noise_scale")? }),
    );
    r.add(
        env,
        |_| {
            model_node(
                "ModelSamplingDiscrete",
                "model/patch",
                "MODEL",
                vec![combo("sampling", &["eps", "v_prediction", "lcm", "x0", "img_to_img", "img_to_img_flow"]), boolean("zsnr", false).advanced()],
            )
        },
        |i, _| patch_model(i, ModelPatch::SamplingDiscrete { sampling: i.string("sampling")?.to_string(), zsnr: i.boolean("zsnr")? }),
    );

    // ------------------------------------------------------------- speed / runtime options
    r.add(
        env,
        |_| {
            model_node(
                "EasyCache",
                "advanced/debug",
                "MODEL",
                vec![
                    float("reuse_threshold", 0.2, 0.0, 3.0, 0.01).advanced().tooltip("The threshold for reusing cached steps."),
                    float("start_percent", 0.15, 0.0, 1.0, 0.01).advanced().tooltip("The relative sampling step to begin use of EasyCache."),
                    float("end_percent", 0.95, 0.0, 1.0, 0.01).advanced().tooltip("The relative sampling step to end use of EasyCache."),
                    boolean("verbose", false).advanced(),
                ],
            )
            .describe("Native EasyCache implementation.")
        },
        |i, _| {
            patch_model(
                i,
                ModelPatch::EasyCache {
                    reuse_threshold: i.float("reuse_threshold")?,
                    start_percent: i.float("start_percent")?,
                    end_percent: i.float("end_percent")?,
                    verbose: i.boolean("verbose")?,
                },
            )
        },
    );
    r.add(
        env,
        |env| {
            // SwarmUI always offers its own default, "pytorch attention", next to whatever is listed here.
            let mut backends = vec![SWARM_DEFAULT_ATTENTION.to_string()];
            backends.extend(env.attention_backends.iter().filter(|b| *b != SWARM_DEFAULT_ATTENTION).cloned());
            model_node("ModelAttentionBackend", "model/patch", "model", vec![combo("attention", &backends).tooltip("The dense attention backend.")])
                .describe("Selects the dense attention implementation for the model.")
        },
        |i, ctx| {
            let chosen = i.string("attention")?;
            // The engine only ever sees its own names: SwarmUI's default means the engine's default.
            let backend = match chosen {
                SWARM_DEFAULT_ATTENTION => ctx.env.attention_backends.first().map(String::as_str).unwrap_or(chosen),
                other => other,
            };
            patch_model(i, ModelPatch::AttentionBackend { backend: backend.to_string() })
        },
    );
    r.add(
        env,
        |_| {
            model_node(
                "BlockSparseAttention",
                "model/patch",
                "model",
                vec![
                    dynamic_combo(
                        "selection",
                        vec![
                            ("sol-attn", vec![float("tau", 1.3, 0.0, 4.0, 0.05)]),
                            ("sla", vec![float("keep_percent", 10.0, 0.5, 95.0, 0.5)]),
                            ("vsa", vec![float("keep_percent", 10.0, 0.5, 95.0, 0.5)]),
                        ],
                    )
                    .tooltip("Method used to choose key blocks for full token-level attention."),
                    float("start_percent", 0.2, 0.0, 1.0, 0.01),
                    float("end_percent", 1.0, 0.0, 1.0, 0.01),
                    string("dense_blocks", "", false).advanced(),
                    int_step("min_tokens", 12288, 0, 1048576, 512).advanced(),
                    int_step("extra_tokens", 256, 0, 256, 64).advanced(),
                    combo_default("sink_conditioning", &["exact_kv", "exact_kv_and_rows", "off"], "exact_kv_and_rows").advanced(),
                    boolean("verbose", false).advanced(),
                ],
            )
            .describe("Applies block-sparse attention to eligible model attention layers, reducing compute for long sequences.")
        },
        |i, _| {
            let method = match i.string("selection")? {
                "sol-attn" => SparseAttentionMethod::SolAttn { tau: i.float("selection.tau")? },
                "sla" => SparseAttentionMethod::Sla { keep_percent: i.float("selection.keep_percent")? },
                "vsa" => SparseAttentionMethod::Vsa { keep_percent: i.float("selection.keep_percent")? },
                other => return Err(input_err(format!("unknown sparse attention method '{other}'"))),
            };
            patch_model(
                i,
                ModelPatch::BlockSparseAttention(SparseAttention {
                    method,
                    start_percent: i.float("start_percent")?,
                    end_percent: i.float("end_percent")?,
                    dense_blocks: i.string("dense_blocks")?.to_string(),
                    min_tokens: i.uint("min_tokens")?,
                    extra_tokens: i.uint("extra_tokens")?,
                    sink_conditioning: i.string("sink_conditioning")?.to_string(),
                    verbose: i.boolean("verbose")?,
                }),
            )
        },
    );
    r.add(
        env,
        |_| model_node("TorchCompileModel", "experimental", "MODEL", vec![combo("backend", &["inductor", "cudagraphs"]).advanced()]),
        |i, _| patch_model(i, ModelPatch::Compile { backend: i.string("backend")?.to_string() }),
    );

    // ------------------------------------------------------------- Seamless Tileable
    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmModelTiling", "Swarm Model Tiling", "SwarmUI/sampling")
                .describe("Adapts a model to use circular padding to enable tiled image results. Only works on UNet based models (eg SDv1, SDXL), not on DiT models (eg SD3, Flux).")
                .input(link("model", "MODEL"))
                .input(combo("tile_axis", &["Both", "X", "Y"]).optional())
                .output("MODEL", "MODEL")
        },
        |i, _| patch_model(i, ModelPatch::Tiling { axis: tile_axis(i)? }),
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmTileableVAE", "Swarm Tileable VAE", "SwarmUI/sampling")
                .describe("Adapts a VAE to use circular padding to enable tiled image results. Use with SwarmModelTiling.")
                .input(link("vae", "VAE"))
                .input(combo("tile_axis", &["Both", "X", "Y"]).optional())
                .output("VAE", "VAE")
        },
        |i, _| Ok(NodeOutput::one(Value::Vae(i.vae("vae")?.with_patch(VaePatch::Tiling { axis: tile_axis(i)? })))),
    );

    // ------------------------------------------------------------- CLIP Stop At Layer
    r.add(
        env,
        |_| {
            NodeSchema::new("CLIPSetLastLayer", "CLIP Set Last Layer", "model/conditioning")
                .input(link("clip", "CLIP"))
                .input(int_step("stop_at_clip_layer", -1, -24, -1, 1).advanced())
                .output("CLIP", "CLIP")
        },
        |i, _| Ok(NodeOutput::one(Value::Clip(i.clip("clip")?.with_patch(TextEncoderPatch::StopAtLayer { layer: i.int("stop_at_clip_layer")? as i32 })))),
    );
}
