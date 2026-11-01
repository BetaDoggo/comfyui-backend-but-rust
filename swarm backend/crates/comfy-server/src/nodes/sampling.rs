//! Latent creation, model sampling patches, and samplers.

use swarm_engine_api::{
    EmptyLatentKind, EmptyLatentRequest, Latent, LatentSamples, ModelPatch, PreviewMode, SampleRequest, VariationSeed,
};

use super::{input_err, ExecContext, NodeInputs, NodeOutput, NodeRegistry, NodeResult};
use crate::env::ServerEnv;
use crate::schema::*;
use crate::value::Value;

/// `comfy.samplers.KSampler.SAMPLERS`.
pub const SAMPLERS: &[&str] = &[
    "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2", "exp_heun_2_x0", "exp_heun_2_x0_sde", "dpm_2",
    "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu",
    "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu", "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ipndm", "ipndm_v", "deis", "cfgpp_ud10_ab", "res_multistep", "res_multistep_cfg_pp",
    "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp", "gradient_estimation", "gradient_estimation_cfg_pp", "er_sde", "seeds_2",
    "seeds_3", "sa_solver", "sa_solver_pece", "ddim", "uni_pc", "uni_pc_bh2",
];

/// `comfy.samplers.KSampler.SCHEDULERS`.
pub const SCHEDULERS: &[&str] = &["simple", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta", "normal", "linear_quadratic", "kl_optimal"];

/// Extra schedulers `SwarmKSampler` adds in front of ComfyUI's.
pub const SWARM_EXTRA_SCHEDULERS: &[&str] = &["turbo", "align_your_steps", "ltxv", "ltxv-image", "flux2", "ideogram4", "ideogram4turbo"];

fn swarm_schedulers() -> Vec<&'static str> {
    SWARM_EXTRA_SCHEDULERS.iter().chain(SCHEDULERS).copied().collect()
}

fn opt_sigma(v: f64) -> Option<f64> {
    (v >= 0.0).then_some(v)
}

fn run_sampler(ctx: &mut ExecContext, req: SampleRequest) -> NodeResult {
    let engine = ctx.engine;
    let mut out = engine.sample(&req, ctx)?;
    out.empty = false;
    Ok(NodeOutput::one(Value::Latent(out)))
}

/// Runs `empty_latent` and marks the result as empty (see `Latent::empty`).
fn empty_latent(ctx: &mut ExecContext, kind: EmptyLatentKind, batch_size: u32) -> NodeResult {
    let mut latent = ctx.engine.empty_latent(&EmptyLatentRequest { kind, batch_size })?;
    latent.empty = true;
    Ok(NodeOutput::one(Value::Latent(latent)))
}

fn exec_swarm_ksampler(i: &NodeInputs, ctx: &mut ExecContext) -> NodeResult {
    let var_strength = i.float("var_seed_strength")?;
    let req = SampleRequest {
        model: i.model("model")?.clone(),
        model_negative: i.opt_model("model_negative")?.cloned(),
        positive: i.conditioning("positive")?.clone(),
        negative: i.conditioning("negative")?.clone(),
        latent: i.latent("latent_image")?.clone(),
        noise_seed: i.int("noise_seed")? as u64,
        add_noise: i.string("add_noise")? == "enable",
        steps: i.uint("steps")?,
        start_at_step: i.uint("start_at_step")?,
        end_at_step: i.uint("end_at_step")?,
        return_with_leftover_noise: i.string("return_with_leftover_noise")? == "enable",
        denoise: 1.0,
        cfg: i.float("cfg")?,
        sampler: i.string("sampler_name")?.to_string(),
        scheduler: i.string("scheduler")?.to_string(),
        variation: (var_strength > 0.0).then(|| VariationSeed { seed: i.int("var_seed").unwrap_or(0) as u64, strength: var_strength }),
        sigma_min: opt_sigma(i.float("sigma_min")?),
        sigma_max: opt_sigma(i.float("sigma_max")?),
        rho: i.float("rho")?,
        previews: PreviewMode::parse(i.string("previews")?),
        tile_size: if i.boolean("tile_sample")? { Some(i.uint("tile_size")?) } else { None },
    };
    run_sampler(ctx, req)
}

pub fn register(r: &mut NodeRegistry, env: &ServerEnv) {
    // ------------------------------------------------------------ latents ---
    r.add(
        env,
        |_| {
            NodeSchema::new("EmptyLatentImage", "Empty Latent Image", "model/latent")
                .input(int_step("width", 1024, 16, MAX_RESOLUTION, 8))
                .input(int_step("height", 1024, 16, MAX_RESOLUTION, 8))
                .input(int("batch_size", 1, 1, 4096))
                .output("LATENT", "LATENT")
        },
        |i, ctx| empty_latent(ctx, EmptyLatentKind::Image4Ch { width: i.uint("width")?, height: i.uint("height")? }, i.uint("batch_size")?),
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("EmptySD3LatentImage", "EmptySD3LatentImage", "model/latent/stable diffusion")
                .input(int_step("width", 1024, 16, MAX_RESOLUTION, 16))
                .input(int_step("height", 1024, 16, MAX_RESOLUTION, 16))
                .input(int("batch_size", 1, 1, 4096))
                .output("LATENT", "LATENT")
        },
        |i, ctx| empty_latent(ctx, EmptyLatentKind::Image16Ch { width: i.uint("width")?, height: i.uint("height")? }, i.uint("batch_size")?),
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("EmptyFlux2LatentImage", "Empty Flux 2 Latent", "model/latent/flux")
                .input(int_step("width", 1024, 16, MAX_RESOLUTION, 16))
                .input(int_step("height", 1024, 16, MAX_RESOLUTION, 16))
                .input(int("batch_size", 1, 1, 4096))
                .output("LATENT", "LATENT")
        },
        |i, ctx| empty_latent(ctx, EmptyLatentKind::Flux2 { width: i.uint("width")?, height: i.uint("height")? }, i.uint("batch_size")?),
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmEmptyMiniMaxH3LatentAV", "Swarm Empty MiniMax H3 AV Latent", "SwarmUI/video")
                .describe("Joint video+audio latent for MiniMax H3.")
                .input(int_step("width", 1344, 32, MAX_RESOLUTION, 32))
                .input(int_step("height", 768, 32, MAX_RESOLUTION, 32))
                .input(int_step("length", 124, 1, 3600, 1).tooltip("Frame count at 24 fps, snapped up to the model's 17k+5 grid"))
                .output("LATENT", "LATENT")
        },
        |i, ctx| {
            empty_latent(ctx, EmptyLatentKind::MiniMaxH3AudioVideo { width: i.uint("width")?, height: i.uint("height")?, length: i.uint("length")? }, 1)
        },
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("LTXVSeparateAVLatent", "Separate AV Latent", "model/latent/ltxv")
                .describe("Split a joint AV latent into its video and audio latents (any AV model, e.g. LTXV or MiniMax H3).")
                .input(link("av_latent", "LATENT"))
                .output("LATENT", "video_latent")
                .output("LATENT", "audio_latent")
        },
        |i, _| {
            let l = i.latent("av_latent")?;
            let LatentSamples::AudioVideo { video, audio } = &l.samples else {
                return Err(input_err("LTXVSeparateAVLatent needs a joint audio/video latent"));
            };
            // Each stream takes its own noise mask (ComfyUI unbinds the NestedTensor of masks).
            let v = Latent { audio_noise_mask: None, ..l.with_samples(LatentSamples::Single(video.clone())) };
            let a = Latent { noise_mask: l.audio_noise_mask.clone(), audio_noise_mask: None, ..l.with_samples(LatentSamples::Single(audio.clone())) };
            Ok(NodeOutput::values(vec![Value::Latent(v), Value::Latent(a)]))
        },
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("LTXVConcatAVLatent", "Concat AV Latent", "model/latent/ltxv")
                .input(link("video_latent", "LATENT"))
                .input(link("audio_latent", "LATENT"))
                .output("LATENT", "latent")
        },
        |i, _| {
            let v = i.latent("video_latent")?;
            let a = i.latent("audio_latent")?;
            let (LatentSamples::Single(vt), LatentSamples::Single(at)) = (&v.samples, &a.samples) else {
                return Err(input_err("LTXVConcatAVLatent needs plain video and audio latents"));
            };
            let mut out = v.with_samples(LatentSamples::AudioVideo { video: vt.clone(), audio: at.clone() });
            // A missing mask means "denoise everything", same as ComfyUI filling it with ones.
            out.audio_noise_mask = a.noise_mask.clone();
            out.pixel.audio_seconds = a.pixel.audio_seconds.or(out.pixel.audio_seconds);
            Ok(NodeOutput::one(Value::Latent(out)))
        },
    );

    // ---------------------------------------------------- model patches ---
    r.add(
        env,
        |_| {
            NodeSchema::new("ModelSamplingFlux", "ModelSamplingFlux", "model/patch/flux")
                .input(link("model", "MODEL"))
                .input(float("max_shift", 1.15, 0.0, 100.0, 0.01).advanced())
                .input(float("base_shift", 0.5, 0.0, 100.0, 0.01).advanced())
                .input(int_step("width", 1024, 16, MAX_RESOLUTION, 8))
                .input(int_step("height", 1024, 16, MAX_RESOLUTION, 8))
                .output("MODEL", "MODEL")
        },
        |i, _| {
            let p = ModelPatch::SamplingFlux { max_shift: i.float("max_shift")?, base_shift: i.float("base_shift")?, width: i.uint("width")?, height: i.uint("height")? };
            Ok(NodeOutput::one(Value::Model(i.model("model")?.with_patch(p))))
        },
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("ModelSamplingAuraFlow", "ModelSamplingAuraFlow", "model/patch")
                .input(link("model", "MODEL"))
                .input(float("shift", 1.73, 0.0, 100.0, 0.01))
                .input(combo_default("sampling", &["flow", "img_to_img_velocity"], "flow").optional().advanced())
                .output("MODEL", "MODEL")
        },
        |i, _| {
            let p = ModelPatch::SamplingAuraFlow { shift: i.float("shift")?, sampling: i.opt_string("sampling")?.unwrap_or("flow").to_string() };
            Ok(NodeOutput::one(Value::Model(i.model("model")?.with_patch(p))))
        },
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("ModelSamplingSD3", "ModelSamplingSD3", "model/patch/stable diffusion")
                .input(link("model", "MODEL"))
                .input(float("shift", 3.0, 0.0, 100.0, 0.01))
                .output("MODEL", "MODEL")
        },
        |i, _| Ok(NodeOutput::one(Value::Model(i.model("model")?.with_patch(ModelPatch::SamplingSd3 { shift: i.float("shift")? })))),
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("MiniMaxH3SigmaShift", "ModelSamplingMiniMaxH3", "model/patch/minimax")
                .describe("Set the video/audio flow shifts.")
                .input(link("model", "MODEL"))
                .input(float("shift_video", 12.0, 0.01, 100.0, 0.01))
                .input(float("shift_audio", 3.0, 0.01, 100.0, 0.01))
                .output("MODEL", "MODEL")
        },
        |i, _| {
            let p = ModelPatch::MiniMaxH3Shift { shift_video: i.float("shift_video")?, shift_audio: i.float("shift_audio")? };
            Ok(NodeOutput::one(Value::Model(i.model("model")?.with_patch(p))))
        },
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmAttnTokenWeights", "Swarm Attn Token Weights", "SwarmUI/clip")
                .describe("Applies per-token prompt weights through attention patches instead of scaling the embedding.")
                .input(link("model", "MODEL"))
                .input(link("conditioning", "CONDITIONING"))
                .output("MODEL", "MODEL")
        },
        |i, _| {
            let p = ModelPatch::AttnTokenWeights { conditioning: Box::new(i.conditioning("conditioning")?.clone()) };
            Ok(NodeOutput::one(Value::Model(i.model("model")?.with_patch(p))))
        },
    );

    // ----------------------------------------------------------- samplers ---
    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmKSampler", "SwarmKSampler", "SwarmUI/sampling")
                .describe("Works like a vanilla Comfy KSamplerAdvanced, but with extra inputs for advanced features such as sigma scale, tiling, previews, etc.")
                .input(link("model", "MODEL"))
                .input(int("noise_seed", 0, 0, MAX_SEED))
                .input(int("steps", 20, 1, 10000))
                .input(float("cfg", 8.0, 0.0, 100.0, 0.5))
                .input(combo("sampler_name", SAMPLERS))
                .input(combo("scheduler", &swarm_schedulers()))
                .input(link("positive", "CONDITIONING"))
                .input(link("negative", "CONDITIONING"))
                .input(link("latent_image", "LATENT"))
                .input(int("start_at_step", 0, 0, 10000))
                .input(int("end_at_step", 10000, 0, 10000))
                .input(int("var_seed", 0, 0, MAX_SEED))
                .input(float("var_seed_strength", 0.0, 0.0, 1.0, 0.05))
                .input(float("sigma_max", -1.0, -1.0, 1000.0, 0.01))
                .input(float("sigma_min", -1.0, -1.0, 1000.0, 0.01))
                .input(float("rho", 7.0, 0.0, 100.0, 0.01))
                .input(combo("add_noise", &["enable", "disable"]))
                .input(combo("return_with_leftover_noise", &["disable", "enable"]))
                .input(combo("previews", &PreviewMode::OPTIONS))
                .input(boolean("tile_sample", false))
                .input(int("tile_size", 1024, 256, 4096))
                .input(link("model_negative", "MODEL").optional())
                .output("LATENT", "LATENT")
        },
        exec_swarm_ksampler,
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("KSamplerAdvanced", "KSampler (Advanced)", "model/sampling")
                .input(link("model", "MODEL"))
                .input(combo("add_noise", &["enable", "disable"]))
                .input(int("noise_seed", 0, 0, MAX_SEED))
                .input(int("steps", 20, 1, 10000))
                .input(float("cfg", 8.0, 0.0, 100.0, 0.1))
                .input(combo("sampler_name", SAMPLERS))
                .input(combo("scheduler", SCHEDULERS))
                .input(link("positive", "CONDITIONING"))
                .input(link("negative", "CONDITIONING"))
                .input(link("latent_image", "LATENT"))
                .input(int("start_at_step", 0, 0, 10000))
                .input(int("end_at_step", 10000, 0, 10000))
                .input(combo("return_with_leftover_noise", &["disable", "enable"]))
                .output("LATENT", "LATENT")
        },
        |i, ctx| {
            let req = SampleRequest {
                model: i.model("model")?.clone(),
                model_negative: None,
                positive: i.conditioning("positive")?.clone(),
                negative: i.conditioning("negative")?.clone(),
                latent: i.latent("latent_image")?.clone(),
                noise_seed: i.int("noise_seed")? as u64,
                add_noise: i.string("add_noise")? == "enable",
                steps: i.uint("steps")?,
                start_at_step: i.uint("start_at_step")?,
                end_at_step: i.uint("end_at_step")?,
                return_with_leftover_noise: i.string("return_with_leftover_noise")? == "enable",
                denoise: 1.0,
                cfg: i.float("cfg")?,
                sampler: i.string("sampler_name")?.to_string(),
                scheduler: i.string("scheduler")?.to_string(),
                variation: None,
                sigma_min: None,
                sigma_max: None,
                rho: 7.0,
                previews: PreviewMode::Default,
                tile_size: None,
            };
            run_sampler(ctx, req)
        },
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("KSampler", "KSampler", "model/sampling")
                .input(link("model", "MODEL"))
                .input(int("seed", 0, 0, MAX_SEED))
                .input(int("steps", 20, 1, 10000))
                .input(float("cfg", 8.0, 0.0, 100.0, 0.1))
                .input(combo("sampler_name", SAMPLERS))
                .input(combo("scheduler", SCHEDULERS))
                .input(link("positive", "CONDITIONING"))
                .input(link("negative", "CONDITIONING"))
                .input(link("latent_image", "LATENT"))
                .input(float("denoise", 1.0, 0.0, 1.0, 0.01))
                .output("LATENT", "LATENT")
        },
        |i, ctx| {
            let req = SampleRequest {
                model: i.model("model")?.clone(),
                model_negative: None,
                positive: i.conditioning("positive")?.clone(),
                negative: i.conditioning("negative")?.clone(),
                latent: i.latent("latent_image")?.clone(),
                noise_seed: i.int("seed")? as u64,
                add_noise: true,
                steps: i.uint("steps")?,
                start_at_step: 0,
                end_at_step: 10000,
                return_with_leftover_noise: false,
                denoise: i.float("denoise")?,
                cfg: i.float("cfg")?,
                sampler: i.string("sampler_name")?.to_string(),
                scheduler: i.string("scheduler")?.to_string(),
                variation: None,
                sigma_min: None,
                sigma_max: None,
                rho: 7.0,
                previews: PreviewMode::Default,
                tile_size: None,
            };
            run_sampler(ctx, req)
        },
    );
}
