//! The custom-sampler graph: `SamplerCustomAdvanced` with its noise, sampler, sigma and guider
//! inputs. SwarmUI builds it for IP2P CFG 2 (SDXL edit models and OmniGen).

use swarm_engine_api::{CustomSampleRequest, Guider, NoiseSource, Opaque, SamplerChoice, ScheduleRequest, Sigmas};

use super::sampling::{SAMPLERS, SCHEDULERS};
use super::{NodeOutput, NodeRegistry};
use crate::env::ServerEnv;
use crate::schema::*;
use crate::value::Value;

fn other<T: std::any::Any + Send + Sync>(ty: &str, v: T) -> Value {
    Value::Other { ty: ty.into(), data: Opaque::new(v) }
}

fn sigmas(s: Sigmas) -> Value {
    other("SIGMAS", s)
}

pub fn register(r: &mut NodeRegistry, env: &ServerEnv) {
    r.add(
        env,
        |_| NodeSchema::new("KSamplerSelect", "KSamplerSelect", "model/sampling/samplers").input(combo("sampler_name", SAMPLERS)).output("SAMPLER", "SAMPLER"),
        |i, _| Ok(NodeOutput::one(other("SAMPLER", SamplerChoice { name: i.string("sampler_name")?.to_string() }))),
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("RandomNoise", "RandomNoise", "model/sampling/noise")
                .input(int("noise_seed", 0, 0, MAX_SEED))
                .output("NOISE", "NOISE")
        },
        |i, _| Ok(NodeOutput::one(other("NOISE", NoiseSource { seed: i.int("noise_seed")? as u64 }))),
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("BasicScheduler", "BasicScheduler", "model/sampling/schedulers")
                .input(link("model", "MODEL"))
                .input(combo("scheduler", SCHEDULERS))
                .input(int("steps", 20, 1, 10000))
                .input(float("denoise", 1.0, 0.0, 1.0, 0.01))
                .output("SIGMAS", "SIGMAS")
        },
        |i, ctx| {
            let (steps, denoise) = (i.uint("steps")?, i.float("denoise")?);
            // As ComfyUI: a partial denoise is the tail of a longer schedule.
            if denoise <= 0.0 {
                return Ok(NodeOutput::one(sigmas(Sigmas(Vec::new()))));
            }
            let total = if denoise < 1.0 { (steps as f64 / denoise) as u32 } else { steps };
            let req = ScheduleRequest::Basic { model: i.model("model")?.clone(), scheduler: i.string("scheduler")?.to_string(), steps: total };
            let full = ctx.engine.model_sigmas(&req)?.0;
            let keep = (steps as usize + 1).min(full.len());
            Ok(NodeOutput::one(sigmas(Sigmas(full[full.len() - keep..].to_vec()))))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("KarrasScheduler", "KarrasScheduler", "model/sampling/schedulers")
                .input(int("steps", 20, 1, 10000))
                .input(float("sigma_max", 14.614642, 0.0, 5000.0, 0.01).advanced())
                .input(float("sigma_min", 0.0291675, 0.0, 5000.0, 0.01).advanced())
                .input(float("rho", 7.0, 0.0, 100.0, 0.01).advanced())
                .output("SIGMAS", "SIGMAS")
        },
        |i, _| Ok(NodeOutput::one(sigmas(Sigmas::karras(i.uint("steps")?, i.float("sigma_min")?, i.float("sigma_max")?, i.float("rho")?)))),
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("SDTurboScheduler", "SDTurboScheduler", "model/sampling/schedulers")
                .input(link("model", "MODEL"))
                .input(int("steps", 1, 1, 10))
                .input(float("denoise", 1.0, 0.0, 1.0, 0.01))
                .output("SIGMAS", "SIGMAS")
        },
        |i, ctx| {
            let req = ScheduleRequest::SdTurbo { model: i.model("model")?.clone(), steps: i.uint("steps")?, denoise: i.float("denoise")? };
            Ok(NodeOutput::one(sigmas(ctx.engine.model_sigmas(&req)?)))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("SplitSigmas", "SplitSigmas", "model/sampling/sigmas")
                .input(link("sigmas", "SIGMAS"))
                .input(int("step", 0, 0, 10000))
                .output("SIGMAS", "high_sigmas")
                .output("SIGMAS", "low_sigmas")
        },
        |i, _| {
            let (hi, lo) = i.other::<Sigmas>("sigmas", "SIGMAS")?.split(i.uint("step")? as usize);
            Ok(NodeOutput::values(vec![sigmas(hi), sigmas(lo)]))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("DualCFGGuider", "Dual CFG Guider", "model/sampling/guiders")
                .input(link("model", "MODEL"))
                .input(link("cond1", "CONDITIONING"))
                .input(link("cond2", "CONDITIONING"))
                .input(link("negative", "CONDITIONING"))
                .input(float("cfg_conds", 8.0, 0.0, 100.0, 0.1))
                .input(float("cfg_cond2_negative", 8.0, 0.0, 100.0, 0.1))
                .input(combo("style", &["regular", "nested"]))
                .output("GUIDER", "GUIDER")
        },
        |i, _| {
            let g = Guider::DualCfg {
                model: i.model("model")?.clone(),
                cond1: i.conditioning("cond1")?.clone(),
                cond2: i.conditioning("cond2")?.clone(),
                negative: i.conditioning("negative")?.clone(),
                cfg_conds: i.float("cfg_conds")?,
                cfg_cond2_negative: i.float("cfg_cond2_negative")?,
                nested: i.string("style")? == "nested",
            };
            Ok(NodeOutput::one(other("GUIDER", g)))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("SamplerCustomAdvanced", "SamplerCustomAdvanced", "model/sampling/custom")
                .input(link("noise", "NOISE"))
                .input(link("guider", "GUIDER"))
                .input(link("sampler", "SAMPLER"))
                .input(link("sigmas", "SIGMAS"))
                .input(link("latent_image", "LATENT"))
                .output("LATENT", "output")
                .output("LATENT", "denoised_output")
        },
        |i, ctx| {
            let req = CustomSampleRequest {
                noise: i.other::<NoiseSource>("noise", "NOISE")?.clone(),
                guider: i.other::<Guider>("guider", "GUIDER")?.clone(),
                sampler: i.other::<SamplerChoice>("sampler", "SAMPLER")?.clone(),
                sigmas: i.other::<Sigmas>("sigmas", "SIGMAS")?.clone(),
                latent: i.latent("latent_image")?.clone(),
            };
            let engine = ctx.engine;
            let out = engine.sample_custom(&req, ctx)?;
            Ok(NodeOutput::values(vec![Value::Latent(out.output), Value::Latent(out.denoised)]))
        },
    );
}
