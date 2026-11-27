//! Compiles csrc/*.cu with nvcc into a static library.
//!
//! Environment overrides:
//!   CUDA_PATH       CUDA toolkit root (default: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0;
//!                   the CUDA installer sets it)
//!   CUTLASS_DIR     CUTLASS checkout (default: ref/comfy-kitchen/third_party/cutlass)
//!   FLASH_ATTN_DIR  flash-attention checkout (default: ref/comfy-kitchen/third_party/flash-attention)
//!   CUDA_ARCH       e.g. "120" (default 120)
//!
//! Compiled units go to `<target>/cuda-objs`, keyed by everything that goes into them, so every profile, `cargo clippy`
//! and the server crates' builds share them, and only a unit whose inputs changed is compiled again.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

const TARGET: &str = "x86_64-pc-windows-msvc";

/// The comfy-kitchen commit the kernels are built and tested against (README, Quick start), and the CUTLASS and
/// flash-attention commits it pins as submodules.
const COMFY_KITCHEN_COMMIT: &str = "f8b7e97a1418d2e529bd5405fbf86312be198ae6";
const CUTLASS_COMMIT: &str = "d4b4b494c3c51bf6507e7ab09fbafd1e9fa94f39";
const FLASH_ATTN_COMMIT: &str = "979702c87a8713a8e0a5e9fee122b90d2ef13be5";

/// The commit checked out in `dir`, when it is a git checkout.
fn git_head(dir: &Path) -> Option<String> {
    let o = Command::new("git").arg("-C").arg(dir).args(["rev-parse", "HEAD"]).output().ok()?;
    o.status.success().then(|| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

fn env_path(var: &str, default: PathBuf) -> PathBuf {
    println!("cargo:rerun-if-env-changed={var}");
    std::env::var_os(var).map(PathBuf::from).unwrap_or(default)
}

fn hash_file_tree(h: &mut DefaultHasher, dir: &Path, recurse: bool) {
    let mut entries: Vec<_> = std::fs::read_dir(dir).unwrap().flatten().collect();
    entries.sort_by_key(|e| e.path());
    for e in entries {
        let p = e.path();
        if p.is_dir() {
            if recurse {
                hash_file_tree(h, &p, true);
            }
        } else if matches!(p.extension().and_then(|s| s.to_str()), Some("cuh" | "h")) {
            p.hash(h);
            std::fs::read(&p).unwrap().hash(h);
        }
    }
}

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    // OUT_DIR is <target>/<profile>/build/diffusion-<hash>/out
    let cache = out.ancestors().nth(4).map_or_else(|| out.clone(), |t| t.join("cuda-objs"));
    std::fs::create_dir_all(&cache).unwrap();
    let csrc = root.join("csrc");
    println!("cargo:rerun-if-changed=csrc");

    let cuda = env_path("CUDA_PATH", PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"));
    let cutlass = env_path("CUTLASS_DIR", root.join("ref/comfy-kitchen/third_party/cutlass"));
    let flash = env_path("FLASH_ATTN_DIR", root.join("ref/comfy-kitchen/third_party/flash-attention"));
    let arch = std::env::var("CUDA_ARCH").unwrap_or_else(|_| "120".into());
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    let flash_src = flash.join("csrc/flash_attn/src");
    assert!(cutlass.join("include/cutlass/cutlass.h").exists(), "CUTLASS not found at {}", cutlass.display());
    assert!(flash_src.join("flash.h").exists(), "flash-attention not found at {}", flash.display());

    let cl = cc::windows_registry::find_tool(TARGET, "cl.exe").expect("MSVC cl.exe not found");
    let cl_dir = cl.path().parent().unwrap().to_path_buf();
    let nvcc = cuda.join("bin/nvcc.exe");

    // Object cache key: our headers (all units) + the unit source itself.
    let mut base_hash = DefaultHasher::new();
    // (vendored code is keyed by the headers next to each unit instead)
    hash_file_tree(&mut base_hash, &csrc, false);
    hash_file_tree(&mut base_hash, &csrc.join("compat"), true);
    arch.hash(&mut base_hash);
    cuda.hash(&mut base_hash);
    // A different CUTLASS or flash-attention means different headers, so it rebuilds every unit; and anything but the
    // tested commits gets a warning.
    for (name, dir, want) in [("CUTLASS", &cutlass, CUTLASS_COMMIT), ("flash-attention", &flash, FLASH_ATTN_COMMIT)] {
        let head = git_head(dir);
        if let Some(h) = head.as_deref().filter(|h| *h != want) {
            let at = dir.display();
            println!("cargo:warning={name} at {at} is at {h}, not the tested {want} (comfy-kitchen {COMFY_KITCHEN_COMMIT}): see the README's Quick start");
        }
        head.hash(&mut base_hash);
    }

    struct Unit {
        src: PathBuf,
        fast_math: bool,
        quiet: bool,
    }
    let units = vec![
        Unit { src: csrc.join("kernels.cu"), fast_math: false, quiet: false },
        Unit { src: csrc.join("gemm.cu"), fast_math: false, quiet: true },
        Unit { src: csrc.join("h3.cu"), fast_math: false, quiet: false },
        Unit { src: csrc.join("trellis.cu"), fast_math: false, quiet: false },
        Unit { src: csrc.join("triposplat.cu"), fast_math: false, quiet: false },
        Unit { src: csrc.join("birefnet.cu"), fast_math: false, quiet: false },
        Unit { src: csrc.join("qwen21.cu"), fast_math: false, quiet: false },
        Unit { src: csrc.join("sensenova.cu"), fast_math: false, quiet: false },
        Unit { src: csrc.join("nemotron.cu"), fast_math: false, quiet: false },
        Unit { src: csrc.join("seedvr.cu"), fast_math: false, quiet: false },
        Unit { src: csrc.join("relayvsr.cu"), fast_math: false, quiet: false },
        Unit { src: csrc.join("sd.cu"), fast_math: false, quiet: true },
        Unit { src: csrc.join("vendor/w6a8_requant.cu"), fast_math: false, quiet: true },
        Unit { src: csrc.join("attn.cu"), fast_math: true, quiet: true },
        Unit { src: csrc.join("vendor/cutlass_gemm_int8.cu"), fast_math: false, quiet: true },
        Unit { src: csrc.join("attn_int8.cu"), fast_math: false, quiet: false },
        Unit { src: csrc.join("wddm.cu"), fast_math: false, quiet: false },
        Unit { src: csrc.join("vendor/sage/sage_attn_launcher.cu"), fast_math: true, quiet: true },
        Unit { src: csrc.join("vendor/sage/quant_qk_int8.cu"), fast_math: true, quiet: true },
        Unit { src: csrc.join("vendor/sage/quant_v_int8.cu"), fast_math: true, quiet: true },
        Unit { src: csrc.join("sparse_attn.cu"), fast_math: false, quiet: false },
        Unit { src: csrc.join("sol3_attn.cu"), fast_math: false, quiet: false },
    ];
    // comfy-kitchen's Sol-Attn kernels, built with its flags
    let mut units = units;
    for f in ["sol_attn", "sol_attn_exact", "sol_attn_preprocess", "sol_attn_producer", "sol_attn_route", "sol_attn_token", "sol_attn_vtranspose"] {
        units.push(Unit { src: csrc.join(format!("vendor/sage/{f}.cu")), fast_math: true, quiet: true });
    }

    let includes = [
        csrc.clone(),
        csrc.join("compat"),
        csrc.join("vendor"),
        csrc.join("vendor/sage"),
        cutlass.join("include"),
        cutlass.join("tools/util/include"),
        flash_src.clone(),
    ];

    let mut handles = Vec::new();
    let mut stems = Vec::new();
    for u in units {
        let mut h = base_hash.clone();
        u.src.hash(&mut h);
        std::fs::read(&u.src).unwrap().hash(&mut h);
        if u.src.starts_with(csrc.join("vendor")) {
            hash_file_tree(&mut h, u.src.parent().unwrap(), false);
        }
        u.fast_math.hash(&mut h);
        let key = h.finish();
        let stem = u.src.file_stem().unwrap().to_string_lossy().to_string();
        let obj = cache.join(format!("{stem}-{key:016x}.obj"));
        // written under another name, then renamed: a build running beside this one never links half an object
        let tmp = cache.join(format!("{stem}-{key:016x}.{}.tmp", std::process::id()));
        stems.push(stem);
        // nvcc would otherwise run vcvars64.bat, which breaks on unusual PATH entries; a clean,
        // ready MSVC environment (cl.exe on PATH, INCLUDE/LIB set) skips that step.
        let mut cmd = Command::new(&nvcc);
        cmd.env_clear();
        for var in ["SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "USERPROFILE", "PROGRAMDATA", "ALLUSERSPROFILE",
            "LOCALAPPDATA", "APPDATA", "NUMBER_OF_PROCESSORS"] {
            if let Some(v) = std::env::var_os(var) {
                cmd.env(var, v);
            }
        }
        cmd.env("PATH", format!(r"{};C:\Windows\System32;{}", cl_dir.display(), cuda.join("bin").display()));
        for (k, v) in cl.get_envs() {
            if !k.eq_ignore_ascii_case("PATH") {
                cmd.env(k, v);
            }
        }
        cmd.arg("-c").arg(&u.src).arg("-o").arg(&tmp);
        cmd.args(["-O3", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-DNDEBUG", "-D_USE_MATH_DEFINES"]);
        cmd.arg(format!("-gencode=arch=compute_{arch},code=sm_{arch}"));
        cmd.arg("-Xcompiler=/MD,/O2,/Zc:__cplusplus,/Zc:preprocessor,/EHsc,/bigobj");
        cmd.args(["-DCOMFY_HAVE_CUTLASS", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"]);
        if u.fast_math {
            cmd.arg("--use_fast_math");
        }
        if u.quiet {
            cmd.arg("-w");
        }
        for i in &includes {
            cmd.arg(format!("-I{}", i.display()));
        }
        if obj.exists() {
            handles.push((obj, tmp, None));
            continue;
        }
        let src = u.src.clone();
        handles.push((
            obj,
            tmp,
            Some(std::thread::spawn(move || {
                let t = std::time::Instant::now();
                let o = cmd.output().expect("failed to run nvcc");
                (src, o, t.elapsed())
            })),
        ));
    }

    let mut objs = Vec::new();
    let mut failed = false;
    for (obj, tmp, h) in handles {
        if let Some(h) = h {
            let (src, o, dt) = h.join().unwrap();
            if !o.status.success() {
                failed = true;
                eprintln!("nvcc failed for {}:\n{}\n{}", src.display(), String::from_utf8_lossy(&o.stdout),
                    String::from_utf8_lossy(&o.stderr));
                let _ = std::fs::remove_file(&tmp);
            } else {
                if std::fs::rename(&tmp, &obj).is_err() {
                    // another build finished the same unit first
                    assert!(obj.exists(), "couldn't move {} to {}", tmp.display(), obj.display());
                    let _ = std::fs::remove_file(&tmp);
                }
                println!("cargo:warning=compiled {} in {:.0}s", src.file_name().unwrap().to_string_lossy(), dt.as_secs_f64());
            }
        }
        objs.push(obj);
    }
    if failed {
        panic!("CUDA compilation failed");
    }
    // Drop each unit's older objects. Units of another CUDA_ARCH go too, and are compiled again when it comes back.
    let current: std::collections::HashSet<_> = objs.iter().filter_map(|o| o.file_name()).map(|n| n.to_owned()).collect();
    for e in std::fs::read_dir(&cache).unwrap().flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let Some(key) = name.strip_suffix(".obj").and_then(|n| n.rsplit_once('-')) else { continue };
        let ours = stems.iter().any(|s| *s == key.0) && key.1.len() == 16 && key.1.chars().all(|c| c.is_ascii_hexdigit());
        if ours && !current.contains(&e.file_name()) {
            let _ = std::fs::remove_file(e.path());
        }
    }

    let lib_tool = cc::windows_registry::find_tool(TARGET, "lib.exe").expect("lib.exe not found");
    let lib = out.join("gpu_kernels.lib");
    let mut cmd = lib_tool.to_command();
    cmd.arg("/nologo").arg(format!("/OUT:{}", lib.display()));
    for o in &objs {
        cmd.arg(o);
    }
    let st = cmd.status().expect("lib.exe failed to start");
    assert!(st.success(), "lib.exe failed");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=gpu_kernels");
    println!("cargo:rustc-link-search=native={}", cuda.join("lib/x64").display());
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    println!("cargo:rustc-link-lib=dylib=cuda");
}
