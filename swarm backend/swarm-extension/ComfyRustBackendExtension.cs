using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;
using FreneticUtilities.FreneticDataSyntax;
using Newtonsoft.Json.Linq;
using SwarmUI.Backends;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace KreaRust.ComfyRustBackend;

/// <summary>Adds a backend type that SwarmUI starts and stops itself, like its own ComfyUI self-start.
/// The stock type can't be used: it always runs ComfyUI's Python and pip setup, which this server has no use for.</summary>
public class ComfyRustBackendExtension : Extension
{
    public override void OnPreInit()
    {
        ExtensionName = "comfy-rust backend";
        Description = "Starts comfy-rust, the native Rust comfy-server, as a SwarmUI backend.";
        License = "GPL-3.0";
    }

    public override void OnInit()
    {
        Program.Backends.RegisterBackendType<ComfyRustSelfStartBackend>("comfy_rust_selfstart", "comfy-rust Self-Starting", "The native Rust engine, automatically launched and managed by this UI server.", isStandard: true);
        // HyperFlow H3 Flash drops blocks, so SwarmUI's own MiniMax H3 check (which needs block 49) misses it.
        // As MiniMax H3's compat class it gets H3's workflow, references included.
        T2IModelClassSorter.Register(new() { ID = "minimax-h3/hyperflow", CompatClass = T2IModelClassSorter.CompatMiniMaxH3, Name = "HyperFlow H3 Flash", StandardWidth = 960, StandardHeight = 960, IsThisModelOfClass = (m, h) =>
        {
            bool has(string key) => h.ContainsKey(key) || h.ContainsKey($"diffusion_model.{key}") || h.ContainsKey($"model.diffusion_model.{key}");
            return has("video_patch_proj.weight") && has("audio_patch_proj.weight") && has("endpoint_time_embedder.proj_in.weight");
        }});
        Program.ModelPathsChangedEvent += OnModelPathsChanged;
        // After SwarmUI's decode cleanup (200): a pi-H3 DWD adapter decodes the video itself at the last step
        WorkflowGenerator.AddStep(UseDwdDecode, 201);
    }

    /// <summary>Whether each LoRA file is a LakonLab DWD adapter (pi-H3's), by path, size and time.</summary>
    public static ConcurrentDictionary<string, bool> DwdAdapters = new();

    /// <summary>True if the safetensors file at <paramref name="path"/> has a DWD pixel head, as pi-H3's DWD adapters do.</summary>
    public static bool IsDwdAdapter(string path)
    {
        FileInfo info = new(path);
        if (!info.Exists)
        {
            return false;
        }
        return DwdAdapters.GetOrAdd($"{path}|{info.Length}|{info.LastWriteTimeUtc.Ticks}", _ =>
        {
            using FileStream f = File.OpenRead(path);
            byte[] len = new byte[8];
            if (f.Read(len, 0, 8) != 8)
            {
                return false;
            }
            long n = BitConverter.ToInt64(len, 0);
            if (n <= 0 || n > 100_000_000)
            {
                return false;
            }
            byte[] header = new byte[n];
            f.ReadExactly(header, 0, (int)n);
            return System.Text.Encoding.UTF8.GetString(header).Contains("\"dwd_norm_out.linear.weight\"");
        });
    }

    /// <summary>The LoRA file a node's <c>lora_name</c> (or <c>adapter_name</c>) names, if SwarmUI knows it.</summary>
    public static string LoraPath(string name)
    {
        T2IModelHandler loras = Program.T2IModelSets["LoRA"];
        name = name.Replace('\\', '/');
        foreach (string key in new[] { name, $"{name}.safetensors" })
        {
            if (loras.Models.TryGetValue(key, out T2IModel m))
            {
                return m.RawFilePath;
            }
        }
        return null;
    }

    /// <summary>Whether the model at <paramref name="path"/> in the workflow passes through a LoRA loader with a DWD adapter.</summary>
    public static bool HasDwdAdapter(WorkflowGenerator g, JArray path)
    {
        for (int depth = 0; depth < 64 && path is not null; depth++)
        {
            if (g.Workflow[$"{path[0]}"] is not JObject node)
            {
                return false;
            }
            string type = $"{node["class_type"]}";
            JObject inputs = node["inputs"] as JObject;
            if (type is "LoraLoader" or "LoraLoaderModelOnly" or "Load LakonLab Adapter")
            {
                string name = $"{inputs?["lora_name"] ?? inputs?["adapter_name"]}";
                string file = LoraPath(name);
                if (file is not null && IsDwdAdapter(file))
                {
                    return true;
                }
            }
            path = inputs?["model"] as JArray;
        }
        return false;
    }

    /// <summary>On MiniMax H3, a video decode of a sampler's output whose model has a pi-H3 DWD adapter becomes ComfyUI-piFlow's
    /// DWD Decode: the adapter's pixel head already decoded the frames at the last step, in place of the VAE.</summary>
    public static void UseDwdDecode(WorkflowGenerator g)
    {
        if (!g.IsMiniMaxH3())
        {
            return;
        }
        void swap(string id, JObject data)
        {
            JArray samples = data["inputs"]?["samples"] as JArray;
            if (samples is null || g.Workflow[$"{samples[0]}"] is not JObject source)
            {
                return;
            }
            // SwarmUI splits H3's joint latent before decoding its video
            if ($"{source["class_type"]}" == "LTXVSeparateAVLatent" && $"{samples[1]}" == "0" && source["inputs"]?["av_latent"] is JArray av
                && g.Workflow[$"{av[0]}"] is JObject joint)
            {
                source = joint;
            }
            if ($"{source["class_type"]}" is not ("SwarmKSampler" or "KSamplerAdvanced" or "KSampler"))
            {
                return;
            }
            if (HasDwdAdapter(g, source["inputs"]?["model"] as JArray))
            {
                data["class_type"] = "DWD Decode";
                data["inputs"] = new JObject() { ["samples"] = samples };
            }
        }
        g.RunOnNodesOfClass("VAEDecode", swap);
        g.RunOnNodesOfClass("VAEDecodeTiled", swap);
    }

    public void OnModelPathsChanged()
    {
        ComfyUISelfStartBackend.IsComfyModelFileEmitted = false;
        foreach (ComfyRustSelfStartBackend backend in Program.Backends.RunningBackendsOfType<ComfyRustSelfStartBackend>())
        {
            if (backend.IsEnabled)
            {
                Program.Backends.ReloadBackend(backend.BackendData).Wait(Program.GlobalProgramCancel);
            }
        }
    }
}

public class ComfyRustSelfStartBackend : ComfyUIAPIAbstractBackend
{
    public class ComfyRustSelfStartSettings : AutoConfiguration
    {
        [ConfigComment("The location of 'selfstart.bat' in the repository's 'swarm backend' folder.")]
        public string StartScript = "";

        [ConfigComment("Any arguments to add to the server launch, such as '--ffmpeg <path>', '--reserve-vram 512' or '--max-side 2048'.")]
        public string ExtraArgs = "";

        [ConfigComment("Whether to send latent previews while sampling.")]
        public bool EnablePreviews = false;

        [ConfigComment("Whether to run bf16 checkpoints (MOSS Image) in their original precision.\nBy default they are quantized to int8 as they load: about 2.5 times as fast and half the VRAM, at a small cost in quality. Int8 checkpoints run as stored either way. Restart the backend after changing it.")]
        public bool NoQuant = false;

        [ConfigComment("Whether to use faster paths whose images aren't byte-identical to the default ones.\nAnima's int8 cross-attention skips the prompt's padding (exact in the math, about 3% faster per step), and int8 MOSS Image blocks use a fast activation (about 3% faster). Images change slightly, about as much as switching the attention backend does. Restart the backend after changing it.")]
        public bool Lossy = false;

        [ConfigComment("Which GPU to use, if multiple are available.\nShould be a single number, like '0'.")]
        public string GPU_ID = "0";

        [ConfigComment("How many extra requests may queue up on this backend while one is processing.")]
        public int OverQueue = 1;

        [ConfigComment("If checked, if the backend crashes it will automatically restart.\nIf false, if the backend crashes it will sit in an errored state until manually restarted.")]
        public bool AutoRestart = true;
    }

    public ComfyRustSelfStartSettings Settings => SettingsRaw as ComfyRustSelfStartSettings;

    public Process RunningProcess;

    public int Port;

    public override string APIAddress => $"http://localhost:{Port}";

    public override string WebAddress => $"http://localhost:{Port}";

    public override bool CanIdle => false;

    public override int OverQueue => Settings.OverQueue;

    public override async Task Init()
    {
        AddLoadStatus("Starting init...");
        // The same model paths file SwarmUI gives ComfyUI, covering every model root and folder setting.
        new ComfyUISelfStartBackend() { AbstractBackendData = AbstractBackendData }.EnsureComfyFile();
        Settings.StartScript = Settings.StartScript.Trim(' ', '"', '\'', '\n', '\r', '\t');
        if (string.IsNullOrWhiteSpace(Settings.StartScript))
        {
            AddLoadStatus("Start script is empty, cannot load.");
            Status = BackendStatus.DISABLED;
            return;
        }
        string modelPaths = Path.GetFullPath($"{Program.DataDir}/comfy-auto-model.yaml");
        string args = $"{Settings.ExtraArgs.Trim()} --port {{PORT}} --extra-model-paths-config \"{modelPaths}\"";
        if (Settings.EnablePreviews)
        {
            args += " --live-previews";
        }
        if (Settings.NoQuant)
        {
            args += " --no-quant";
        }
        if (Settings.Lossy)
        {
            args += " --lossy";
        }
        AddLoadStatus("Starting self-start comfy-rust process...");
        await NetworkBackendUtils.DoSelfStart(Settings.StartScript, this, $"ComfyRust-{BackendData.ID}", $"backend-{BackendData.ID}", Settings.GPU_ID, args, InitInternal, (p, r) => { Port = p; RunningProcess = r; }, Settings.AutoRestart);
    }

    public override async Task Shutdown()
    {
        await base.Shutdown();
        try
        {
            if (RunningProcess is null || RunningProcess.HasExited)
            {
                return;
            }
            Logs.Info($"Shutting down self-start comfy-rust backend (port={Port}) process #{RunningProcess.Id}...");
            Utilities.KillProcess(RunningProcess, 10);
        }
        catch (Exception ex)
        {
            Logs.Error($"Error stopping comfy-rust backend process: {ex.ReadableString()}");
        }
    }
}
