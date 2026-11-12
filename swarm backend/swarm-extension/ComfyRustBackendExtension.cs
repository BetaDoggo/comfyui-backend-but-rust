using System.Diagnostics;
using System.IO;
using FreneticUtilities.FreneticDataSyntax;
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
