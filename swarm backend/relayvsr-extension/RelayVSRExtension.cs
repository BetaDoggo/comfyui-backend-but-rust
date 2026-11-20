using System.IO;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;
using SwarmUI.Accounts;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace KreaRust.RelayVSR;

/// <summary>Adds a RelayVSR stage: the finished image or video is upscaled x2 or x4 by Comfy-Org's RelayVSR, as its
/// example workflow does. Every 15th frame and the last go through the keyframe model (Wan 2.2 TI2V 5B) in one euler step
/// at CFG 1, then the Flash decoder restores every frame from those keyframes and the low-quality video. Runs on any
/// ComfyUI-type backend with the <c>RelayVSRConditioning</c> and <c>VSRDecode</c> nodes: comfy-rust, or a ComfyUI new
/// enough to have them.</summary>
public class RelayVSRExtension : Extension
{
    public const string FeatureID = "relayvsr";

    public static T2IModelCompatClass CompatRelayVSR;

    public static T2IRegisteredParam<T2IModel> Model;

    public static T2IRegisteredParam<string> Decoder;

    public static T2IRegisteredParam<double> Detail;

    public static T2IParamGroup Group;

    /// <summary>The keyframe model's tensors start with its low-quality projection beside a Wan patch embedding.</summary>
    static bool IsRelayVSR(JObject h) => (h.ContainsKey("lq_proj_in.conv1.weight") && h.ContainsKey("patch_embedding.weight"))
        || (h.ContainsKey("model.diffusion_model.lq_proj_in.conv1.weight") && h.ContainsKey("model.diffusion_model.patch_embedding.weight"));

    static bool IsFlashDecoder(JObject h) => h.ContainsKey("lr_encoder.layers.0.weight") && h.ContainsKey("condition_encoder.projection.weight");

    public override void OnPreInit()
    {
        ExtensionName = "RelayVSR";
        Description = "Video upscaling x2 or x4 with RelayVSR, as a stage after the generation.";
        License = "GPL-3.0";
    }

    public override void OnInit()
    {
        CompatRelayVSR = T2IModelClassSorter.RegisterCompat(new() { ID = "relayvsr", ShortCode = "RelayVSR", LorasTargetTextEnc = false });
        // The keyframe model is a Wan 2.2 TI2V 5B, so SwarmUI's own class for that, registered first, would claim it.
        if (T2IModelClassSorter.ModelClasses.TryGetValue("wan-2_2-ti2v-5b", out T2IModelClass wan5b))
        {
            Func<T2IModel, JObject, bool> isWan = wan5b.IsThisModelOfClass;
            wan5b.IsThisModelOfClass = (m, h) => !IsRelayVSR(h) && isWan(m, h);
        }
        T2IModelClassSorter.Register(new() { ID = "relayvsr", CompatClass = CompatRelayVSR, Name = "RelayVSR", StandardWidth = 960, StandardHeight = 960, IsThisModelOfClass = (m, h) => IsRelayVSR(h) });
        T2IModelClassSorter.Register(new() { ID = "relayvsr/vae", CompatClass = CompatRelayVSR, Name = "RelayVSR Flash Decoder", StandardWidth = 960, StandardHeight = 960, IsThisModelOfClass = (m, h) => IsFlashDecoder(h) });
        ComfyUIBackendExtension.NodeToFeatureMap["RelayVSRConditioning"] = FeatureID;

        Group = new("RelayVSR", Toggles: true, Open: false, OrderPriority: -2.4, Description: "RelayVSR upscales the result of the normal generation, a video or a picture, by 2 or 4.");
        Model = T2IParamTypes.Register<T2IModel>(new("RelayVSR Model", "Which RelayVSR model to upscale with. Its file sets the factor: x2 or x4.",
            "None", IgnoreIf: "None", FeatureFlag: FeatureID, Group: Group, Subtype: "Stable-Diffusion", ChangeWeight: 9, DoNotPreview: true, OrderPriority: -10,
            GetValues: (session) => ["None", .. T2IParamTypes.CleanModelList(Program.MainSDModels.ListModelsFor(session).Where(m => m.ModelClass?.CompatClass?.ID == "relayvsr").OrderBy(m => m.Name).Select(m => m.Name))]
            ));
        Decoder = T2IParamTypes.Register<string>(new("RelayVSR Decoder", "The RelayVSR Flash decoder, of the same factor as the model.\n'Automatic' picks the one whose file name has the model's x2 or x4.",
            "Automatic", FeatureFlag: FeatureID, Group: Group, DoNotPreview: true, OrderPriority: -9,
            GetValues: (session) => ["Automatic", .. T2IParamTypes.CleanModelList(FlashDecoders(session).Select(m => m.Name))]
            ));
        Detail = T2IParamTypes.Register<double>(new("RelayVSR Detail", "How much of the detail the decoder adds to the plain bicubic upscale of each frame is kept.\n1 is the model's full output, 0 is a bicubic upscale.",
            "1", Min: 0, Max: 1, Step: 0.01, ViewType: ParamViewType.SLIDER, FeatureFlag: FeatureID, Group: Group, DoNotPreview: true, OrderPriority: -8
            ));

        // After SeedVR (40) and before the final cleanups (50), on the decoded frames.
        WorkflowGenerator.AddStep(g =>
        {
            if (!g.UserInput.TryGet(Model, out T2IModel model) || model is null)
            {
                return;
            }
            if (!g.Features.Contains(FeatureID))
            {
                throw new SwarmUserErrorException("RelayVSR is enabled, but the backend has no RelayVSRConditioning node. Use the comfy-rust backend, or update ComfyUI.");
            }
            T2IModel decoder = PickDecoder(g, model);
            g.SaveOptionalIntermediate();
            WGNodeData media = g.CurrentMedia.AsRawImage(g.CurrentVae);
            long seed = g.UserInput.Get(T2IParamTypes.Seed) + 10;
            string unet = g.CreateNode("UNETLoader", new JObject()
            {
                ["unet_name"] = model.ToString(g.ModelFolderFormat),
                ["weight_dtype"] = "default"
            });
            string cond = g.CreateNode("RelayVSRConditioning", new JObject()
            {
                ["model"] = new JArray(unet, 0),
                ["images"] = media.Path
            });
            // the example workflow's sampler: one euler step at CFG 1
            string sampled = g.CreateNode("KSampler", new JObject()
            {
                ["model"] = new JArray(unet, 0),
                ["seed"] = seed,
                ["steps"] = 1,
                ["cfg"] = 1.0,
                ["sampler_name"] = "euler",
                ["scheduler"] = "simple",
                ["positive"] = new JArray(cond, 0),
                ["negative"] = new JArray(cond, 0),
                ["latent_image"] = new JArray(cond, 1),
                ["denoise"] = 1.0
            });
            string vae = g.CreateNode("VAELoader", new JObject()
            {
                ["vae_name"] = decoder.ToString(g.ModelFolderFormat)
            });
            string decoded = g.CreateNode("VSRDecode", new JObject()
            {
                ["vae"] = new JArray(vae, 0),
                ["samples"] = new JArray(sampled, 0),
                ["images"] = media.Path,
                ["detail"] = g.UserInput.Get(Detail, 1)
            });
            WGNodeData result = media.WithPath([decoded, 0], mayHaveAlpha: false);
            int? factor = FactorOf(model.Name);
            result.Width = media.Width * factor;
            result.Height = media.Height * factor;
            g.CurrentMedia = result;
        }, 41);
    }

    static IEnumerable<T2IModel> FlashDecoders(Session session) =>
        Program.T2IModelSets["VAE"].ListModelsFor(session).Where(m => m.ModelClass?.CompatClass?.ID == "relayvsr").OrderBy(m => m.Name);

    /// <summary>The factor a released file's name gives, as in <c>relayvsr_x2_...</c>; null when it gives none.</summary>
    static int? FactorOf(string name)
    {
        Match m = Regex.Match(Path.GetFileName(name), @"(?:^|[^a-z0-9])x([24])(?:[^0-9]|$)", RegexOptions.IgnoreCase);
        return m.Success ? int.Parse(m.Groups[1].Value) : null;
    }

    static T2IModel PickDecoder(WorkflowGenerator g, T2IModel model)
    {
        string chosen = g.UserInput.Get(Decoder, "Automatic");
        if (chosen != "Automatic")
        {
            return Program.T2IModelSets["VAE"].GetModel(chosen) ?? throw new SwarmUserErrorException($"RelayVSR decoder '{chosen}' not found.");
        }
        List<T2IModel> all = [.. FlashDecoders(g.UserInput.SourceSession)];
        if (all.Count == 0)
        {
            throw new SwarmUserErrorException("RelayVSR needs its Flash decoder in your VAE folder (relayvsr_flash_vae_x2_fp32.safetensors or relayvsr_flash_vae_x4_fp32.safetensors).");
        }
        int? factor = FactorOf(model.Name);
        List<T2IModel> same = factor is null ? all : [.. all.Where(d => FactorOf(d.Name) == factor)];
        if (same.Count == 1)
        {
            return same[0];
        }
        if (same.Count == 0)
        {
            throw new SwarmUserErrorException($"No RelayVSR Flash decoder for x{factor} in your VAE folder ({model.Name} is x{factor}): add relayvsr_flash_vae_x{factor}_fp32.safetensors.");
        }
        throw new SwarmUserErrorException($"Can't tell which RelayVSR decoder goes with '{model.Name}': choose one in RelayVSR Decoder.");
    }
}
