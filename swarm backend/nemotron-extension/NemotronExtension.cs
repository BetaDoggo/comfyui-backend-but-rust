using System.Collections.Generic;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace KreaRust.Nemotron;

/// <summary>Generates with NVIDIA's Nemotron-Labs-Image 30B-A3B on the comfy-rust backend. The model isn't diffusion: a
/// language model writes the image as visual tokens and its own decoder turns them into pixels, all in comfy-rust's
/// <c>NemotronImageGenerate</c> node. When the selected model is one, this replaces the whole normal workflow (loader,
/// prompt encoding, sampler, VAE decode) with that node and its save, and refuses the settings it can't honour.</summary>
public class NemotronExtension : Extension
{
    public const string FeatureID = "nemotron_image";

    public const string ClassID = "nemotron-labs-image";

    public static T2IModelCompatClass CompatNemotron;

    public static T2IRegisteredParam<double> Temperature;

    public static T2IParamGroup Group;

    /// <summary>The file tools/nemotron_image_convert.py writes: its format in the metadata, or the generator's
    /// embeddings beside the image decoder.</summary>
    static bool IsNemotron(JObject h) => h["__metadata__"]?["nemotron_image"] is not null
        || (h.ContainsKey("backbone.embeddings.weight") && h.ContainsKey("image_decoder.decoder.blocks.0.attn.q_proj.weight"));

    /// <summary>Prompt syntax SwarmUI leaves for the backend to act on, none of which this model can.</summary>
    static readonly Regex SwarmTags = new(@"<(break|alternate|alt|fromto|region|object|segment|seg|clear|extend|base|refiner|video|videoswap|edit)\b|\bembedding:", RegexOptions.IgnoreCase);

    public override void OnPreInit()
    {
        ExtensionName = "Nemotron-Labs-Image";
        Description = "Text to image with Nemotron-Labs-Image 30B-A3B, on the comfy-rust backend.";
        License = "GPL-3.0";
    }

    public override void OnInit()
    {
        CompatNemotron = T2IModelClassSorter.RegisterCompat(new() { ID = ClassID, ShortCode = "Nemotron", LorasTargetTextEnc = false });
        T2IModelClassSorter.Register(new() { ID = ClassID, CompatClass = CompatNemotron, Name = "Nemotron-Labs-Image", StandardWidth = 1024, StandardHeight = 1024, IsThisModelOfClass = (m, h) => IsNemotron(h) });
        ComfyUIBackendExtension.NodeToFeatureMap["NemotronImageGenerate"] = FeatureID;
        Group = new("Nemotron-Labs-Image", Toggles: false, Open: false, OrderPriority: -2.3, Description: "Settings for Nemotron-Labs-Image models. Steps, CFG, sampler and scheduler don't apply to them.");
        Temperature = T2IParamTypes.Register<double>(new("Nemotron Temperature", "Softmax temperature of the image tokens, for Nemotron-Labs-Image models only.\nThe model card uses 0.9. 0 always takes the likeliest token.",
            "0.9", Min: 0, Max: 2, Step: 0.01, ViewType: ParamViewType.SLIDER, FeatureFlag: FeatureID, Group: Group, OrderPriority: -10
            ));

        // Before the model loader (-15): the rest of the normal workflow doesn't apply, so this ends it.
        WorkflowGenerator.AddStep(g =>
        {
            T2IModel model = g.UserInput.Get(T2IParamTypes.Model);
            if (model?.ModelClass?.ID != ClassID)
            {
                return;
            }
            if (!g.Features.Contains(FeatureID))
            {
                throw new SwarmUserErrorException("Nemotron-Labs-Image runs only on the comfy-rust backend, which has the NemotronImageGenerate node.");
            }
            string unet = model.ToString(g.ModelFolderFormat);
            g.SkipFurtherSteps = true;
            if (g.UserInput.Get(T2IParamTypes.JustLoadModel, false))
            {
                g.CreateNode("NemotronImageLoad", new JObject() { ["unet_name"] = unet });
                return;
            }
            Refuse(g);
            int width = g.UserInput.GetImageWidth(1024), height = g.UserInput.GetImageHeight(1024);
            if (width % 16 != 0 || height % 16 != 0)
            {
                throw new SwarmUserErrorException($"Nemotron-Labs-Image needs a width and height that are multiples of 16, not {width}x{height}.");
            }
            int batch = g.UserInput.Get(T2IParamTypes.BatchSize, 1);
            if (batch < 1 || batch > 8)
            {
                throw new SwarmUserErrorException($"Nemotron-Labs-Image samples 1 to 8 images together, not a batch size of {batch}.");
            }
            string generated = g.CreateNode("NemotronImageGenerate", new JObject()
            {
                ["unet_name"] = unet,
                ["prompt"] = g.UserInput.Get(T2IParamTypes.Prompt, ""),
                ["width"] = width,
                ["height"] = height,
                ["seed"] = g.UserInput.Get(T2IParamTypes.Seed),
                ["batch_size"] = batch,
                ["temperature"] = g.UserInput.Get(Temperature, 0.9)
            });
            g.CurrentMedia = new WGNodeData([generated, 0], g, WGNodeData.DT_IMAGE, g.CurrentCompat()) { Width = width, Height = height };
            g.CurrentMedia.SaveOutput(null, null, "9");
        }, -16);
    }

    /// <summary>Throws for every setting the model has no use for, rather than generating without it.</summary>
    static void Refuse(WorkflowGenerator g)
    {
        T2IParamInput input = g.UserInput;
        List<string> refused = [];
        void check(bool used, string what)
        {
            if (used)
            {
                refused.Add(what);
            }
        }
        check(input.TryGet(T2IParamTypes.Loras, out List<string> loras) && loras.Count > 0, "LoRAs");
        check(!string.IsNullOrWhiteSpace(input.Get(T2IParamTypes.NegativePrompt, "")), "a negative prompt");
        check(SwarmTags.IsMatch(input.Get(T2IParamTypes.Prompt, "")), "prompt syntax such as <break>, <segment>, <region> or embeddings");
        check(input.TryGet(T2IParamTypes.InitImage, out _), "an init image");
        check(input.TryGet(T2IParamTypes.MaskImage, out _), "a mask");
        check(input.TryGet(T2IParamTypes.PromptImages, out List<Image> images) && images.Count > 0, "prompt images");
        check(input.TryGet(T2IParamTypes.RefinerMethod, out _) || input.TryGet(T2IParamTypes.RefinerModel, out _)
            || (input.TryGet(T2IParamTypes.RefinerUpscale, out double upscale) && upscale != 1), "the refiner or upscale");
        foreach (T2IParamTypes.ControlNetParamHolder cn in T2IParamTypes.Controlnets)
        {
            check(cn is not null && (input.TryGet(cn.Model, out _) || input.TryGet(cn.Image, out _)), "ControlNet");
        }
        check(input.TryGet(T2IParamTypes.VariationSeedStrength, out double variation) && variation > 0, "a variation seed");
        check(input.TryGet(T2IParamTypes.VideoModel, out _), "image to video");
        check(input.Get(T2IParamTypes.RemoveBackground, false), "Remove Background");
        // later stages this workflow ends before: SeedVR, and the RelayVSR extension's
        check(input.TryGet(ComfyUIBackendExtension.SeedVRModel, out _), "SeedVR");
        check(T2IParamTypes.TryGetType("relayvsrmodel", out T2IParamType relayvsr, input) && input.TryGetRaw(relayvsr, out _), "RelayVSR");
        if (refused.Count > 0)
        {
            throw new SwarmUserErrorException($"Nemotron-Labs-Image doesn't support {string.Join(", ", new HashSet<string>(refused))}. Turn them off for this model.");
        }
    }
}
