using System.IO;
using FreneticUtilities.FreneticExtensions;
using System.Net.Http;
using System.Net.WebSockets;
using System.Text;
using Newtonsoft.Json.Linq;
using SwarmUI.Accounts;
using SwarmUI.Backends;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Core;
using SwarmUI.Utils;
using SwarmUI.WebAPI;

namespace KreaRust.Pixal3D;

/// <summary>A tab that turns a photo of an object into a 3D model with Pixal3D. It runs the <c>Pixal3DImageTo3D</c> node on a
/// ComfyUI-type backend: comfy-rust has it natively, and ComfyUI self-start backends get it from <c>comfy_nodes</c>. An image
/// without a transparent background is first cut out by ComfyUI's own <c>RemoveBackground</c> node and a BiRefNet model.</summary>
public class Pixal3DExtension : Extension
{
    public static PermInfo PermGenerate = Permissions.Register(new("pixal3d_generate", "Pixal3D Generate", "Allows generating 3D models in the Pixal3D tab.", PermissionDefault.USER, Permissions.GroupUser, PermSafetyLevel.SAFE));

    public const string NodeClass = "Pixal3DImageTo3D";

    /// <summary>ComfyUI's background removal loader, which lists the BiRefNet files.</summary>
    public const string BackgroundLoaderClass = "LoadBackgroundRemovalModel";

    public override void OnPreInit()
    {
        ExtensionName = "Pixal3D";
        Description = "Image to 3D with Pixal3D, in its own tab.";
        License = "GPL-3.0";
        ScriptFiles.Add("Assets/pixal3d.js");
        StyleSheetFiles.Add("Assets/pixal3d.css");
    }

    public override void OnInit()
    {
        ComfyUISelfStartBackend.CustomNodePaths.Add(Path.GetFullPath($"{FilePath}comfy_nodes"));
        API.RegisterAPICall(Pixal3DListModels, false, PermGenerate);
        API.RegisterAPICall(Pixal3DGenerateWS, true, PermGenerate);
    }

    static HttpClient Http => ComfyUIAPIAbstractBackend.HttpClient;

    /// <summary>A node's schema on a running ComfyUI-type backend, or null where it isn't installed.</summary>
    static async Task<JObject> NodeInfo(ComfyUIAPIAbstractBackend backend, string nodeClass = NodeClass)
    {
        try
        {
            string raw = await Http.GetStringAsync($"{backend.APIAddress}/object_info/{nodeClass}");
            return JObject.Parse(raw)[nodeClass] as JObject;
        }
        catch (Exception ex)
        {
            Logs.Debug($"Pixal3D: no {nodeClass} on {backend.APIAddress}: {ex.Message}");
            return null;
        }
    }

    static List<string> Options(JObject info, string input) => info?["input"]?["required"]?[input]?[0] is JArray a ? [.. a.Select(v => $"{v}")] : [];

    [API.APIDescription("The model files the Pixal3D node can use, read from the first backend that has it.", """
        "unet_name": ["Pixal3D/pixal3d_int8_convrot.safetensors", ...], "clip_vision_name": [...], "shape_vae_name": [...], "texture_vae_name": [...],
        "bg_removal_name": ["birefnet.safetensors", ...]  // empty where the backend has no background removal
        """)]
    public static async Task<JObject> Pixal3DListModels(Session session)
    {
        foreach (ComfyUIAPIAbstractBackend backend in Program.Backends.RunningBackendsOfType<ComfyUIAPIAbstractBackend>())
        {
            JObject info = await NodeInfo(backend);
            if (info is not null)
            {
                return new JObject()
                {
                    ["unet_name"] = JArray.FromObject(Options(info, "unet_name")),
                    ["clip_vision_name"] = JArray.FromObject(Options(info, "clip_vision_name")),
                    ["shape_vae_name"] = JArray.FromObject(Options(info, "shape_vae_name")),
                    ["texture_vae_name"] = JArray.FromObject(Options(info, "texture_vae_name")),
                    ["bg_removal_name"] = JArray.FromObject(Options(await NodeInfo(backend, BackgroundLoaderClass), "bg_removal_name"))
                };
            }
        }
        return new JObject() { ["error"] = $"No running backend has the {NodeClass} node. Use the comfy-rust backend, or a ComfyUI self-start backend (restart it after installing this extension)." };
    }

    static readonly string[] IntInputs = ["seed", "resolution", "structure_steps", "shape_steps", "refine_steps", "texture_steps", "max_faces"];
    static readonly string[] FileInputs = ["unet_name", "clip_vision_name", "shape_vae_name", "texture_vae_name"];

    [API.APIDescription("Turns an image into a 3D model: its transparent background is the mask, or with remove_background a BiRefNet model (bg_removal_name) cuts the object out first. Sends progress updates, then the saved GLB.", """
        "progress": 0.5  // repeated, 0 to 1
        "status": "..."  // repeated
        "model": "View/local/pixal3d/2026-01-02/0304-42.glb"  // last
        """)]
    public static async Task<JObject> Pixal3DGenerateWS(WebSocket socket, Session session, JObject raw)
    {
        string image = $"{raw["image"]}";
        if (image.StartsWith("data:"))
        {
            image = image.After(",");
        }
        if (string.IsNullOrWhiteSpace(image))
        {
            return new JObject() { ["error"] = "No image given." };
        }
        JObject inputs = new()
        {
            ["image"] = new JArray("1", 0),
            ["mask"] = new JArray("1", 1),
            ["filename_prefix"] = "3d/pixal3d"
        };
        foreach (string name in FileInputs)
        {
            inputs[name] = $"{raw[name]}";
        }
        foreach (string name in IntInputs)
        {
            if (raw.TryGetValue(name, out JToken v))
            {
                inputs[name] = v.Value<long>();
            }
        }
        foreach (string name in new[] { "fov", "cfg" })
        {
            if (raw.TryGetValue(name, out JToken v))
            {
                inputs[name] = v.Value<double>();
            }
        }
        if (raw.TryGetValue("texture", out JToken tex))
        {
            inputs["texture"] = tex.Value<bool>();
        }
        long seed = inputs.Value<long?>("seed") ?? -1;
        if (seed < 0)
        {
            seed = Random.Shared.NextInt64(0, int.MaxValue);
            inputs["seed"] = seed;
        }
        bool removeBackground = raw.TryGetValue("remove_background", out JToken rem) && rem.Value<bool>();
        string bgModel = $"{raw["bg_removal_name"]}";
        if (removeBackground && string.IsNullOrWhiteSpace(bgModel))
        {
            return new JObject() { ["error"] = "Removing the background needs a BiRefNet model: put birefnet.safetensors from Comfy-Org/BiRefNet in the background_removal model folder, then refresh the models." };
        }
        JObject workflow = new()
        {
            ["1"] = new JObject() { ["class_type"] = "SwarmLoadImageB64", ["inputs"] = new JObject() { ["image_base64"] = image } },
            ["2"] = new JObject() { ["class_type"] = NodeClass, ["inputs"] = inputs }
        };
        if (removeBackground)
        {
            // ComfyUI's own nodes: the model's foreground mask replaces the (all-opaque) image's alpha
            workflow["3"] = new JObject() { ["class_type"] = BackgroundLoaderClass, ["inputs"] = new JObject() { ["bg_removal_name"] = bgModel } };
            workflow["4"] = new JObject() { ["class_type"] = "RemoveBackground", ["inputs"] = new JObject() { ["bg_removal_model"] = new JArray("3", 0), ["image"] = new JArray("1", 0) } };
            inputs["mask"] = new JArray("4", 0);
        }
        async Task send(JObject o) => await socket.SendJson(o, API.WebsocketTimeout);

        // the tab closing, or sending {"interrupt": true}, cancels
        using CancellationTokenSource cancel = CancellationTokenSource.CreateLinkedTokenSource(Program.GlobalProgramCancel);
        _ = Utilities.RunCheckedTask(async () =>
        {
            try
            {
                while (!cancel.IsCancellationRequested && socket.State == WebSocketState.Open)
                {
                    byte[] msg = await socket.ReceiveData(1024 * 1024, cancel.Token);
                    if (msg is null || socket.State != WebSocketState.Open || Encoding.UTF8.GetString(msg).Contains("interrupt"))
                    {
                        cancel.Cancel();
                    }
                }
            }
            catch (Exception)
            {
                cancel.Cancel();
            }
        });

        await send(new JObject() { ["status"] = "Waiting for a backend..." });
        using T2IBackendAccess access = await Program.Backends.GetNextT2IBackend(TimeSpan.FromMinutes(30), null, null, b => b.Backend is ComfyUIAPIAbstractBackend, session, null, cancel.Token);
        ComfyUIAPIAbstractBackend backend = access.Backend as ComfyUIAPIAbstractBackend;
        if (await NodeInfo(backend) is null)
        {
            return new JObject() { ["error"] = $"The backend at {backend.APIAddress} has no {NodeClass} node." };
        }
        if (removeBackground && await NodeInfo(backend, BackgroundLoaderClass) is null)
        {
            return new JObject() { ["error"] = $"The backend at {backend.APIAddress} has no background removal nodes ({BackgroundLoaderClass}). Update it, or give the tab an image with a transparent background." };
        }
        string clientId = Guid.NewGuid().ToString();
        using ClientWebSocket comfy = await NetworkBackendUtils.ConnectWebsocket(backend.APIAddress, $"ws?clientId={clientId}");
        JObject posted = await Http.PostJSONString($"{backend.APIAddress}/prompt", new JObject() { ["prompt"] = workflow, ["client_id"] = clientId }.ToString(), cancel.Token);
        if (posted.ContainsKey("error"))
        {
            return new JObject() { ["error"] = $"The backend refused the workflow: {posted["error"]}\n{posted["node_errors"]}" };
        }
        string promptId = $"{posted["prompt_id"]}";
        await send(new JObject() { ["status"] = removeBackground ? "Removing the background and generating..." : "Generating...", ["progress"] = 0 });
        string failure = null;
        try
        {
            while (true)
            {
                byte[] data = await comfy.ReceiveData(Utilities.ExtraLargeMaxReceive, cancel.Token);
                if (data is null || comfy.CloseStatus.HasValue)
                {
                    failure = "The backend closed its connection.";
                    break;
                }
                if (data.Length < 8 || Encoding.ASCII.GetString(data, 0, 8) != "{\"type\":")
                {
                    continue;
                }
                JObject msg = JObject.Parse(Encoding.UTF8.GetString(data));
                JObject d = msg["data"] as JObject;
                if (d is null || $"{d["prompt_id"]}" != promptId)
                {
                    continue;
                }
                string type = $"{msg["type"]}";
                if (type == "progress" && d.Value<double>("max") > 0)
                {
                    await send(new JObject() { ["progress"] = d.Value<double>("value") / d.Value<double>("max") });
                }
                else if (type == "execution_error")
                {
                    failure = $"{d["exception_message"]}".Trim();
                    break;
                }
                else if (type == "execution_interrupted")
                {
                    failure = "Interrupted.";
                    break;
                }
                else if (type == "executing" && d["node"].Type == JTokenType.Null)
                {
                    break;
                }
            }
        }
        catch (OperationCanceledException)
        {
            await Http.PostAsync($"{backend.APIAddress}/interrupt", new StringContent(new JObject() { ["prompt_id"] = promptId }.ToString()));
            return new JObject() { ["error"] = "Interrupted." };
        }
        if (failure is not null)
        {
            return new JObject() { ["error"] = failure };
        }
        JObject history = JObject.Parse(await Http.GetStringAsync($"{backend.APIAddress}/history/{promptId}"));
        JToken saved = history[promptId]?["outputs"]?["2"]?["3d"]?[0];
        if (saved is null)
        {
            return new JObject() { ["error"] = "The backend finished without a model." };
        }
        byte[] glb = await Http.GetByteArrayAsync($"{backend.APIAddress}/view?filename={Uri.EscapeDataString($"{saved["filename"]}")}&subfolder={Uri.EscapeDataString($"{saved["subfolder"]}")}&type=output");
        await Http.PostAsync($"{backend.APIAddress}/history", new StringContent(new JObject() { ["delete"] = new JArray(promptId) }.ToString()));

        string rel = $"pixal3d/{DateTime.Now:yyyy-MM-dd}/{DateTime.Now:HHmmss}-{seed}";
        string full = Path.GetFullPath(UserImageHistoryHelper.GetRealPathFor(session.User, $"{session.User.OutputDirectory}/{rel}"));
        Directory.CreateDirectory(Path.GetDirectoryName(full));
        int n = 0;
        while (File.Exists($"{full}{(n == 0 ? "" : $"-{n}")}.glb"))
        {
            n++;
        }
        string suffix = n == 0 ? "" : $"-{n}";
        await File.WriteAllBytesAsync($"{full}{suffix}.glb", glb);
        string prefix = Program.ServerSettings.Paths.AppendUserNameToOutputPath ? $"View/{session.User.UserID}/" : "Output/";
        return new JObject() { ["model"] = $"{prefix}{rel}{suffix}.glb", ["seed"] = seed, ["bytes"] = glb.Length };
    }
}
