"""Where the reference scripts find ComfyUI, SwarmUI and the models, and the setup ComfyUI's dynamic memory path needs.
Every script in tools/ that runs ComfyUI starts with:

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import comfy_env
    COMFY = comfy_env.use_comfy()        # SwarmUI's ComfyUI; use_comfy("ref") for ref/ComfyUI

The first line is needed because SwarmUI's python_embeded, and `python -I`, don't put the script's folder on sys.path.

Environment variables (docs/TESTING.md#environment-variables):
    SWARM_DIR    the SwarmUI folder, whose dlbackend/comfy/ComfyUI most scripts run
    COMFY_DIR    another ComfyUI checkout, in place of either; for a ComfyUI newer than SwarmUI's
    MODELS_ROOT  the models folder, in SwarmUI's layout
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _need(var, what):
    return os.environ.get(var) or sys.exit(f"set {var} to {what}")


def swarm_dir(given=None):
    """The SwarmUI folder: `given` (a script's own argument), else SWARM_DIR."""
    return given or _need("SWARM_DIR", "the SwarmUI folder")


def comfy_dir(which="swarm", swarm=None):
    """COMFY_DIR if set, else SwarmUI's ComfyUI (`which="swarm"`, in `swarm` or SWARM_DIR) or ref/ComfyUI (`"ref"`)."""
    if os.environ.get("COMFY_DIR"):
        return os.path.abspath(os.environ["COMFY_DIR"])
    if which == "ref":
        return os.path.join(REPO, "ref", "ComfyUI")
    return os.path.join(swarm_dir(swarm), "dlbackend", "comfy", "ComfyUI")


def use_comfy(which="swarm", swarm=None, chdir=False):
    """Put that ComfyUI first on sys.path, and with `chdir` make it the working folder. Returns its path."""
    path = comfy_dir(which, swarm)
    if not os.path.isdir(os.path.join(path, "comfy")):
        sys.exit(f"no ComfyUI at {path}")
    sys.path.insert(0, path)
    if chdir:
        os.chdir(path)
    return path


def swarm_nodes(*sub, swarm=None):
    """A folder of SwarmUI's own ComfyUI nodes, src/BuiltinExtensions/ComfyUIBackend/ExtraNodes[/sub...]."""
    return os.path.join(swarm_dir(swarm), "src", "BuiltinExtensions", "ComfyUIBackend", "ExtraNodes", *sub)


def models_root():
    return _need("MODELS_ROOT", "the models folder")


def model(*parts):
    """A file in the models folder, such as model("clip", "qwen3vl_4b.safetensors")."""
    return os.path.join(models_root(), *parts)


def aimdo_start():
    """Before torch is imported: start comfy_aimdo, which ComfyUI's dynamic model patcher needs. Without it and
    `aimdo_enable`, a headless script silently takes ComfyUI's legacy path, whose LoRA seed key differs."""
    import comfy_aimdo.control
    comfy_aimdo.control.init()


def aimdo_enable():
    """After comfy.cli_args, comfy.model_management and comfy.model_patcher are imported: register the devices and
    make ModelPatcherDynamic the patcher, as ComfyUI's own startup does."""
    import comfy_aimdo.control
    import comfy.cli_args
    import comfy.memory_management
    import comfy.model_management
    devices = comfy.model_management.get_all_torch_devices()
    headroom = int(comfy.cli_args.args.vram_headroom * 1024 ** 3)
    assert comfy_aimdo.control.init_devices((d.index, headroom) for d in devices)
    comfy.memory_management.aimdo_enabled = True
    # the flag is set before comfy.model_patcher is first imported, as in ComfyUI's startup
    import comfy.model_patcher
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
