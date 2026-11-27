"""What a golden file was recorded from, for its "source" key: the script, and the commit of each checkout or the
version of each package it ran. A checkout with uncommitted changes gets "+dirty" after its commit.

    from golden_source import source
    json.dump({"source": source(comfyui=comfy_dir), ...}, f)
    json.dump({"source": source(packages=["comfy_kitchen"]), ...}, f)

Golden files written before this was added have no "source" key: what they came from wasn't recorded.
"""
import importlib.metadata
import json
import os
import subprocess
import sys


def git_commit(path):
    """The commit checked out at `path`, "+dirty" when it has uncommitted changes, or None if it isn't a checkout."""
    def git(*args):
        r = subprocess.run(["git", "-C", path, *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    head = git("rev-parse", "HEAD")
    if head is None:
        return None
    return head + ("+dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


def package(name):
    """An installed package's version, with the commit pip recorded if it was installed from git."""
    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    out = dist.version
    url = dist.read_text("direct_url.json")
    commit = url and json.loads(url).get("vcs_info", {}).get("commit_id")
    return f"{out} ({commit})" if commit else out


def source(packages=(), **checkouts):
    """{"script": the running script, <name>: commit for each checkout path, <package>: version for each package}."""
    out = {"script": os.path.basename(sys.argv[0])}
    out.update({name: git_commit(os.path.abspath(path)) for name, path in checkouts.items()})
    out.update({name: package(name) for name in packages})
    return out
