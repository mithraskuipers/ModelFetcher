"""Private library of downloaded models. Standard library only.

Everything lives in the library/ folder next to this file. That folder is git-ignored
(in the repo's .gitignore and by a .gitignore written inside the folder itself), so
your links, model types and preview images are never committed.

  library/library.json                    one entry per downloaded model
  library/previews/<recognizable-dir>/    that model's preview images

Paths inside library.json are relative to the library/ folder (for example
"previews/civitai-previews__moss-image-turbo-loras__model-3006599__version-123"),
so a browsing feature can do LIBRARY_DIR / entry["previews"]["dir"] / file_name.
"""
import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path

LIBRARY_DIR = Path(__file__).resolve().parent / "library"
LIBRARY_FILE = LIBRARY_DIR / "library.json"
PREVIEWS_DIR = LIBRARY_DIR / "previews"
SCHEMA = 1

_lock = threading.Lock()

# Civitai's own type names mapped to the short type keys used in the library.
_CIVITAI_TYPES = {
    "lora": "lora", "locon": "lora", "dora": "lora",
    "checkpoint": "checkpoint",
    "textualinversion": "embedding", "embedding": "embedding",
    "vae": "vae", "controlnet": "controlnet", "upscaler": "upscaler", "hypernetwork": "hypernetwork",
}


def ensure():
    """Create the library folders and the self-ignoring .gitignore."""
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    guard = LIBRARY_DIR / ".gitignore"
    if not guard.exists():
        guard.write_text("*\n", encoding="utf-8")


def norm_type(value):
    """'LoRA ' -> 'lora', 'My Type!' -> 'my-type'. Empty if nothing usable is left."""
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")[:32]


def guess_type(civitai_type):
    key = re.sub(r"[^a-z]", "", str(civitai_type or "").lower())
    return _CIVITAI_TYPES.get(key) or norm_type(civitai_type) or "other"


def preview_dir_name(slug, model_id, version_id):
    s = re.sub(r"[^A-Za-z0-9]+", "-", str(slug or "")).strip("-")[:50] or "model"
    return f"civitai-previews__{s}__model-{model_id}__version-{version_id}"


def _file_entry(f, full, model_id, version_id):
    return {
        "id": f["id"], "name": f["name"], "size": f["size"], "sha256": f.get("sha256"),
        "type": full.get("type"), "format": full.get("format"), "fp": full.get("fp"),
        "hashes": full.get("hashes") or {},
        "air": f"civitai:{model_id}@{version_id}+{f['id']}",
    }


def civitai_entry(url, meta, model_type, files, local_dir):
    """Build the library entry for a Civitai download. `meta` is the result of civitai.lookup()."""
    model_id, version_id = meta["model_id"], meta["version_id"]
    page_url = f"https://{meta['host']}/models/{model_id}?modelVersionId={version_id}"
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    by_id = {f["id"]: f for f in meta.get("files") or []}
    details = dict(meta.get("details") or {})
    details["fetched_at"] = now  # when the stats below were read; they change over time
    return {
        "id": f"civitai-{model_id}-{version_id}",
        "source": "civitai",
        "name": meta.get("repo_id"),
        "version_name": meta.get("version_name"),
        "model_type": model_type,
        "civitai_type": meta.get("repo_type"),
        "base_model": meta.get("base_model"),
        "trigger_words": meta.get("trigger_words") or [],
        "nsfw": bool(meta.get("nsfw")),
        "url": url or page_url,
        "page_url": page_url,
        "model_id": model_id,
        "version_id": version_id,
        "details": details,
        "files": [_file_entry(f, by_id.get(f["id"]) or {}, model_id, version_id) for f in files],
        "local_dir": str(local_dir),
        "previews": None,
    }


def _empty():
    return {"schema": SCHEMA, "models": []}


def load():
    try:
        data = json.loads(LIBRARY_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _empty()
    except (OSError, ValueError):
        data = None
    if isinstance(data, dict) and isinstance(data.get("models"), list):
        return data
    try:  # unreadable file: keep a copy instead of silently overwriting it
        os.replace(LIBRARY_FILE, LIBRARY_FILE.with_name("library.broken.json"))
    except OSError:
        pass
    return _empty()


def _save(data):
    tmp = LIBRARY_FILE.with_name("library.json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, LIBRARY_FILE)


def upsert(entry):
    """Add the entry, or replace the one with the same id (downloading again updates it)."""
    ensure()
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    entry["downloaded_at"] = now
    with _lock:
        data = load()
        for i, old in enumerate(data["models"]):
            if old.get("id") == entry["id"]:
                entry["first_downloaded_at"] = old.get("first_downloaded_at") or old.get("downloaded_at") or now
                data["models"][i] = entry
                break
        else:
            entry["first_downloaded_at"] = now
            data["models"].append(entry)
        _save(data)
