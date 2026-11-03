import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import mimetypes
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from core import (
    NAME_RE,
    REPO_DIR,
    SAFE_DOWNLOADS,
    CONFIG_FILE,
    HfApi,
    get_token,
    inside_repo_unsafe,
    load_config,
    login,
    logout,
    needs_auth,
    parse_target,
    prepare_folder,
    save_config,
)

import civitai
import library

WEB_DIR = REPO_DIR / "web"
SESSION = secrets.token_urlsafe(24)
CONFIG = load_config()
JOBS = {}
LOOKUPS = {}  # recent Civitai lookups, so a download can reuse the model details and image list
LOCK = threading.Lock()
PORT = 0
SERVER = None
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_who = {}
library.resolve(CONFIG.get("library_dir"))  # remembered library folder, else the default one

DIALOG_CODE = (
    "import sys, tkinter as tk\n"
    "from tkinter import filedialog\n"
    "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
    "p = filedialog.askdirectory(title=sys.argv[2] if len(sys.argv) > 2 else 'Select destination folder', initialdir=sys.argv[1] or None)\n"
    "print(p or '')\n"
)


class ApiError(Exception):
    pass


def free_space(path):
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return None


def whoami_cached(token):
    if not token:
        return None
    if token not in _who:
        try:
            _who[token] = HfApi().whoami().get("name")
        except Exception:
            return None
    return _who[token]


def dest_info():
    """Return (path, source). source is 'remembered' (last folder you chose) or 'default'
    (this tool's git-ignored downloads folder, used until you choose something else)."""
    last = CONFIG.get("last_dir")
    if last and Path(last).is_dir() and not inside_repo_unsafe(Path(last)):
        return last, "remembered"
    try:
        prepare_folder(SAFE_DOWNLOADS)
    except OSError:
        return None, None
    return str(SAFE_DOWNLOADS), "default"


def current_dest():
    return dest_info()[0]


def state(_data=None):
    token = get_token()
    dest, dest_source = dest_info()
    method = CONFIG.get("dest_method") or "browse"
    if dest_source == "default" or (dest and Path(dest).resolve() == SAFE_DOWNLOADS.resolve()):
        method = "safe"
    return {
        "dest": dest,
        "dest_source": dest_source,
        "dest_method": method,
        "free": free_space(dest) if dest else None,
        "recent": [p for p in CONFIG.get("recent", []) if Path(p).is_dir()],
        "safe_downloads": str(SAFE_DOWNLOADS),
        "token": bool(token),
        "token_tail": token[-4:] if token else None,
        "user": whoami_cached(token),
        "high_performance": bool(CONFIG.get("high_performance")),
        "settings_file": str(CONFIG_FILE),
        "civitai_key": bool(CONFIG.get("civitai_key")),
        "civitai_key_tail": (CONFIG.get("civitai_key") or "")[-4:],
    }


def set_dest(data):
    raw = str(data.get("path", "")).strip().strip('"')
    if not raw:
        raise ApiError("Enter a folder path.")
    path = Path(os.path.expandvars(raw)).expanduser()
    if inside_repo_unsafe(path):
        raise ApiError("That folder is inside this tool's own git folder. Pick one elsewhere or use the downloads folder.")
    try:
        prepare_folder(path)
    except OSError as e:
        raise ApiError(f"Cannot use that folder: {e}")
    path = path.resolve()
    recent = [p for p in CONFIG.get("recent", []) if p != str(path)]
    CONFIG["recent"] = ([str(path)] + recent)[:5]
    CONFIG["last_dir"] = str(path)
    method = str(data.get("method", ""))
    CONFIG["dest_method"] = method if method in ("browse", "type", "safe", "recent") else "browse"
    save_config(CONFIG)
    return state()


def browse(data):
    what = data.get("what")
    if what == "library":  # the library page: pick the folder that holds library.json
        initial = CONFIG.get("library_dir") or str(Path.home())
        title = "Select the library folder (the one that contains library.json)"
    elif what == "models":  # the library page: pick the folder the model files were moved to
        initial = CONFIG.get("last_dir") or str(Path.home())
        title = "Select the folder that now contains your models"
    else:
        initial = CONFIG.get("last_dir") or str(Path.home())
        title = "Select destination folder"
    try:
        out = subprocess.run(
            [sys.executable, "-c", DIALOG_CODE, initial, title],
            capture_output=True, text=True, timeout=900, creationflags=NO_WINDOW,
        )
    except Exception:
        raise ApiError("The folder window is not available. Use 'Type a path' instead.")
    if out.returncode != 0:
        raise ApiError("The folder window is not available. Use 'Type a path' instead.")
    lines = out.stdout.strip().splitlines()
    return {"path": lines[-1] if lines else ""}


def inspect_civitai(data):
    query = str(data.get("query", ""))
    key = CONFIG.get("civitai_key")
    try:
        result = civitai.lookup(query, key, data.get("version_id"))
        LOOKUPS[(result["host"], result["version_id"])] = result
        while len(LOOKUPS) > 20:
            LOOKUPS.pop(next(iter(LOOKUPS)))
        return result
    except civitai.AuthError as e:
        try:
            p = civitai.parse_url(query)
            url = f"https://{p['host']}/models/{p['model_id']}" if p["model_id"] else f"https://{p['host']}"
        except ValueError:
            url = "https://civitai.com"
        return {"ok": False, "kind": "auth", "source": "civitai", "has_token": bool(key), "url": url,
                "message": str(e) if not key else "Civitai refused access. Your API key may be wrong, or the model is restricted."}
    except ValueError as e:
        return {"ok": False, "kind": "fail", "message": str(e)}
    except Exception as e:
        return {"ok": False, "kind": "fail", "message": f"Civitai lookup failed: {type(e).__name__}: {e}"}


def inspect_repo(data):
    if civitai.is_civitai(str(data.get("query", ""))):
        return inspect_civitai(data)
    try:
        spec = parse_target(str(data.get("query", "")))
    except ValueError as e:
        raise ApiError(str(e))
    try:
        info = HfApi().repo_info(
            spec["repo_id"], repo_type=spec["repo_type"], revision=spec["revision"], files_metadata=True
        )
    except Exception as e:
        if needs_auth(e):
            return {
                "ok": False,
                "kind": "auth",
                "has_token": bool(get_token()),
                "url": f"https://huggingface.co/{spec['repo_id']}",
                "message": "No access to this repo. It may be gated, private, or misspelled.",
            }
        msg = str(e).splitlines()[0] if str(e) else type(e).__name__
        return {"ok": False, "kind": "fail", "message": msg}
    files = sorted(({"name": s.rfilename, "size": s.size or 0} for s in info.siblings), key=lambda f: f["name"])
    if not files:
        return {"ok": False, "kind": "fail", "message": "That repo has no files."}
    return {
        "ok": True,
        "repo_id": spec["repo_id"],
        "repo_type": spec["repo_type"],
        "revision": spec["revision"],
        "filename": spec["filename"],
        "gated": bool(getattr(info, "gated", False)),
        "private": bool(getattr(info, "private", False)),
        "files": files,
    }


def esc_pattern(name):
    return name.replace("[", "[[]").replace("*", "[*]").replace("?", "[?]")


def already_running(target, names, skip=None):
    """True if a running download writes any of these files into the same folder (two workers must not share a .part file)."""
    names = set(names)
    with LOCK:
        return any(j is not skip and j["status"] == "running" and j["target"] == str(target) and names & {n for n, _ in j["files"]}
                   for j in JOBS.values())


def launch_job(label, target, files, spec, entry=None, notice=""):
    if already_running(target, [n for n, _ in files]):
        raise ApiError("This model is already downloading to that folder.")
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise ApiError(f"Cannot create the folder {target}: {e}")
    job_id = secrets.token_hex(4)
    tmp = tempfile.mkdtemp(prefix="mfetcher_")
    spec = dict(spec, local_dir=str(target), progress_file=os.path.join(tmp, "progress.json"))
    images = spec.pop("images", None)
    if images:  # a long image list does not fit on the command line, so the worker reads it from a file
        images_file = os.path.join(tmp, "images.json")
        with open(images_file, "w", encoding="utf-8") as f:
            json.dump(images, f)
        spec["images_file"] = images_file
    job = {
        "id": job_id, "repo_id": label, "target": str(target), "files": files,
        "total": sum(s for _, s in files), "done": 0, "speed": 0.0, "status": "running",
        "error": "", "started": time.time(), "tmp": tmp, "spec": spec, "proc": None, "cancel": False,
        "entry": entry, "note": notice, "notice": notice,
    }
    with LOCK:
        JOBS[job_id] = job
    threading.Thread(target=run_job, args=(job,), daemon=True).start()
    return {"id": job_id}


def job_env(job):
    env = dict(os.environ)
    if job["spec"].get("source") == "civitai" and CONFIG.get("civitai_key"):
        env["CIVITAI_API_KEY"] = CONFIG["civitai_key"]  # via environment, not the command line
    return env


def civitai_meta(host, version_id, url):
    """Model details for a download: the cached lookup, or a fresh one if the cache lost it."""
    meta = LOOKUPS.get((host, version_id))
    if meta:
        return meta
    try:
        return civitai.lookup(url, CONFIG.get("civitai_key"), version_id)
    except Exception:
        return {}


def entry_present(m):
    """True if every file of a library entry is still in its saved folder."""
    folder = Path(m.get("local_dir") or "")
    names = [f["name"] for f in m.get("files") or [] if f.get("name")]
    return bool(names) and all((folder / n).is_file() for n in names)


def same_size(path, expected):
    try:
        return bool(expected) and abs(path.stat().st_size - expected) <= 1024
    except OSError:
        return False


def unique_target(dest, slug, model_id, version_id, names):
    """A folder of its own for a model whose file names are already taken: <name>__<model id>, then with the version id."""
    for cand in (f"{slug}__{model_id}", f"{slug}__{model_id}-{version_id}", *[f"{slug}__{model_id}-{version_id}-{n}" for n in range(2, 50)]):
        folder = Path(dest) / cand
        if not any((folder / n).exists() for n in names):
            return folder
    raise ApiError("Could not find a free folder name for this model.")


def start_civitai_download(data, target=None, force=False):
    """Start a Civitai download, or ask first.

    Returns {"confirm": {...}} instead of starting when the model is already in the library, or when the same
    files already sit in the folder. Send the request again with "resolve" set to one of the offered choices.
    Files that belong to a different model are never touched: that download goes to a folder of its own.
    """
    host = str(data.get("host", ""))
    if not civitai.host_ok(host):
        raise ApiError("Invalid Civitai address.")
    try:
        version_id = int(data.get("version_id"))
        files = []
        for f in data.get("files", []):
            sha = str(f.get("sha256") or "").lower()
            files.append({"id": int(f["id"]), "name": civitai.clean_name(f["name"]), "size": int(f.get("size") or 0),
                          "sha256": sha if re.fullmatch(r"[0-9a-f]{64}", sha) else None})
    except (TypeError, ValueError, KeyError):
        raise ApiError("Invalid file list.")
    if not files:
        raise ApiError("Select at least one file.")
    label = str(data.get("repo_id") or "Civitai model")
    url = str(data.get("url") or "").strip()[:2000]
    meta = civitai_meta(host, version_id, url)
    if meta.get("version_id") != version_id:
        raise ApiError("Could not read this model's details from Civitai. Check your connection and Civitai key, then try again.")
    model_type = library.norm_type(data.get("model_type")) or library.guess_type(meta.get("repo_type"))
    try:
        library.ensure()
    except OSError as e:
        raise ApiError(f"Cannot use the library folder: {e}")

    entry_id = f"civitai-{meta['model_id']}-{meta['version_id']}"
    models = library.load()["models"]
    existing = next((m for m in models if m.get("id") == entry_id), None)
    names = [f["name"] for f in files]
    resolve = str(data.get("resolve") or "")
    notice = []

    if target is None:  # started from the downloader page: work out where the files go and whether that is safe
        dest = current_dest()
        if not dest:
            raise ApiError("Choose a destination folder first.")
        slug = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(data.get("folder") or label))[:80] or "civitai-model"
        target = Path(dest) / slug if data.get("subfolder") else Path(dest)

        if existing and resolve != "replace":
            return {"confirm": {"reason": "in_library", "name": label, "choices": ["replace"],
                                "existing": {"name": existing.get("name"), "version_name": existing.get("version_name"), "model_type": existing.get("model_type"),
                                             "downloaded_at": existing.get("downloaded_at"), "folder": existing.get("local_dir"), "present": entry_present(existing)}}}
        if existing and entry_present(existing):
            target, force = Path(existing["local_dir"]), True  # replace the files where they are
        else:
            clash = [(n, f["size"]) for n, f in zip(names, files) if (target / n).exists()]
            if clash:
                other_owner = any(m.get("id") != entry_id and Path(m.get("local_dir") or "") == target
                                  and {f.get("name") for f in m.get("files") or []} & set(names) for m in models)
                if other_owner or not all(same_size(target / n, size) for n, size in clash):
                    # Different model, same file names: never overwrite or skip. Give this one its own folder.
                    target = unique_target(dest, slug, meta["model_id"], meta["version_id"], names)
                    notice.append(f"Another model already uses those file names, so this one was saved in its own folder ({target.name}).")
                elif not resolve:
                    return {"confirm": {"reason": "on_disk", "name": label, "folder": str(target), "files": [n for n, _ in clash], "choices": ["replace", "keep"]}}
                elif resolve == "replace":
                    force = True
        twins = [m for m in models if m.get("id") != entry_id and str(m.get("name") or "").strip().lower() == label.strip().lower()]
        if twins:
            notice.append(f"You also have a different model named \"{label}\" (model {twins[0].get('model_id')}). Both are kept, and the library shows their IDs.")

    entry = library.civitai_entry(url, meta, model_type, files, target)
    spec = {"source": "civitai", "host": host, "version_id": version_id, "files": files, "force": bool(force)}
    images = [i for i in meta.get("images") or [] if i.get("url")]
    if images:
        name = library.preview_dir_name(meta.get("folder"), meta["model_id"], meta["version_id"])
        spec["images"] = images
        spec["preview_dir"] = str(library.PREVIEWS_DIR / name)
    return launch_job(label, target, [(f["name"], f["size"]) for f in files], spec, entry, " ".join(notice))


def start_download(data):
    if data.get("source") == "civitai":
        return start_civitai_download(data)
    repo_id = str(data.get("repo_id", ""))
    repo_type = str(data.get("repo_type", "model"))
    revision = data.get("revision") or None
    if not NAME_RE.match(repo_id) or repo_type not in ("model", "dataset", "space"):
        raise ApiError("Invalid repo.")
    files = [(str(f["name"]), int(f.get("size") or 0)) for f in data.get("files", [])]
    if not files:
        raise ApiError("Select at least one file.")
    dest = current_dest()
    if not dest:
        raise ApiError("Choose a destination folder first.")
    name = "".join(c if (c.isalnum() or c in "._-") else "_" for c in repo_id.split("/")[-1])
    target = Path(dest) / name if data.get("subfolder") else Path(dest)
    return launch_job(repo_id, target, files, {
        "source": "huggingface",
        "repo_id": repo_id,
        "repo_type": repo_type,
        "revision": revision,
        "allow_patterns": None if data.get("all") else [esc_pattern(n) for n, _ in files],
    })


def hook_bytes(path):
    try:
        bars = json.loads(Path(path).read_text(encoding="utf-8")).get("bars", [])
    except Exception:
        return 0
    return int(max((b.get("n", 0) for b in bars if b.get("unit") == "B"), default=0))


def completed_bytes(job):
    """Bytes of finished files plus partial bytes of files still downloading.

    Partial files whose size equals a full expected size are ignored: Xet preallocates
    them, so their size says nothing about progress (the progress hook covers that path).
    """
    total = 0
    base = Path(job["target"])
    sizes = {size for _, size in job["files"]}
    for name, size in job["files"]:
        p = base / name
        try:
            if p.is_file() and p.stat().st_size == size:
                total += size
        except OSError:
            pass
    try:
        for p in (base / ".cache" / "huggingface" / "download").rglob("*.incomplete"):
            try:
                part = p.stat().st_size
            except OSError:
                continue
            if part not in sizes:
                total += part
    except OSError:
        pass
    return total


def record_library(job, result):
    """After a successful Civitai download: write the model (and its saved previews) into library.json."""
    entry, preview_dir = job["entry"], job["spec"].get("preview_dir")
    parts = []
    if preview_dir:
        prev = result.get("previews") or {}
        saved = [str(n) for n in prev.get("saved") or []]
        entry["previews"] = {"dir": "previews/" + Path(preview_dir).name, "count": len(saved), "files": saved}
        if prev.get("error"):
            parts.append("preview images could not be downloaded")
        else:
            parts.append(f"{len(saved)} preview image{'' if len(saved) == 1 else 's'} saved")
            if prev.get("failed"):
                parts.append(f"{prev['failed']} failed")
    try:
        library.upsert(entry)
        parts.append(f"added to library as '{entry['model_type']}'")
    except OSError as e:
        parts.append(f"could not update the library file ({e})")
    note = ", ".join(parts)
    job["note"] = (job.get("notice", "") + " " + note[:1].upper() + note[1:]).strip()


def run_job(job):
    out = []
    try:
        proc = subprocess.Popen(
            [sys.executable, str(REPO_DIR / "worker.py"), json.dumps(job["spec"])],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, creationflags=NO_WINDOW,
            env=job_env(job),
        )
    except Exception as e:
        job["status"], job["error"] = "error", str(e)
        return
    job["proc"] = proc
    threading.Thread(target=lambda: out.extend(proc.stdout), daemon=True).start()
    samples = []
    while proc.poll() is None:
        done = max(completed_bytes(job), hook_bytes(job["spec"]["progress_file"]))
        done = min(done, job["total"]) if job["total"] else done
        job["done"] = max(job["done"], done)
        now = time.time()
        samples.append((now, job["done"]))
        samples = [s for s in samples if now - s[0] <= 6]
        if len(samples) > 1 and now > samples[0][0]:
            job["speed"] = max(0.0, (samples[-1][1] - samples[0][1]) / (samples[-1][0] - samples[0][0]))
        time.sleep(0.5)
    proc.wait()
    time.sleep(0.2)
    job["speed"] = 0.0
    if job["cancel"]:
        job["status"] = "cancelled"
        return
    result = {}
    for line in reversed(out):
        try:
            result = json.loads(line)
            break
        except ValueError:
            continue
    if proc.returncode == 0 and result.get("ok"):
        job["done"] = job["total"]
        job["status"] = "done"
        if job.get("entry"):
            record_library(job, result)
    else:
        job["status"] = "error"
        job["error"] = result.get("error") or "Download failed."


def jobs_list(_data=None):
    rows = []
    with LOCK:
        for j in JOBS.values():
            eta = None
            if j["status"] == "running" and j["speed"] > 0:
                eta = int((j["total"] - j["done"]) / j["speed"])
            rows.append({
                "id": j["id"], "repo_id": j["repo_id"], "target": j["target"], "status": j["status"],
                "error": j["error"], "done": j["done"], "total": j["total"], "speed": j["speed"],
                "eta": eta, "nfiles": len(j["files"]), "started": j["started"], "note": j.get("note", ""),
            })
    return sorted(rows, key=lambda r: -r["started"])


def cancel_job(data):
    job = JOBS.get(str(data.get("id")))
    if job and job["status"] == "running" and job["proc"]:
        job["cancel"] = True
        job["proc"].terminate()
    return {}


def retry_job(data):
    """Run a failed or cancelled download again. Partial files are resumed, finished files are kept."""
    with LOCK:
        job = JOBS.get(str(data.get("id")))
    if not job:
        raise ApiError("Unknown download.")
    if job["status"] not in ("error", "cancelled"):
        raise ApiError("Only failed or cancelled downloads can be retried.")
    if already_running(job["target"], [n for n, _ in job["files"]], skip=job):
        raise ApiError("The same files are already downloading.")
    try:
        os.remove(job["spec"]["progress_file"])  # old progress would make the bar jump
    except OSError:
        pass
    job.update(status="running", error="", note="", cancel=False, speed=0.0, proc=None)
    threading.Thread(target=run_job, args=(job,), daemon=True).start()
    return {}


def library_redownload(data):
    """Download a library model again into its saved folder. With force, finished files are replaced by a fresh copy."""
    entry = next((m for m in library.load()["models"] if m.get("id") == data.get("id")), None)
    if not entry or entry.get("source") != "civitai":
        raise ApiError("Only Civitai models in the library can be downloaded again.")
    host = urlparse(entry.get("page_url") or entry.get("url") or "").netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    payload = {
        "source": "civitai", "host": host, "version_id": entry.get("version_id"), "repo_id": entry.get("name"),
        "url": entry.get("url"), "model_type": entry.get("model_type"),
        "files": [{"id": f.get("id"), "name": f.get("name"), "size": f.get("size"), "sha256": f.get("sha256")} for f in entry.get("files") or []],
    }
    if not entry.get("local_dir"):
        raise ApiError("This entry has no saved folder.")
    return start_civitai_download(payload, target=Path(entry["local_dir"]), force=bool(data.get("force")))


def remove_job(data):
    with LOCK:
        job = JOBS.get(str(data.get("id")))
        if job and job["status"] != "running":
            shutil.rmtree(job["tmp"], ignore_errors=True)
            del JOBS[job["id"]]
    return {}


def open_folder(data):
    path = str(data.get("path", ""))
    known = {j["target"] for j in JOBS.values()} | {CONFIG.get("last_dir")}
    if path not in known or not Path(path).is_dir():
        raise ApiError("Unknown folder.")
    show_in_explorer(path)
    return {}


def show_in_explorer(path):
    if os.name == "nt":
        os.startfile(path)
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])


def library_location(_data=None):
    """Where the library is being read from, and whether one was found there."""
    remembered = CONFIG.get("library_dir")
    return {
        "dir": str(library.LIBRARY_DIR),
        "found": library.has_library(library.LIBRARY_DIR),
        "default": str(library.DEFAULT_DIR),
        "remembered": remembered,
        "remembered_missing": bool(remembered) and not library.has_library(remembered),
    }


def library_list(_data=None):
    """Everything the library page needs. `present` says whether the model files are still on disk."""
    models = library.load()["models"]
    for m in models:
        m["present"] = entry_present(m)
    return {"models": models, "location": library_location()}


def library_locate(data):
    """Point the tool at an existing library folder (picked with the folder window or typed) and remember it."""
    found = library.find_in(data.get("path", ""))
    if not found:
        raise ApiError("No library found there. Pick the folder that contains library.json (usually named 'library').")
    library.use(found)
    CONFIG["library_dir"] = str(found)
    save_config(CONFIG)
    return library_list()


def scan_for_files(root, wanted, max_depth=6):
    """{folder: {file name: size}} for every file under `root` whose name is in `wanted`. One walk, hidden folders skipped."""
    found = {}
    base = len(Path(root).parts)
    for cur, dirs, files in os.walk(root):
        if len(Path(cur).parts) - base >= max_depth:
            dirs[:] = []
        else:
            dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            if name in wanted:
                try:
                    found.setdefault(cur, {})[name] = os.path.getsize(os.path.join(cur, name))
                except OSError:
                    pass
    return found


def library_relocate(data):
    """After moving model files: search a folder and re-point every library entry whose files are missing.

    An entry matches a folder when all its file names are there with the saved sizes. Matches are never guessed:
    if several folders fit, the old folder name decides, and if that does not settle it the entry is skipped.
    """
    root = Path(os.path.expandvars(str(data.get("path", "")).strip().strip('"'))).expanduser()
    if not str(data.get("path", "")).strip() or not root.is_dir():
        raise ApiError("That folder does not exist.")
    missing = [m for m in library.load()["models"] if not entry_present(m)]
    wanted = {f["name"] for m in missing for f in m.get("files") or [] if f.get("name")}
    index = scan_for_files(root, wanted) if wanted else {}
    changes, ambiguous = {}, 0
    for m in missing:
        files = [f for f in m.get("files") or [] if f.get("name")]
        if not files:
            continue
        hits = [d for d, have in index.items()
                if all(f["name"] in have and (not f.get("size") or abs(have[f["name"]] - f["size"]) <= 1024) for f in files)]
        if len(hits) > 1:
            old_name = Path(m.get("local_dir") or "").name
            hits = [d for d in hits if Path(d).name == old_name] or hits
        if len(hits) == 1:
            changes[m["id"]] = hits[0]
        elif hits:
            ambiguous += 1
    fixed = library.set_local_dirs(changes) if changes else 0
    result = library_list()
    result.update(fixed=fixed, ambiguous=ambiguous, still_missing=sum(1 for m in result["models"] if not m["present"]))
    return result


def library_open(data):
    entry = next((m for m in library.load()["models"] if m.get("id") == data.get("id")), None)
    folder = Path((entry or {}).get("local_dir") or "")
    if not entry or not folder.is_dir():
        raise ApiError("That folder no longer exists. The model may have been moved or deleted.")
    show_in_explorer(str(folder))
    return {}


def save_token(data):
    token = str(data.get("token", "")).strip()
    if not token:
        raise ApiError("Paste a token first.")
    try:
        login(token=token, add_to_git_credential=False)
    except Exception as e:
        raise ApiError(f"Login failed: {str(e).splitlines()[0]}")
    return state()


def remove_token(_data=None):
    try:
        logout()
    except Exception as e:
        raise ApiError(f"Could not remove token: {e}")
    return state()


def save_civitai_key(data):
    key = str(data.get("key", "")).strip()
    if not key:
        raise ApiError("Paste an API key first.")
    CONFIG["civitai_key"] = key
    save_config(CONFIG)
    return state()


def remove_civitai_key(_data=None):
    CONFIG.pop("civitai_key", None)
    save_config(CONFIG)
    return state()


def set_perf(data):
    on = bool(data.get("high_performance"))
    CONFIG["high_performance"] = on
    save_config(CONFIG)
    if on:
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    else:
        os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
    return state()


def quit_app(_data=None):
    for j in list(JOBS.values()):
        if j["proc"] and j["proc"].poll() is None:
            j["cancel"] = True
            j["proc"].terminate()
    threading.Thread(target=SERVER.shutdown, daemon=True).start()
    return {}


ROUTES = {
    ("GET", "state"): state, ("GET", "jobs"): jobs_list, ("GET", "library"): library_list, ("POST", "library_open"): library_open,
    ("GET", "library_location"): library_location, ("POST", "library_locate"): library_locate, ("POST", "library_relocate"): library_relocate,
    ("POST", "retry"): retry_job, ("POST", "library_redownload"): library_redownload,
    ("POST", "set_dest"): set_dest, ("POST", "browse"): browse, ("POST", "inspect"): inspect_repo,
    ("POST", "download"): start_download, ("POST", "cancel"): cancel_job, ("POST", "remove_job"): remove_job,
    ("POST", "open"): open_folder, ("POST", "token"): save_token, ("POST", "remove_token"): remove_token,
    ("POST", "perf"): set_perf, ("POST", "quit"): quit_app,
    ("POST", "civitai_key"): save_civitai_key, ("POST", "remove_civitai_key"): remove_civitai_key,
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_body(self, code, body, ctype="application/json", cache="no-store"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(data)

    def host_ok(self):
        return self.headers.get("Host", "") in (f"127.0.0.1:{PORT}", f"localhost:{PORT}")

    def handle_api(self, method, name, data):
        if self.headers.get("X-Session") != SESSION:
            return self.send_body(403, {"error": "Bad session."})
        fn = ROUTES.get((method, name))
        if not fn:
            return self.send_body(404, {"error": "Not found."})
        try:
            result = fn(data)
        except ApiError as e:
            return self.send_body(400, {"error": str(e)})
        except Exception as e:
            return self.send_body(500, {"error": f"{type(e).__name__}: {e}"})
        self.send_body(200, result)

    def do_GET(self):
        if not self.host_ok():
            return self.send_body(403, {"error": "Bad host."})
        path = urlparse(self.path).path
        if path == "/":
            html = (WEB_DIR / "index.html").read_text(encoding="utf-8").replace("__SESSION__", SESSION)
            return self.send_body(200, html.encode("utf-8"), "text/html; charset=utf-8")
        if path.rstrip("/") == "/library":
            html = (WEB_DIR / "library.html").read_text(encoding="utf-8").replace("__SESSION__", SESSION)
            return self.send_body(200, html.encode("utf-8"), "text/html; charset=utf-8")
        if path.startswith("/library/previews/"):
            return self.send_preview(path, urlparse(self.path).query)
        if path.startswith("/api/"):
            return self.handle_api("GET", path[5:], {})
        self.send_body(404, {"error": "Not found."})

    def send_preview(self, path, query):
        """Serve one saved preview image. Needs the session key, and never leaves library/previews/."""
        if parse_qs(query).get("s", [""])[0] != SESSION:
            return self.send_body(403, {"error": "Bad session."})
        root = library.PREVIEWS_DIR.resolve()
        try:
            file = (library.LIBRARY_DIR / unquote(path[len("/library/"):])).resolve()
            file.relative_to(root)
            data = file.read_bytes()
        except (ValueError, OSError):
            return self.send_body(404, {"error": "Not found."})
        ctype = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        self.send_body(200, data, ctype, cache="private, max-age=86400")

    def do_POST(self):
        if not self.host_ok():
            return self.send_body(403, {"error": "Bad host."})
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self.send_body(400, {"error": "Bad request."})
        if path.startswith("/api/"):
            return self.handle_api("POST", path[5:], data)
        self.send_body(404, {"error": "Not found."})


def main():
    global PORT, SERVER
    for port in range(8765, 8786):
        try:
            SERVER = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            PORT = port
            break
        except OSError:
            continue
    if SERVER is None:
        print("Could not find a free port between 8765 and 8785.")
        return
    url = f"http://127.0.0.1:{PORT}/"
    print("ModelFetcher is running.")
    print(f"Downloader: {url}")
    print(f"Library:    {url}library")
    print(f"Library data folder: {library.LIBRARY_DIR}" + ("" if library.has_library(library.LIBRARY_DIR) else " (no library found there yet)"))
    print("Close this window or press Ctrl+C to stop.")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.8, webbrowser.open, args=[url]).start()
    try:
        SERVER.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for j in list(JOBS.values()):
            if j["proc"] and j["proc"].poll() is None:
                j["proc"].terminate()
            shutil.rmtree(j["tmp"], ignore_errors=True)
        print("Stopped.")


if __name__ == "__main__":
    main()
