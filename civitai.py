"""Civitai support: link parsing, model lookup and downloading.

Kept separate from the Hugging Face code. Uses only the standard library.

How a Civitai link maps to a download:
  page:      https://civitai.com/models/<modelId>/<slug>?modelVersionId=<versionId>
  api:       https://civitai.com/api/v1/models/<modelId>   (lists all versions and files)
  download:  https://civitai.com/api/download/models/<versionId>?fileId=<fileId>
"""
import hashlib
import json
import os
import re
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse

HOSTS = ["civitai.com", "civitai.red", "civitai.green"]
SCHEME = "https"
UA = "ModelFetcher/1.0"
CHUNK = 1024 * 1024
# Same transform the Civitai web page uses for its gallery thumbnails.
PREVIEW_PARAMS = "anim=false,width=800,optimized=true"
MEDIA_EXT = {
    "image/jpeg": ".jpeg", "image/jpg": ".jpeg", "image/png": ".png", "image/webp": ".webp",
    "image/gif": ".gif", "image/avif": ".avif", "video/mp4": ".mp4", "video/webm": ".webm",
}
URL_EXT = {".jpg": ".jpeg", ".jpeg": ".jpeg", ".png": ".png", ".webp": ".webp", ".gif": ".gif",
           ".avif": ".avif", ".mp4": ".mp4", ".webm": ".webm"}


class AuthError(Exception):
    pass


def host_ok(host):
    h = (host or "").lower().split(":")[0]
    if h.startswith("www."):
        h = h[4:]
    return any(h == x or h.endswith("." + x) for x in HOSTS)


def _base_host(netloc):
    n = netloc.lower()
    return n[4:] if n.startswith("www.") else n


def is_civitai(text):
    t = text.strip().strip("\"'")
    if t.lower().startswith("civitai:"):
        return True
    parts = t.split()
    if not parts:
        return False
    first = parts[0]
    return host_ok(urlparse(first if "://" in first else "//" + first).netloc)


def parse_url(text):
    parts = text.strip().strip("\"'").split()
    if not parts:
        raise ValueError("Nothing to parse.")
    t = parts[0]
    if t.lower().startswith("civitai:"):
        t = "https://civitai.com/models/" + t[8:].lstrip("/")
    if "://" not in t:
        t = "https://" + t
    u = urlparse(t)
    if not host_ok(u.netloc):
        raise ValueError("That is not a Civitai link.")
    path = [p for p in u.path.split("/") if p]
    q = parse_qs(u.query)

    def num(key):
        v = (q.get(key) or [""])[0]
        return int(v) if v.isdigit() else None

    out = {"host": _base_host(u.netloc), "model_id": None, "version_id": num("modelVersionId"),
           "file_id": num("fileId"), "slug": None}
    if len(path) >= 2 and path[0] == "models" and path[1].isdigit():
        out["model_id"] = int(path[1])
        out["slug"] = path[2] if len(path) > 2 else None
    elif len(path) >= 4 and path[:3] == ["api", "download", "models"] and path[3].isdigit():
        out["version_id"] = int(path[3])  # direct download link
    elif len(path) >= 2 and path[0] == "model-versions" and path[1].isdigit():
        out["version_id"] = int(path[1])
    if out["model_id"] is None and out["version_id"] is None:
        raise ValueError("Could not find a model number in that Civitai link. "
                         "Use the model page address, like https://civitai.com/models/12345")
    return out


def clean_name(name):
    name = os.path.basename(str(name).replace("\\", "/")).strip()
    name = re.sub(r'[<>:"|?*\x00-\x1f]', "_", name).strip(". ")
    return name or "file"


def _headers(key, extra=None):
    h = {"User-Agent": UA}
    if key:
        h["Authorization"] = f"Bearer {key}"
    if extra:
        h.update(extra)
    return h


def api_get(host, path, key):
    req = urllib.request.Request(f"{SCHEME}://{host}{path}", headers=_headers(key, {"Accept": "application/json"}))
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise AuthError("Civitai says you need to be logged in for this model. Add a Civitai API key in Settings.")
        if e.code == 404:
            raise ValueError("Civitai does not know that model. Check the link.")
        raise ValueError(f"Civitai answered with HTTP {e.code}.")
    except urllib.error.URLError as e:
        raise ValueError(f"Could not reach Civitai ({e.reason}).")
    except ValueError:
        raise ValueError("Civitai sent an answer that could not be read.")


_STAT_KEYS = {
    "downloadCount": "downloads", "thumbsUpCount": "thumbs_up", "thumbsDownCount": "thumbs_down",
    "ratingCount": "rating_count", "rating": "rating", "commentCount": "comments",
    "favoriteCount": "favorites", "tippedAmountCount": "buzz_tipped",
}


def _stats(d):
    """The numbers shown under 'Stats' / 'Reviews' on a Civitai page, under readable names."""
    d = d if isinstance(d, dict) else {}
    return {new: d[old] for old, new in _STAT_KEYS.items() if isinstance(d.get(old), (int, float))}


def lookup(text, key=None, version_override=None):
    spec = parse_url(text)
    host = spec["host"]
    if version_override:
        spec["version_id"] = int(version_override)
    if spec["model_id"] is None:
        spec["model_id"] = api_get(host, f"/api/v1/model-versions/{spec['version_id']}", key)["modelId"]
    m = api_get(host, f"/api/v1/models/{spec['model_id']}", key)
    versions = m.get("modelVersions") or []
    if not versions:
        raise ValueError("This model has no downloadable versions.")
    chosen = next((v for v in versions if v.get("id") == spec["version_id"]), versions[0])
    files = []
    for f in chosen.get("files") or []:
        md = f.get("metadata") or {}
        files.append({
            "id": f["id"],
            "name": clean_name(f.get("name") or f"file-{f['id']}"),
            "size": int(round((f.get("sizeKB") or 0) * 1024)),
            "type": f.get("type"),
            "primary": bool(f.get("primary")),
            "format": md.get("format"),
            "fp": md.get("fp"),
            "model_size": md.get("size"),
            "sha256": ((f.get("hashes") or {}).get("SHA256") or "").lower() or None,
            "hashes": {str(k): str(v).lower() for k, v in (f.get("hashes") or {}).items() if v},
        })
    if not files:
        raise ValueError("This version has no files.")
    images = [{"url": im["url"], "type": im.get("type") or "image", "width": im.get("width"), "height": im.get("height")}
              for im in (chosen.get("images") or []) if isinstance(im, dict) and im.get("url")]
    name = m.get("name") or f"model-{spec['model_id']}"
    details = {
        "type": m.get("type"),
        "published_at": chosen.get("publishedAt"),
        "created_at": chosen.get("createdAt"),
        "updated_at": chosen.get("updatedAt"),
        "base_model": chosen.get("baseModel"),
        "air": f"civitai:{spec['model_id']}@{chosen.get('id')}",
        "version_stats": _stats(chosen.get("stats")),
        "model_stats": _stats(m.get("stats")),
    }
    return {
        "ok": True, "source": "civitai", "host": host,
        "repo_id": name, "repo_type": m.get("type") or "Model", "revision": None, "filename": None,
        "folder": spec["slug"] or re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or f"model-{spec['model_id']}",
        "gated": False, "private": False, "nsfw": bool(m.get("nsfw")),
        "model_id": spec["model_id"], "version_id": chosen.get("id"), "version_name": chosen.get("name"),
        "base_model": chosen.get("baseModel"), "trigger_words": chosen.get("trainedWords") or [],
        "early_access": bool(chosen.get("earlyAccessEndsAt")),
        "versions": [{"id": v.get("id"), "name": v.get("name"), "base_model": v.get("baseModel")} for v in versions],
        "selected_file_id": spec["file_id"], "files": files, "images": images, "details": details,
    }


def download_url(host, version_id, file_id):
    return f"{SCHEME}://{host}/api/download/models/{int(version_id)}?fileId={int(file_id)}"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def open_follow(url, key, start=0):
    """Open a download. Redirects are followed by hand so the API key is only ever sent to Civitai."""
    opener = urllib.request.build_opener(_NoRedirect)
    for _ in range(6):
        extra = {"Range": f"bytes={start}-"} if start else None
        h = _headers(key if host_ok(urlparse(url).netloc) else None, extra)
        try:
            return opener.open(urllib.request.Request(url, headers=h), timeout=60)
        except urllib.error.HTTPError as e:
            loc = e.headers.get("Location")
            if e.code in (301, 302, 303, 307, 308) and loc:
                url = urljoin(url, loc)
                continue
            raise
    raise ValueError("Too many redirects from Civitai.")


def _need_key():
    return AuthError("Civitai wants you logged in for this file. Add a Civitai API key in Settings "
                     "(some files also need early access or an accepted license on the model page).")


def _download_one(url, final, expected, sha256, key, on_bytes):
    if final.is_file() and expected and abs(final.stat().st_size - expected) <= 1024:
        on_bytes(final.stat().st_size)  # already downloaded
        return
    part = final.with_name(final.name + ".part")
    existing = part.stat().st_size if part.exists() else 0
    try:
        try:
            resp = open_follow(url, key, existing)
        except urllib.error.HTTPError as e:
            if e.code == 416 and existing:  # stale partial file
                part.unlink()
                existing = 0
                resp = open_follow(url, key, 0)
            else:
                raise
        with resp:
            if resp.headers.get_content_type() == "text/html":
                raise _need_key()
            resume = existing > 0 and resp.status == 206
            hasher = hashlib.sha256() if sha256 else None
            if resume:
                if hasher:
                    with open(part, "rb") as f:
                        for block in iter(lambda: f.read(CHUNK), b""):
                            hasher.update(block)
                on_bytes(existing)
            with open(part, "ab" if resume else "wb") as out:
                while True:
                    block = resp.read(CHUNK)
                    if not block:
                        break
                    out.write(block)
                    if hasher:
                        hasher.update(block)
                    on_bytes(len(block))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise _need_key()
        if e.code == 404:
            raise ValueError("Civitai could not find that file. It may have been removed.")
        raise ValueError(f"Civitai answered with HTTP {e.code}.")
    except urllib.error.URLError as e:
        raise ValueError(f"Network problem: {e.reason}. Start it again to resume.")
    if hasher and hasher.hexdigest() != sha256:
        part.unlink(missing_ok=True)
        raise ValueError("The downloaded file does not match Civitai's checksum. Start it again.")
    os.replace(part, final)


def download_all(spec, key, on_bytes):
    dest = Path(spec["local_dir"])
    dest.mkdir(parents=True, exist_ok=True)
    for f in spec["files"]:
        _download_one(download_url(spec["host"], spec["version_id"], f["id"]), dest / f["name"],
                      f.get("size") or 0, f.get("sha256"), key, on_bytes)


def preview_url(url):
    """Rewrite a gallery image URL to the 800px preview version (the transform sits before the file name)."""
    u = urlparse(url)
    segs = u.path.split("/")
    if len(segs) >= 2 and "=" in segs[-2]:
        segs[-2] = PREVIEW_PARAMS
    else:
        segs.insert(len(segs) - 1, PREVIEW_PARAMS)
    return u._replace(path="/".join(segs), query="", fragment="").geturl()


def download_previews(images, dest):
    """Download every preview image into `dest`. Best effort: one bad image never stops the rest.

    Files are named 01_<original-name>.<ext>, 02_..., in the order of the Civitai gallery.
    Already downloaded images are kept, so running it again only fills the gaps.
    Returns {"saved": [file names], "failed": count}.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    have = {p.stem: p.name for p in dest.iterdir() if p.is_file() and p.suffix != ".part"}
    saved, failed = [], 0
    for i, img in enumerate(images, 1):
        url = str((img or {}).get("url") or "")
        u = urlparse(url)
        if u.scheme != "https" or not host_ok(u.netloc):
            failed += 1
            continue
        stem = f"{i:02d}_" + (clean_name(os.path.splitext(os.path.basename(u.path))[0])[:60] or "image")
        if stem in have:
            saved.append(have[stem])
            continue
        part = dest / (stem + ".part")
        try:
            with urllib.request.urlopen(urllib.request.Request(preview_url(url), headers={"User-Agent": UA}), timeout=60) as r:
                ext = MEDIA_EXT.get(r.headers.get_content_type()) or URL_EXT.get(os.path.splitext(u.path)[1].lower())
                if not ext:
                    raise ValueError("not an image")
                with open(part, "wb") as out:
                    shutil.copyfileobj(r, out, CHUNK)
            final = dest / (stem + ext)
            os.replace(part, final)
            saved.append(final.name)
        except (urllib.error.URLError, OSError, ValueError):
            part.unlink(missing_ok=True)
            failed += 1
    return {"saved": saved, "failed": failed}
