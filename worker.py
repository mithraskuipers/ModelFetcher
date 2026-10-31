"""Runs one download in its own process so the web UI can cancel it and show progress."""
import inspect
import json
import os
import sys
import threading
import time


def run_civitai(spec):
    import civitai

    progress_file = spec.get("progress_file")
    total = [0]
    last = [0.0]

    def write(force=False):
        now = time.time()
        if not progress_file or (not force and now - last[0] < 0.3):
            return
        last[0] = now
        try:
            tmp = progress_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"bars": [{"desc": "civitai", "unit": "B", "n": total[0]}]}, f)
            os.replace(tmp, progress_file)
        except OSError:
            pass

    def on_bytes(n):
        total[0] += n
        write()

    try:
        civitai.download_all(spec, os.environ.get("CIVITAI_API_KEY"), on_bytes)
        write(True)
        previews = None
        if spec.get("preview_dir") and spec.get("images_file"):
            # Previews are a bonus: whatever goes wrong here, the model download still counts as done.
            try:
                with open(spec["images_file"], encoding="utf-8") as f:
                    images = json.load(f)
                previews = civitai.download_previews(images, spec["preview_dir"])
            except Exception:
                previews = {"saved": [], "failed": 0, "error": True}
        print(json.dumps({"ok": True, "previews": previews}))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e) or type(e).__name__}))
        sys.exit(1)


def main():
    spec = json.loads(sys.argv[1])
    if spec.get("source") == "civitai":
        return run_civitai(spec)
    progress_file = spec.get("progress_file")
    bars = []
    lock = threading.Lock()

    def flush():
        if not progress_file:
            return
        with lock:
            data = [
                {"desc": str(getattr(b, "desc", "")), "unit": str(getattr(b, "unit", "")), "n": float(b.n or 0)}
                for b in bars
            ]
        try:
            tmp = progress_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"bars": data}, f)
            os.replace(tmp, progress_file)
        except OSError:
            pass

    def flusher():
        while True:
            flush()
            time.sleep(0.3)

    try:
        from tqdm import tqdm as base_tqdm
        devnull = open(os.devnull, "w")

        class Hook(base_tqdm):
            def __init__(self, *args, **kwargs):
                kwargs["disable"] = False
                kwargs["file"] = devnull
                super().__init__(*args, **kwargs)
                with lock:
                    bars.append(self)

        threading.Thread(target=flusher, daemon=True).start()
    except Exception:
        Hook = None

    try:
        from huggingface_hub import snapshot_download

        kwargs = dict(
            repo_id=spec["repo_id"],
            repo_type=spec["repo_type"],
            revision=spec.get("revision"),
            local_dir=spec["local_dir"],
            allow_patterns=spec.get("allow_patterns"),
        )
        if Hook is not None and "tqdm_class" in inspect.signature(snapshot_download).parameters:
            kwargs["tqdm_class"] = Hook
        snapshot_download(**kwargs)
        flush()
        print(json.dumps({"ok": True}))
    except Exception as e:
        name = type(e).__name__
        msg = str(e).splitlines()[0] if str(e) else name
        if "Gated" in name or "401" in msg or "403" in msg:
            msg = "Access denied. Accept the license on the model page and add a token in Settings."
        print(json.dumps({"ok": False, "error": msg}))
        sys.exit(1)


if __name__ == "__main__":
    main()
