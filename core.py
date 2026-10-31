import fnmatch
import json
import os
import re
import shutil
import sys
import webbrowser
from getpass import getpass
from pathlib import Path
from urllib.parse import urlparse

REPO_DIR = Path(__file__).resolve().parent
SAFE_DOWNLOADS = REPO_DIR / "downloads"


def settings_path():
    base = os.environ.get("APPDATA")
    root = Path(base) if base else Path.home() / ".config"
    new = root / "modelfetcher" / "settings.json"
    old = root / "modelfetch" / "settings.json"  # name used by earlier versions
    if not new.exists() and old.exists():
        try:
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, new)
        except OSError:
            return old
    return new


CONFIG_FILE = settings_path()


def load_config():
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(config):
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")
    except OSError:
        pass


# Must be set before huggingface_hub is imported.
if load_config().get("high_performance"):
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

try:
    from huggingface_hub import HfApi, get_token, hf_hub_download, login, logout, snapshot_download
    try:
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
except ImportError:
    print("huggingface_hub is not installed. Run start.bat to set up the environment.")
    sys.exit(1)

NAME_RE = re.compile(r"^[\w.\-]+(/[\w.\-]+)?$")


def token_status():
    token = get_token()
    return f"set (ends with ...{token[-4:]})" if token else "not set"


def header(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def fmt_size(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def ask(prompt, default=None):
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip().strip('"')
    return value or (default or "")


def yes_no(prompt, default=True):
    hint = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{hint}]: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False


def menu(options, default=1):
    for i, text in enumerate(options, 1):
        print(f"  [{i}] {text}")
    while True:
        raw = input(f"Choose 1-{len(options)} [{default}]: ").strip()
        if not raw:
            return default - 1
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print("  Invalid choice.")


def browse_folder(initial):
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(
            title="Select destination folder", initialdir=str(initial), mustexist=False
        )
        root.destroy()
        return Path(path) if path else None
    except Exception as e:
        print(f"  Folder window not available ({e}).")
        return None


def is_inside(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except (ValueError, OSError):
        return False


def inside_repo_unsafe(path):
    return is_inside(path, REPO_DIR) and not is_inside(path, SAFE_DOWNLOADS)


def prepare_folder(path):
    path.mkdir(parents=True, exist_ok=True)
    if is_inside(path, SAFE_DOWNLOADS):
        SAFE_DOWNLOADS.mkdir(parents=True, exist_ok=True)
        guard = SAFE_DOWNLOADS / ".gitignore"
        if not guard.exists():
            guard.write_text("*\n", encoding="utf-8")


def choose_destination(config):
    header("Step 1 of 4: Destination folder")
    cwd = Path.cwd()
    last = config.get("last_dir")
    options = []
    if not inside_repo_unsafe(cwd):
        options.append(("cwd", f"Current folder: {cwd}"))
    if last and Path(last) != cwd and not inside_repo_unsafe(Path(last)):
        options.append(("last", f"Last used: {last}"))
    options.append(("safe", f"This tool's downloads folder (git-ignored): {SAFE_DOWNLOADS}"))
    options.append(("browse", "Browse for a folder (opens a window, can create new folders)"))
    options.append(("type", "Type or paste a path"))

    while True:
        kind = options[menu([o[1] for o in options])][0]
        if kind == "cwd":
            path = cwd
        elif kind == "last":
            path = Path(last)
        elif kind == "safe":
            path = SAFE_DOWNLOADS
        elif kind == "browse":
            print("  A folder window is opening (check the taskbar if you do not see it).")
            path = browse_folder(last or cwd)
            if path is None:
                print("  No folder selected.")
                continue
        else:
            raw = ask("Folder path")
            if not raw:
                continue
            path = Path(os.path.expandvars(raw)).expanduser()
        if inside_repo_unsafe(path):
            print("  Not allowed: that folder is inside this tool's own git folder.")
            print("  Pick a folder elsewhere, or use the git-ignored downloads folder.")
            continue
        try:
            prepare_folder(path)
        except OSError as e:
            print(f"  Cannot use that folder: {e}")
            continue
        path = path.resolve()
        config["last_dir"] = str(path)
        save_config(config)
        print(f"  Destination: {path}")
        return path


def parse_parts(parts):
    repo_type = "model"
    if parts and parts[0] in ("datasets", "spaces"):
        repo_type = parts[0][:-1]
        parts = parts[1:]
    if not parts:
        raise ValueError("Could not find a repo name in that input.")
    repo_id = "/".join(parts[:2])
    return repo_type, repo_id, parts[2:]


def parse_target(text):
    text = text.strip().strip('"').strip("'")
    tokens = text.split()
    if len(tokens) >= 2 and tokens[0] in ("hf", "huggingface-cli") and tokens[1] == "download":
        tokens = tokens[2:]
    tokens = [t for t in tokens if not t.startswith("--")]
    if not tokens:
        raise ValueError("Nothing to parse.")

    first = tokens[0].strip('"').strip("'")
    revision = None
    filename = None

    if first.lower().startswith(("http://", "https://")) or "huggingface.co" in first.lower():
        if not first.lower().startswith("http"):
            first = "https://" + first
        parts = [p for p in urlparse(first).path.split("/") if p]
        repo_type, repo_id, rest = parse_parts(parts)
        if len(rest) >= 2 and rest[0] in ("blob", "resolve", "tree"):
            revision = rest[1]
            if rest[0] != "tree" and len(rest) > 2:
                filename = "/".join(rest[2:])
    else:
        parts = [p for p in first.split("/") if p]
        repo_type, repo_id, rest = parse_parts(parts)
        if rest:
            filename = "/".join(rest)

    if not NAME_RE.match(repo_id):
        raise ValueError(f"'{repo_id}' does not look like a valid repo name (expected user/name).")
    if len(tokens) > 1 and not filename:
        filename = tokens[1].strip('"').strip("'")
    return {"repo_type": repo_type, "repo_id": repo_id, "filename": filename, "revision": revision}


def ask_model():
    header("Step 2 of 4: Model")
    print("Paste a Hugging Face URL, or type a name like  user/model-name")
    print("Add a file name after the name to get only that file.")
    print("Datasets: use  datasets/user/name")
    while True:
        raw = ask("Model or URL")
        if not raw:
            continue
        try:
            return parse_target(raw)
        except ValueError as e:
            print(f"  {e}")


def needs_auth(e):
    if isinstance(e, (GatedRepoError, RepositoryNotFoundError)):
        return True
    code = getattr(getattr(e, "response", None), "status_code", None)
    return code in (401, 403)


def prompt_login():
    print("  Create a token at https://huggingface.co/settings/tokens (read access is enough).")
    print("  Nothing shows while you paste it, that is normal.")
    token = getpass("  Token (Enter to cancel): ").strip()
    if not token:
        return False
    try:
        login(token=token, add_to_git_credential=False)
        try:
            print(f"  Logged in as {HfApi().whoami().get('name', 'unknown')}.")
        except Exception:
            print("  Logged in.")
        return True
    except Exception as e:
        print(f"  Login failed: {str(e).splitlines()[0]}")
        return False


def fetch_files(api, spec):
    for attempt in range(2):
        try:
            info = api.repo_info(
                spec["repo_id"],
                repo_type=spec["repo_type"],
                revision=spec["revision"],
                files_metadata=True,
            )
            files = [(s.rfilename, s.size or 0) for s in info.siblings]
            if not files:
                print("  That repo has no files.")
                return None
            return files
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if attempt == 0 and needs_auth(e) and not get_token():
                print("  No access. The repo may be gated, private, or misspelled.")
                print("  If it is gated, accept the license on its web page first.")
                if prompt_login():
                    continue
                return None
            print(f"  Could not read repo: {str(e).splitlines()[0] if str(e) else type(e).__name__}")
            return None
    return None


def parse_selection(raw, n):
    chosen = set()
    for part in raw.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            rng = range(int(a), int(b) + 1)
        else:
            rng = [int(part)]
        for i in rng:
            if not 1 <= i <= n:
                raise ValueError
            chosen.add(i)
    return sorted(chosen)


def pick_from(files):
    for i, (name, size) in enumerate(files, 1):
        print(f"  {i:>3}. {name}  ({fmt_size(size)})")
    print("Enter numbers like 1,3,5-7 or 'all'.")
    while True:
        raw = input("Files: ").strip().lower()
        if raw == "all":
            return [n for n, _ in files]
        try:
            idxs = parse_selection(raw, len(files))
        except ValueError:
            print("  Invalid selection.")
            continue
        if idxs:
            return [files[i - 1][0] for i in idxs]


def filter_then_pick(files):
    while True:
        text = ask("Filter text (example: Q4_K_M, or a pattern like *.safetensors)")
        if not text:
            continue
        pattern = text if any(c in text for c in "*?[") else f"*{text}*"
        matches = [f for f in files if fnmatch.fnmatch(f[0].lower(), pattern.lower())]
        if matches:
            return pick_from(matches)
        print("  No files match. Try again.")


def find_file(files, wanted):
    names = [n for n, _ in files]
    if wanted in names:
        return wanted
    hits = [n for n in names if n.lower() == wanted.lower() or n.split("/")[-1].lower() == wanted.lower()]
    return hits[0] if len(hits) == 1 else None


def choose_files(spec, files):
    if spec["filename"]:
        match = find_file(files, spec["filename"])
        if match:
            return [match]
        print(f"  File '{spec['filename']}' not found in the repo, choose from the list instead.")
    header("Step 3 of 4: What to download")
    total = sum(s for _, s in files)
    idx = menu(
        [
            f"Everything ({len(files)} files, {fmt_size(total)})",
            "Pick files from a list",
            "Filter by text (example: Q4_K_M)",
        ]
    )
    if idx == 0:
        return [n for n, _ in files]
    if idx == 1:
        return pick_from(files)
    return filter_then_pick(files)


def download(spec, names, is_all, target):
    target.mkdir(parents=True, exist_ok=True)
    common = dict(
        repo_id=spec["repo_id"],
        repo_type=spec["repo_type"],
        revision=spec["revision"],
        local_dir=str(target),
    )
    if is_all:
        snapshot_download(**common)
    else:
        for i, name in enumerate(names, 1):
            print(f"\n[{i}/{len(names)}] {name}")
            hf_hub_download(filename=name, **common)


def run_once(api, dest):
    spec = ask_model()
    files = fetch_files(api, spec)
    if files is None:
        return
    names = choose_files(spec, files)
    is_all = len(names) == len(files)
    sizes = dict(files)
    total = sum(sizes[n] for n in names)

    header("Step 4 of 4: Confirm")
    repo_name = spec["repo_id"].split("/")[-1]
    use_sub = yes_no(f"Put the files in a subfolder named '{repo_name}'?", default=is_all)
    target = dest / repo_name if use_sub else dest

    print()
    print(f"  Repo:        {spec['repo_id']} ({spec['repo_type']})")
    print(f"  Files:       {len(names)} ({fmt_size(total)})")
    if len(names) <= 5:
        for n in names:
            print(f"               {n}")
    print(f"  Destination: {target}")
    try:
        free = shutil.disk_usage(dest).free
        print(f"  Free space:  {fmt_size(free)}")
        if total > free:
            print("  WARNING: not enough free disk space for this download.")
    except OSError:
        pass
    if not yes_no("Start download?", True):
        print("  Cancelled.")
        return

    try:
        download(spec, names, is_all, target)
    except KeyboardInterrupt:
        print("\nDownload interrupted. Run again with the same choices to resume.")
        return
    except Exception as e:
        print(f"\nDownload failed: {str(e).splitlines()[0] if str(e) else type(e).__name__}")
        return

    print(f"\nDone. Files are in: {target}")
    if os.name == "nt" and yes_no("Open the folder in Explorer?", False):
        os.startfile(str(target))


def set_token():
    print()
    print("A token is optional. It is needed for gated models (Llama, Gemma, ...)")
    print("and gives higher rate limits. Create one with 'Read' access.")
    print("It is saved in your user profile by Hugging Face's own login, never in this folder.")
    if yes_no("Open the token page in your browser?", True):
        webbrowser.open("https://huggingface.co/settings/tokens")
    prompt_login()


def remove_token():
    if not get_token():
        print("  No token is set.")
        return
    if yes_no("Remove the saved token?", False):
        try:
            logout()
            print("  Token removed.")
        except Exception as e:
            print(f"  Could not remove token: {e}")


def settings_menu(config):
    while True:
        header("Settings")
        perf = bool(config.get("high_performance"))
        print(f"  Token:             {token_status()}")
        print(f"  High performance:  {'on' if perf else 'off'} (faster downloads, uses more CPU and bandwidth)")
        print(f"  Settings file:     {CONFIG_FILE}")
        print("  Settings and token live in your user profile, never in the repo folder.")
        idx = menu(
            ["Set or replace token", "Remove token", "Toggle high performance mode", "Back"],
            default=4,
        )
        if idx == 0:
            set_token()
        elif idx == 1:
            remove_token()
        elif idx == 2:
            config["high_performance"] = not perf
            save_config(config)
            print(f"  High performance is now {'on' if not perf else 'off'}. Takes effect next time you start the tool.")
        else:
            return


def main():
    print("ModelFetcher, terminal mode (press Ctrl+C at any time to quit)")
    print(f"Token: {token_status()} (optional, set it under Settings)")
    config = load_config()
    api = HfApi()
    dest = choose_destination(config)
    while True:
        header("Main menu")
        print(f"  Destination: {dest}")
        idx = menu(
            [
                "Download a model",
                "Change destination folder",
                "Settings (token, speed)",
                "Quit",
            ]
        )
        if idx == 0:
            run_once(api, dest)
        elif idx == 1:
            dest = choose_destination(config)
        elif idx == 2:
            settings_menu(config)
        else:
            break


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")
