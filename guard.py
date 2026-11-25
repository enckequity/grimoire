"""Point the model to Grimoire for data files: a hint when one is mentioned, a block on raw reads."""

import os
import re
from pathlib import Path

# Not .json or .txt: those are often config and logs, which the read guard must not block.
DATA_SUFFIXES = {".csv", ".tsv", ".parquet", ".pq", ".sqlite", ".sqlite3", ".db", ".jsonl", ".ndjson"}
DEFAULT_LIMIT_KB = 256
_SPLIT = re.compile(r"[\s'\"`<>()]+")
_DATA_END = re.compile(r"[\w*?\]-]\.(?:csv|tsv|parquet|pq|sqlite3?|db|jsonl|ndjson)(?:\.gz|\.zst)?$", re.IGNORECASE)
MAX_TOKEN = 1024  # a longer "word" is not a path; skipping it keeps the per-message cost linear


def data_file_mentions(user_message):
    """Data file names or paths in a message, in order, at most 3. Splits into words first, then checks
    each word's ending: one pass, so a 40k-character word costs no more than its length."""
    found = []
    for word in _SPLIT.split(user_message or ""):
        word = word.rstrip(".,;:?!")
        if word and len(word) <= MAX_TOKEN and _DATA_END.search(word):
            found.append(word)
    return list(dict.fromkeys(found))[:3]


def is_data_path(path):
    """True for a data file name, also compressed: sales.csv, events.jsonl.gz. Not log.txt.gz."""
    name = Path(path).name.lower()
    for packed in (".gz", ".zst"):
        if name.endswith(packed):
            name = name[: -len(packed)]
    return Path(name).suffix in DATA_SUFFIXES


def data_file_hint(user_message):
    """One line of context when the message names a data file; None otherwise (no cost)."""
    names = data_file_mentions(user_message)
    if not names:
        return None
    return (
        f"[Grimoire] Data file mentioned: {', '.join(names)}. Call tool_call with name \"grimoire_attach\" and "
        "arguments {\"path\": \"<the path>\"}, then answer with grimoire_query (DuckDB SQL, cite query ids). "
        "Do not read the file or parse it with code."
    )


# Folders that hold credentials, app databases and Hermes's own state. Grimoire does not open files there
# unless the user sets plugins.entries.grimoire.settings.allow_private_folders: true.
_PRIVATE = (".hermes", ".ssh", ".gnupg", ".aws", ".config", ".kube", ".docker", "Library")
_PRIVATE_OK = ("Library/Mobile Documents", "Library/CloudStorage")  # iCloud Drive, Dropbox and similar


HERMES_HOME = "the assistant's home folder"


def _under(real, folder):
    folder = os.path.realpath(str(folder))
    return real == folder or real.startswith(folder.rstrip(os.sep) + os.sep)


def _document_cache():
    """The active profile's inbound document cache (files a user sends in chat), or None."""
    try:
        from gateway.platforms.base import get_document_cache_dir
        return get_document_cache_dir()
    except Exception:
        return None


def _hermes_refuses(real_path):
    """True when Hermes' own read guard blocks the path (credential stores, device and process files, .env files
    anywhere), or it is in the Hermes home or root (every profile, its state, sessions and memory). The one opening
    is the active profile's inbound document cache, where files sent in chat arrive. Fails closed when the guard is
    there but raises; outside a Hermes runtime (the self-test) only the ~ folders below apply."""
    try:
        from agent.file_safety import get_read_block_error
        from hermes_constants import get_default_hermes_root, get_hermes_home
    except ImportError:
        return False
    try:
        if get_read_block_error(str(real_path)):
            return True
        real = os.path.realpath(real_path)
        if not any(_under(real, root) for root in (get_hermes_home(), get_default_hermes_root())):
            return False
        cache = _document_cache()
        return not (cache and _under(real, cache))
    except Exception:
        return True


def private_folder(real_path, home=None):
    """The private folder that holds real_path, or None. HERMES_HOME for the Hermes home and anything Hermes'
    read guard refuses: allow_private_folders does not open those."""
    if _hermes_refuses(real_path):
        return HERMES_HOME
    home = os.path.realpath(home or os.path.expanduser("~"))
    try:
        rel = Path(os.path.realpath(real_path)).relative_to(home)
    except ValueError:
        return None
    text = rel.as_posix().casefold()  # macOS and Windows file names ignore case: ~/.HERMES is ~/.hermes
    if any(text == ok.casefold() or text.startswith(ok.casefold() + "/") for ok in _PRIVATE_OK):
        return None
    top = rel.parts[0].casefold() if rel.parts else ""
    return next((p for p in _PRIVATE if p.casefold() == top), None)


def private_error(folder):
    if folder == HERMES_HOME:
        return ("Grimoire does not open this file: it is a protected file, or it is in the assistant's own folders, "
                "which hold its settings, history and credentials. Copy the data file to another folder first.")
    return (f"Grimoire does not open files in ~/{folder}, which holds private app data or "
            "credentials. Copy the file to another folder, or ask the user to set "
            "plugins.entries.grimoire.settings.allow_private_folders: true.")


def check(tool_name, args, cwd=None, limit_kb=DEFAULT_LIMIT_KB):
    """Return a pre_tool_call block directive, or None to let the call run."""
    if tool_name != "read_file" or not limit_kb:
        return None
    raw = (args or {}).get("path")
    if not raw:
        return None
    path = Path(os.path.expanduser(str(raw)))
    if not path.is_absolute():
        path = Path(cwd or os.getcwd()) / path
    if not is_data_path(path):
        return None
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size < limit_kb * 1024:
        return None
    return {
        "action": "block",
        "message": (
            f"Grimoire stopped this read: {path.name} is a {size // 1024} KB data file, so reading it raw "
            "wastes tokens and gets cut off. Call grimoire_attach with this path instead, then answer with "
            f"grimoire_query. Path: {path}"
        ),
    }
