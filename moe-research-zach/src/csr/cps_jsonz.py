"""csr-context-prior-specialization-v2 lossless JSON compression (deviation 6, disk policy): a finished evaluation JSON
is replaced by <name>.json.gz after verifying that the decompressed bytes hash to the original sha256; every
replacement is recorded in artifacts/.../compressed_files.jsonl (path, original sha256 / bytes, gz sha256 / bytes).
`load_json` reads either form, so readers are unaffected."""
import gzip
import hashlib
import json
import os
from pathlib import Path

from csr import guard
from csr.cps_lib import ART

MANIFEST = ART / "compressed_files.jsonl"
ROOT = Path(__file__).resolve().parents[2]


def load_json(path):
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    gz = path.with_name(path.name + ".gz")
    if gz.exists():
        with gzip.open(gz, "rt") as f:
            return json.load(f)
    raise FileNotFoundError(path)


def exists(path):
    path = Path(path)
    return path.exists() or path.with_name(path.name + ".gz").exists()


def compress(path, why):
    """Replace path by path.gz (lossless, verified). Only files under the V2 artifact directory."""
    path = Path(path).resolve()
    assert str(path).startswith(str(ART.resolve()) + os.sep), f"refusing to compress outside V2 ART: {path}"
    if not path.exists():
        return None
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    gz = path.with_name(path.name + ".gz")
    tmp = path.with_name(path.name + ".gz.tmp")
    with open(tmp, "wb") as f:
        with gzip.GzipFile(filename=path.name, mode="wb", compresslevel=6, fileobj=f, mtime=0) as g:
            g.write(raw)
        f.flush()
        os.fsync(f.fileno())
    with gzip.open(tmp, "rb") as g:
        assert hashlib.sha256(g.read()).hexdigest() == sha, f"gzip round-trip mismatch for {path}"
    os.replace(tmp, gz)
    os.chmod(gz, 0o400)
    gzb = gz.read_bytes()
    rec = {"time": guard.now(), "path": str(path.relative_to(ROOT)), "orig_sha256": sha, "orig_bytes": len(raw),
           "gz_path": str(gz.relative_to(ROOT)), "gz_sha256": hashlib.sha256(gzb).hexdigest(), "gz_bytes": len(gzb),
           "why": why}
    with open(MANIFEST, "a") as f:
        f.write(json.dumps(rec) + "\n")
    os.chmod(path, 0o600)
    path.unlink()
    return rec
