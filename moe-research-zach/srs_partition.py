#!/usr/bin/env python
"""csr-sequence-routing-search-v1: deterministic 70/30 document-hash partition of the existing calib documents into
CALIB-SEARCH and CALIB-SELECTION. Reads only calib document ids and lengths (no evaluation). Never overwrites."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr.srs_run import ART, PARTITION, PARTITION_SALT, SEARCH_FRACTION, calib_partition, load_raw_split  # noqa: E402


def main():
    guard.check_root()
    d = guard.ROOT / "artifacts/data/fwe-rep-v1"
    sp = load_raw_split(d, "calib")
    se, sl = calib_partition(sp.doc_ids)
    L = sp.lengths()
    h = lambda x: hashlib.sha256(json.dumps(x).encode()).hexdigest()  # noqa: E731
    doc = {"rule": f"u = int.from_bytes(sha256('{PARTITION_SALT}|' + doc_id)[:8], 'little') / 2**64; "
                   f"CALIB-SEARCH iff u < {SEARCH_FRACTION}", "salt": PARTITION_SALT,
           "calib_docs_sha256": h(sp.doc_ids), "calib_n_docs": sp.n_docs,
           "search_doc_indices": se, "selection_doc_indices": sl,
           "search_indices_sha256": h(se), "selection_indices_sha256": h(sl),
           "search_n_docs": len(se), "selection_n_docs": len(sl),
           "search_valid_tokens": int(sum(L[i] - 1 for i in se)), "selection_valid_tokens": int(sum(L[i] - 1 for i in sl)),
           "created": guard.now()}
    guard.atomic_write_json(PARTITION, doc, mode=0o400)
    sha = guard.sha256_file(PARTITION)
    guard.ledger("SRS_V1_PARTITION", f"{PARTITION.relative_to(guard.ROOT)} sha256={sha} search={len(se)} "
                                     f"selection={len(sl)}")
    print(json.dumps({k: v for k, v in doc.items() if "indices" not in k or "sha" in k}, indent=1), sha)


if __name__ == "__main__":
    main()
