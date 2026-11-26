"""ccsr1 shared helpers: base hidden-state table aligned with the extraction tables, document index, batching."""
import numpy as np
import torch

from csr import guard


@torch.no_grad()
def hidden_table(model, sp, rows, dev, micro_rows=16, want_tgt=True):
    """rms0(h) (fp16) for every SCORED token, in exactly the extraction-table order (same rows, same micro-batching,
    same flattening). Returns (X (N, 384) fp16 on dev, tgt (N,) int64 on dev)."""
    from csr.jsr_bank import rms0
    from csr.orv1_lib import make_batch, to_dev
    xs, ts = [], []
    for i in range(0, len(rows), micro_rows):
        bt = to_dev(make_batch(rows[i:i + micro_rows], sp), dev)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            o = model({"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]})
        dm = bt["doc_ids"].reshape(-1) >= 0
        H = o["hidden"].reshape(-1, o["hidden"].shape[-1])[dm]
        Y = bt["targets"].reshape(-1)[dm]
        sc = Y >= 0
        xs.append(rms0(H[sc]).to(torch.float16))
        ts.append(Y[sc])
    return torch.cat(xs), torch.cat(ts)


def doc_index(doc, pos):
    """Documents as contiguous runs of the table with positions 0..n-1. Returns starts, lengths (np.int64)."""
    doc = np.asarray(doc)
    pos = np.asarray(pos).astype(np.int64)
    brk = np.nonzero(np.diff(doc) != 0)[0] + 1
    starts = np.concatenate([[0], brk]).astype(np.int64)
    lens = np.diff(np.concatenate([starts, [len(doc)]])).astype(np.int64)
    if len(np.unique(doc[starts])) != len(starts):
        raise guard.GuardError("document tokens are not contiguous in the table")
    exp = np.arange(len(doc)) - np.repeat(starts, lens)
    if not np.array_equal(exp, pos):
        raise guard.GuardError("table positions are not 0..n-1 within each document")
    return starts, lens


def batch_index(starts, lens, docs, dev):
    """Padded index matrix (B, Lmax) into the table and validity mask for the given document indices."""
    st = torch.as_tensor(starts[docs], device=dev)
    ln = torch.as_tensor(lens[docs], device=dev)
    L = int(ln.max())
    t = torch.arange(L, device=dev).view(1, -1)
    mask = t < ln.view(-1, 1)
    idx = (st.view(-1, 1) + t).clamp(max=int(starts[-1] + lens[-1] - 1))
    return idx, mask, ln
