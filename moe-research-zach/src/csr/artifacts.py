"""Pinned external artifacts (tokenizer, frozen sentence encoder) and the frozen encoder wrapper."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .guard import ROOT
from .data import TOKENIZER_REPO, TOKENIZER_REV

ART = ROOT / "artifacts"
TOKENIZER_DIR = ART / "tokenizer" / "pythia-12b"
ENCODER_REPO = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REV = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
ENCODER_LICENSE = "apache-2.0"
ENCODER_DIR = ART / "encoder" / "all-MiniLM-L6-v2"
ENCODER_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "vocab.txt", "modules.json", "1_Pooling/config.json",
                 "sentence_bert_config.json"]
ENCODER_MAX_LEN = 256  # sentence-transformers default max_seq_length for this model
ENCODER_PREPROC = "pythia-decode -> strip -> BERT uncased wordpiece, truncate 256, mean-pool over mask, L2 normalize"


def download_pinned(repo, rev, files, local_dir: Path):
    from huggingface_hub import hf_hub_download
    local_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for f in files:
        p = hf_hub_download(repo, f, revision=rev, local_dir=str(local_dir))
        out[f] = p
    links = [str(p) for p in local_dir.rglob("*") if p.is_symlink()]
    assert not links, f"symlinks created: {links}"
    return out


def load_lm_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOKENIZER_DIR / "tokenizer.json"))


class FrozenEncoder:
    """Frozen, pinned sentence encoder (BertModel, native transformers code; no trust_remote_code)."""

    def __init__(self, device="cpu", batch_size=128):
        import torch
        from transformers import BertModel, BertTokenizerFast
        cfg = json.loads((ENCODER_DIR / "config.json").read_text())
        assert cfg.get("model_type") == "bert", cfg.get("model_type")
        self.tok = BertTokenizerFast(tokenizer_file=str(ENCODER_DIR / "tokenizer.json"),
                                     vocab_file=str(ENCODER_DIR / "vocab.txt"), do_lower_case=True)
        self.model = BertModel.from_pretrained(str(ENCODER_DIR), local_files_only=True,
                                               add_pooling_layer=False).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device, self.batch_size = device, batch_size
        self.dim = cfg["hidden_size"]

    def param_checksum(self):
        import torch
        with torch.no_grad():
            return float(sum(p.double().sum().item() for p in self.model.parameters()))

    def embed(self, texts):
        import torch
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        with torch.inference_mode():
            for s in range(0, len(order), self.batch_size):
                idx = order[s:s + self.batch_size]
                enc = self.tok([texts[i] for i in idx], padding=True, truncation=True, max_length=ENCODER_MAX_LEN,
                               return_tensors="pt").to(self.device)
                h = self.model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                e = torch.nn.functional.normalize(e.float(), dim=-1)
                out[idx] = e.cpu().numpy()
        return out
