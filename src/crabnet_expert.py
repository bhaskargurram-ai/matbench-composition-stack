"""CPU CrabNet expert (Amendment 2).

Network: the original CrabNet architecture from the `crabnet` package (crabnet.kingcrab: mat2vec element
embeddings, fractional encodings, transformer encoder, residual output network, masked average). The
package's training wrapper is broken on Python 3.12, so the training loop here is our own:
AdamW, L1 loss on the standardised target (regression) or BCE (classification), a cosine schedule,
early stopping on an internal 10% split of the rows the expert is fitted on, and the best-validation
weights kept. Element embeddings: mat2vec table bundled with the CBFV package.
Exposes fit(X_df, y) / predict(X_df) on a DataFrame that has a 'composition' column (string formulas).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import torch
from torch import nn

import CBFV
import crabnet.kingcrab as kc

_M2V = pd.read_csv(os.path.join(os.path.dirname(CBFV.__file__), "element_properties", "mat2vec.csv"), index_col=0)
_SYM = {s: i + 1 for i, s in enumerate(_M2V.index)}  # 0 = padding


class _Embedder(nn.Module):
    def __init__(self, d_model, compute_device=None):
        super().__init__()
        cbfv = _M2V.values
        self.fc_mat2vec = nn.Linear(cbfv.shape[1], d_model)
        cat = np.concatenate([np.zeros((1, cbfv.shape[1])), cbfv])
        self.cbfv = nn.Embedding.from_pretrained(torch.as_tensor(cat, dtype=torch.float32))

    def forward(self, src):
        return self.fc_mat2vec(self.cbfv(src))


kc.Embedder = _Embedder  # the encoder looks the class up at construction time


def encode(formulas, max_el=None):
    from pymatgen.core import Composition
    rows = []
    for f in formulas:
        c = Composition(f).fractional_composition.get_el_amt_dict()
        items = sorted(c.items(), key=lambda kv: -kv[1])
        rows.append(items)
    n = max_el or max(len(r) for r in rows)
    src = np.zeros((len(rows), n), dtype=np.int64)
    frac = np.zeros((len(rows), n), dtype=np.float32)
    for i, r in enumerate(rows):
        for j, (el, amt) in enumerate(r[:n]):
            src[i, j] = _SYM.get(el, 0)
            frac[i, j] = amt
    return torch.as_tensor(src), torch.as_tensor(frac)


class CrabNetExpert:
    def __init__(self, task_type, seed=0, d_model=256, N=3, heads=4, epochs=300, batch=128, lr=5e-4,
                 patience=40, threads=4):
        self.task_type, self.seed = task_type, seed
        self.kw = dict(d_model=d_model, N=N, heads=heads)
        self.epochs, self.batch, self.lr, self.patience, self.threads = epochs, batch, lr, patience, threads

    def _net(self):
        torch.manual_seed(self.seed)
        return kc.CrabNet(out_dims=2, **self.kw)

    def fit(self, X, y):
        torch.set_num_threads(self.threads)
        rng = np.random.default_rng(self.seed)
        src, frac = encode(X["composition"].tolist())
        self.max_el = src.shape[1]
        y = np.asarray(y, dtype=np.float32)
        if self.task_type == "regression":
            self.mu, self.sd = float(y.mean()), float(y.std() + 1e-8)
            yt = (y - self.mu) / self.sd
            loss_fn = nn.L1Loss()
        else:
            yt = y
            loss_fn = nn.BCEWithLogitsLoss()
        idx = rng.permutation(len(y))
        nv = max(1, int(0.1 * len(y)))
        va, tr = idx[:nv], idx[nv:]
        yt = torch.as_tensor(yt)
        net = self._net()
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        best, best_state, wait = np.inf, None, 0
        g = torch.Generator().manual_seed(self.seed)
        for ep in range(self.epochs):
            net.train()
            perm = torch.as_tensor(tr)[torch.randperm(len(tr), generator=g)]
            for b in range(0, len(perm), self.batch):
                ib = perm[b:b + self.batch]
                if len(ib) < 2:
                    continue
                out = net(src[ib], frac[ib])[:, 0]
                loss = loss_fn(out, yt[ib])
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            sched.step()
            net.eval()
            with torch.no_grad():
                vl = float(loss_fn(net(src[va], frac[va])[:, 0], yt[va]))
            if vl < best - 1e-6:
                best, wait = vl, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        net.load_state_dict(best_state)
        self.net = net.eval()
        return self

    def predict(self, X):
        torch.set_num_threads(self.threads)
        src, frac = encode(X["composition"].tolist(), max_el=None)
        out = []
        with torch.no_grad():
            for b in range(0, len(src), 512):
                out.append(self.net(src[b:b + 512], frac[b:b + 512])[:, 0].numpy())
        o = np.concatenate(out)
        if self.task_type == "regression":
            return o * self.sd + self.mu
        return 1.0 / (1.0 + np.exp(-o))
