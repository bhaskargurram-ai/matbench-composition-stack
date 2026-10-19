"""MODNet expert (Amendment 2): MODNet's own relevance-redundancy feature selection (computed only on
the rows the expert is fitted on) and a MODNet feed-forward network. Inputs: our label-free feature
matrix (group F2). To bound CPU cost the architecture is fixed (no GA hyperparameter search):
n_feat, layer sizes and epochs below; recorded in the plan."""
from __future__ import annotations
import os, warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import pandas as pd


class MODNetExpert:
    def __init__(self, task_type, seed=0, n_feat=300, epochs=600, threads=4):
        self.task_type, self.seed, self.n_feat, self.epochs, self.threads = task_type, seed, n_feat, epochs, threads

    def fit(self, X, y):
        warnings.filterwarnings("ignore")
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(self.threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.keras.utils.set_random_seed(self.seed)
        from modnet.preprocessing import MODData
        from modnet.models import MODNetModel
        X = X.copy()
        X = X.fillna(X.median()).fillna(0.0)
        self.cols = list(X.columns); self.med = X.median()
        ids = [f"id{i}" for i in range(len(X))]
        Xf = pd.DataFrame(X.values, index=ids, columns=self.cols)
        y = np.asarray(y, dtype=float)
        tname = "t"
        md = MODData(materials=[None] * len(ids), targets=y.reshape(-1, 1), target_names=[tname],
                     structure_ids=ids, df_featurized=Xf)
        md.feature_selection(n=self.n_feat, n_jobs=self.threads, random_state=self.seed)
        self.md = md
        if self.task_type == "regression":
            self.model = MODNetModel([[[tname]]], weights={tname: 1}, num_neurons=[[256], [128], [16], [16]],
                                     n_feat=self.n_feat, act="elu")
        else:
            self.model = MODNetModel([[[tname]]], weights={tname: 1}, num_neurons=[[256], [128], [16], [16]],
                                     n_feat=self.n_feat, act="elu", num_classes={tname: 2})
        self.model.fit(md, epochs=self.epochs, batch_size=64, val_fraction=0.1, verbose=0)
        return self

    def _md(self, X):
        from modnet.preprocessing import MODData
        X = X.copy()[self.cols].fillna(self.med).fillna(0.0)
        ids = [f"q{i}" for i in range(len(X))]
        md = MODData(materials=[None] * len(ids), structure_ids=ids, df_featurized=pd.DataFrame(X.values, index=ids, columns=self.cols))
        md.optimal_features = self.md.optimal_features
        return md

    def predict(self, X):
        md = self._md(X)
        if self.task_type == "regression":
            return self.model.predict(md).values[:, 0]
        p = self.model.predict(md, return_prob=True)
        return p.values[:, -1] if p.shape[1] > 1 else p.values[:, 0]
