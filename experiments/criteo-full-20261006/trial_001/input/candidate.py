"""Deep pooled-embedding branch plus one explicit pairwise (FM) interaction branch.

Control / ablation: with config.model.use_fm = False this file reproduces the
untracked baseline wiring (same field-specific embedding table, same deep MLP
widths and dropout, same host optimizer, schedule and metric), so the only
difference between the two arms is whether the parameterless pairwise term
reaches the evaluated logit.
"""

import torch
from torch import nn


class CTRModel(nn.Module):
    def __init__(self, schema, config):
        super().__init__()
        self.field_count = int(schema['categorical_width'])
        self.buckets = int(schema['buckets'])
        self.dense_width = int(schema['dense_width'])
        self.embedding_dim = int(config.get('embedding_dim', 16))
        self.dropout_p = float(config.get('dropout', 0.1))
        self.use_fm = bool(config.get('use_fm', True))

        # one shared field-specific table: row = field_index * buckets + bucket
        self.embedding = nn.Embedding(self.field_count * self.buckets, self.embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)
        self.register_buffer('offsets', torch.arange(self.field_count) * self.buckets)

        width = self.field_count * self.embedding_dim + self.dense_width
        layers = []
        for hidden in config.get('hidden_dims', [128, 64]):
            layers.extend([nn.Linear(width, int(hidden)), nn.ReLU(), nn.Dropout(self.dropout_p)])
            width = int(hidden)
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def embed_fields(self, categorical):
        # [B, F] int64 ids -> [B, F, D] float vectors (shared by both branches)
        return self.embedding(categorical + self.offsets)

    def deep_logit(self, dense, vectors):
        # retained baseline deep path
        flat = vectors.flatten(1)
        return self.network(torch.cat([dense, flat], dim=1)).squeeze(-1)

    def fm_logit(self, vectors):
        # sum_{i<j} <e_i, e_j> via 0.5 * ((sum_f e_f)^2 - sum_f e_f^2)
        total = vectors.sum(dim=1)
        pairwise = 0.5 * (total.pow(2) - vectors.pow(2).sum(dim=1))
        return pairwise.sum(dim=-1)

    def fuse_logits(self, deep_logit, fm_logit):
        # additive fusion of the two branch logits; no sigmoid
        return deep_logit + fm_logit

    def forward(self, dense, categorical):
        vectors = self.embed_fields(categorical)
        deep_logit = self.deep_logit(dense, vectors)
        if not self.use_fm:
            return deep_logit
        return self.fuse_logits(deep_logit, self.fm_logit(vectors))


def build_model(schema, config):
    return CTRModel(schema, config)
