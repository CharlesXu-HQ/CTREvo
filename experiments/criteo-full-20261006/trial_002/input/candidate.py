"""Two parallel branches on one pooled field-embedding input: a retained deep
trunk and an explicit bounded-degree cross branch, read out by one joint head.

Parent recipe: trial_001 (shared field-specific embedding table + deep pooled
MLP + a parameterless pairwise scalar fused additively into the logit).
Local edit: the shared table and the trunk's hidden layers are kept; the failed
logit-level scalar and the unweighted addition are replaced by two vector cross
layers over the same pooled vector and a single learned head over [cross ; deep].

Control / ablation: num_cross_layers = 0 keeps x_L = x0, turning this file into
a wide-linear-plus-deep arm at the same head, so the cross recurrences can be
toggled without touching the embedding table, trunk, optimizer or metric.
"""

import torch
from torch import nn


class CrossLayer(nn.Module):
    """One vector cross layer: x_{l+1} = x_l + x0 * (W x_l + b)."""

    def __init__(self, dim):
        super().__init__()
        self.transform = nn.Linear(int(dim), int(dim))
        nn.init.normal_(self.transform.weight, std=0.01)
        nn.init.zeros_(self.transform.bias)

    def forward(self, x0, x):
        return x + x0 * self.transform(x)


class CTRModel(nn.Module):
    def __init__(self, schema, config):
        super().__init__()
        count = int(schema['categorical_width'])
        buckets = int(schema['buckets'])
        dim = int(config.get('embedding_dim', 16))
        drop = float(config.get('dropout', 0.1))
        self.num_cross_layers = max(0, int(config.get('num_cross_layers', 2)))
        self.field_count = count
        self.buckets = buckets
        self.dense_width = int(schema['dense_width'])
        self.embedding_dim = dim
        self.dropout_p = drop

        # retained shared field-specific table (row = field * buckets + bucket)
        self.embedding = nn.Embedding(count * buckets, dim)
        nn.init.normal_(self.embedding.weight, std=0.01)
        self.register_buffer('offsets', torch.arange(count) * buckets)

        self.input_dim = count * dim + self.dense_width

        # retained trunk hidden layers; its terminal scalar projection is gone
        layers = []
        width = self.input_dim
        for hidden in config.get('hidden_dims', [128, 64]):
            layers.extend([nn.Linear(width, int(hidden)), nn.ReLU(), nn.Dropout(drop)])
            width = int(hidden)
        self.trunk = nn.Sequential(*layers)
        self.trunk_out = width

        self.cross_layers = nn.ModuleList(
            [CrossLayer(self.input_dim) for _ in range(self.num_cross_layers)])
        self.head = nn.Linear(self.input_dim + self.trunk_out, 1)

    def embed_fields(self, categorical):
        # [B, F] int64 ids -> [B, F, D] float vectors (shared by both branches)
        return self.embedding(categorical + self.offsets)

    def pooled_input(self, dense, vectors):
        return torch.cat([dense, vectors.flatten(1)], dim=1)

    def deep_representation(self, pooled):
        return self.trunk(pooled)

    def cross_representation(self, pooled):
        x = pooled
        for layer in self.cross_layers:
            x = layer(pooled, x)
        return x

    def fuse_logits(self, deep_repr, cross_repr):
        return self.head(torch.cat([cross_repr, deep_repr], dim=1)).squeeze(-1)

    def forward(self, dense, categorical):
        vectors = self.embed_fields(categorical)
        pooled = self.pooled_input(dense, vectors)
        deep_repr = self.deep_representation(pooled)
        cross_repr = self.cross_representation(pooled)
        return self.fuse_logits(deep_repr, cross_repr)


def build_model(schema, config):
    return CTRModel(schema, config)
