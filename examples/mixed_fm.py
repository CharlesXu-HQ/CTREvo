"""Optional mixed-field FM extension of the embedding MLP, not an Agent default.

The Criteo dense input contains 13 standardized numeric fields followed by their
13 missing indicators. Missing indicators remain in the MLP; this example adds
no explicit missingness interactions. Category embeddings are shared with the
MLP. Three FM terms represent category-category (cc), numeric-category (nc), and
numeric-numeric (nn) relations. ``enabled_pairs`` selects which terms reach the
score; setting it to [] provides a same-source control.

All modules are constructed and called for every configuration. Only fusion
gates change, preserving shared initial parameters under the same random seed.
Retraining a gated control can still change shared embeddings and MLP weights;
its measured difference is a controlled recipe comparison, not proof that every
individual pair caused the gain.
"""

import math

import torch
from torch import nn

from model_evo_harness.models.pytorch.explicit import GroupedFM, NumericFieldEmbedding


class InteractionFusion(nn.Module):
    """Expose term selection and additive fusion as one observable module."""

    def __init__(self, enabled_pairs, scale):
        super().__init__()
        names = ('cc', 'nc', 'nn')
        if (not isinstance(enabled_pairs, list)
                or any(not isinstance(name, str) or name not in names for name in enabled_pairs)
                or len(set(enabled_pairs)) != len(enabled_pairs)):
            raise ValueError('enabled_pairs must contain distinct cc, nc or nn names')
        if type(scale) not in (int, float) or not math.isfinite(scale):
            raise ValueError('fm_scale must be finite')
        self.register_buffer('enabled', torch.tensor([float(name in enabled_pairs) for name in names]))
        self.scale = scale

    def forward(self, deep_logit, terms):
        return deep_logit + self.scale * (terms * self.enabled).sum(dim=1)


class CTRModel(nn.Module):
    def __init__(self, schema, config):
        super().__init__()
        if schema['dense_width'] != 26:
            raise ValueError('this example expects 13 numeric values followed by 13 missing indicators')
        dimension = config.get('embedding_dim', 16)
        count, buckets = schema['categorical_width'], schema['buckets']
        self.embedding = nn.Embedding(count * buckets, dimension)
        nn.init.normal_(self.embedding.weight, std=.01)
        self.register_buffer('offsets', torch.arange(count) * buckets)
        width = count * dimension + schema['dense_width']
        layers = []
        for hidden in config.get('hidden_dims', [128, 64]):
            layers.extend([nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(config.get('dropout', .1))])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)
        # Construct after the original MLP, independent of enabled_pairs.
        self.numeric_embedding = NumericFieldEmbedding(13, dimension)
        self.grouped_fm = GroupedFM(count + 13,
            {'categorical': list(range(count)), 'numeric': list(range(count, count + 13))},
            [('categorical', 'categorical'), ('numeric', 'categorical'), ('numeric', 'numeric')])
        self.fusion = InteractionFusion(config.get('enabled_pairs', ['cc', 'nc']),
                                         config.get('fm_scale', .05))

    def forward(self, dense, categorical):
        embedded = self.embedding(categorical + self.offsets)
        deep_logit = self.network(torch.cat([dense, embedded.flatten(1)], dim=1)).squeeze(-1)
        numeric = self.numeric_embedding(dense[:, :13], present=~dense[:, 13:].bool())
        terms = self.grouped_fm(torch.cat([embedded, numeric], dim=1))
        return self.fusion(deep_logit, terms)


def build_model(schema, config):
    return CTRModel(schema, config)
