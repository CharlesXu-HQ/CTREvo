"""Native embedding MLP baseline; the Agent may edit this entire candidate."""

import torch
from torch import nn


class CTRModel(nn.Module):
    def __init__(self, schema, config):
        super().__init__()
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

    def forward(self, dense, categorical):
        embedded = self.embedding(categorical + self.offsets).flatten(1)
        return self.network(torch.cat([dense, embedded], dim=1)).squeeze(-1)


def build_model(schema, config):
    return CTRModel(schema, config)
