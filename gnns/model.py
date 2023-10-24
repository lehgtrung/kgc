import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, Linear, HeteroConv
from torch_geometric.data import HeteroData
import pandas as pd
from tqdm import tqdm
import numpy as np


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, adj_tensor,
                 max_rule_length, max_rank):
        super().__init__()

        self.adj_tensor = adj_tensor
        self.num_rels = adj_tensor.size(-1)
        self.max_rule_length = max_rule_length
        self.max_rank = max_rank

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('entity', 'also_see', 'entity'): SAGEConv((-1, -1), hidden_channels),
                ('entity', 'member_of_domain_region', 'entity'): SAGEConv((-1, -1), hidden_channels),
                ('entity', 'synset_domain_topic_of', 'entity'): SAGEConv((-1, -1), hidden_channels),
                ('entity', 'hypernym', 'entity'): SAGEConv((-1, -1), hidden_channels),
                ('entity', 'verb_group', 'entity'): SAGEConv((-1, -1), hidden_channels),
                ('entity', 'member_of_domain_usage', 'entity'): SAGEConv((-1, -1), hidden_channels),
                ('entity', 'similar_to', 'entity'): SAGEConv((-1, -1), hidden_channels),
                ('entity', 'derivationally_related_form', 'entity'): SAGEConv((-1, -1), hidden_channels),
                ('entity', 'has_part', 'entity'): SAGEConv((-1, -1), hidden_channels),
                ('entity', 'member_meronym', 'entity'): SAGEConv((-1, -1), hidden_channels),
                ('entity', 'inv_hypernym', 'entity'): SAGEConv((-1, -1), hidden_channels),
                ('entity', 'inv_instance_hypernym', 'entity'): SAGEConv((-1, -1), hidden_channels),
                ('entity', 'instance_hypernym', 'entity'): SAGEConv((-1, -1), hidden_channels),
                ('entity', 'inv_synset_domain_topic_of', 'entity'): SAGEConv((-1, -1), hidden_channels),
                ('entity', 'inv_derivationally_related_form', 'entity'): SAGEConv((-1, -1), hidden_channels),
                ('entity', 'inv_similar_to', 'entity'): SAGEConv((-1, -1), hidden_channels),
                ('entity', 'inv_member_of_domain_usage', 'entity'): SAGEConv((-1, -1), hidden_channels),
                ('entity', 'inv_member_meronym', 'entity'): SAGEConv((-1, -1), hidden_channels),
                ('entity', 'inv_also_see', 'entity'): SAGEConv((-1, -1), hidden_channels),
                ('entity', 'inv_verb_group', 'entity'): SAGEConv((-1, -1), hidden_channels),
                ('entity', 'inv_has_part', 'entity'): SAGEConv((-1, -1), hidden_channels),
                ('entity', 'inv_member_of_domain_region', 'entity'): SAGEConv((-1, -1), hidden_channels)},
                aggr='sum')
            self.convs.append(conv)

        self.fully_add = Linear(hidden_channels, self.num_rels)
        self.fully_mul = Linear(hidden_channels, self.num_rels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        x = x_dict['entity']


        # Decode a list of numbers for addition
        add_coff = (
            self.fully_add(x).relu().unsqueeze(1)
            .expand(x.size(0), x.size(0), self.num_rels)
        )

        added_adj_tensor = add_coff + self.adj_tensor


        # Decode a list of numbers for multiplication
        mul_coff = (
            self.fully_mul(x).relu().unsqueeze(1)
            .expand(x.size(0), x.size(0), self.num_rels)
        )
        mul_coff = mul_coff * added_adj_tensor
        mul_coff = torch.sum(mul_coff, dim=-1)

        return mul_coff