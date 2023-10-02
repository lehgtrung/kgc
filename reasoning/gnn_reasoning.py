
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import WordNet18RR
from torch_geometric.nn import SAGEConv, to_hetero
from reasoning_utils import load_data_encoded
from torch_geometric.data import HeteroData
import pandas as pd


def import_dataset_pyg(df: pd.DataFrame, _list_ents, _list_rels):
    _htr_data = HeteroData()
    _htr_data['entity'].x = torch.rand((len(_list_ents), 32))
    for rel in _list_rels:
        if not rel.startswith('inv'):
            df_query = df.query(f'relation == "{rel}"')
            _htr_data['entity', rel, 'entity'].edge_index = torch.tensor([df_query['head_idx'].tolist(),
                                                                         df_query['tail_idx'].tolist()])
        else:
            df_query = df.query(f'inv_relation == "{rel}"')
            _htr_data['entity', rel, 'entity'].edge_index = torch.tensor([df_query['tail_idx'].tolist(),
                                                                         df_query['head_idx'].tolist()])
    return _htr_data


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)


if __name__ == '__main__':
    df_train, list_ents, list_rels = load_data_encoded('../WN18RR/test.txt')
    htr_data = import_dataset_pyg(df_train, list_ents, list_rels).to('cuda')

    model = GNN(hidden_channels=64, out_channels=1)
    model = to_hetero(model, htr_data.metadata(), aggr='sum').to('cuda')

    with torch.no_grad():  # Initialize lazy modules.
        out = model(htr_data.x_dict, htr_data.edge_index_dict)
        print(out)

