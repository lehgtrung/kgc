import torch
import os
import glob
import re
import random
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Dataset, HeteroData
from dataset_utils import *


def sample_negative(anchor_entity, pos_entity, entities, num_neg=32):
    neg_samples = []
    for i in range(num_neg):
        neg_sample = random.choice(entities)
        while neg_sample == anchor_entity or neg_sample == pos_entity:
            neg_sample = random.choice(entities)
        neg_samples.append(neg_sample)
    return neg_samples


def load_subgraph(path):
    rels = []
    heads = []
    tails = []
    with open(path, 'r') as f:
        lines = [e.strip() for e in f.readlines()]
        lines.pop(0)
        for line in lines:
            result = re.search(r'(\w+)\((\w+),(\w+)\)', line)
            rels.append(result.group(1))
            heads.append(result.group(2))
            tails.append(result.group(3))
    df = pd.DataFrame({
        'head': heads,
        'relation': rels,
        'tail': tails
    })
    list_ents = list(set(df['head'].tolist() + df['tail'].tolist()))
    df['head_idx'] = df['head'].apply(lambda x: list_ents.index(x))
    df['tail_idx'] = df['tail'].apply(lambda x: list_ents.index(x))
    return df, list_ents


def load_embeddings(base_path, indices):
    embeddings = []

    for index in indices:
        # Construct the file path
        file_path = os.path.join(base_path, f'{index}.pt')

        # Check if the file exists before attempting to load it
        if os.path.exists(file_path):
            # Load the PyTorch tensor from the file
            tensor = torch.load(file_path)

            # Append the loaded tensor to the list
            embeddings.append(tensor)
        else:
            print(f"File not found: {file_path}")

    # Convert the list of tensors to a single PyTorch tensor
    embeddings_tensor = torch.stack(embeddings)

    return embeddings_tensor


def create_htr_data_from_subgraph(df_subgraph: pd.DataFrame,
                                  global_list_rels,
                                  embeddings,
                                  relation,
                                  anchor_idx,
                                  pos_idx,
                                  neg_idx):
    _htr_data = HeteroData()
    _htr_data['entity'].x = embeddings
    subgraph_size = len(embeddings)
    # _htr_data.adj_tensor = torch.zeros(len(global_list_rels),
    #                                    subgraph_size,
    #                                    subgraph_size)
    for rel in global_list_rels:
        df_query = df_subgraph.query(f'relation == "{rel}"')
        _htr_data['entity', rel, 'entity'].edge_index = torch.tensor([df_query['head_idx'].tolist(),
                                                                      df_query['tail_idx'].tolist()])
    _htr_data.anchor_mask = torch.eye(subgraph_size)[anchor_idx]
    _htr_data.pos_mask = torch.eye(subgraph_size)[pos_idx]
    _htr_data.neg_mask = torch.eye(subgraph_size)[neg_idx]
    _htr_data.relation = relation
    return _htr_data


class KGDataset(Dataset):
    def __init__(self, root, num_neg, test=False,
                 transform=None, pre_transform=None, pre_filter=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.subgraph_path = os.path.join(root, 'subgraphs')
        self.embedding_path = os.path.join(root, 'embeddings')
        self.data_path = os.path.join(root, 'data.txt')
        self.global_list_rels_path = os.path.join(root, 'list_rels.txt')
        self.global_list_ents_path = os.path.join(root, 'list_ents.txt')
        self.global_list_rels = None
        self.global_list_ents = None
        self.test = test
        self.data = None
        self.num_neg = num_neg
        self.incremental = 0
        super(KGDataset, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return 'data.txt'

    @property
    def processed_file_names(self):
        files = []
        for i in range(100):
            files.append(f'processed/{i}.pt')
        return files

    def download(self):
        pass

    def len(self):
        return self.data.shape[0] * self.num_neg * 2  # fwd and bwd
        # return self.incremental

    def process(self):
        self.data = load_data_raw(self.data_path).head(4)
        data_length = self.data.shape[0]

        with open(self.global_list_ents_path, 'r') as f:
            self.global_list_ents = [e.strip() for e in f.readlines()]
        with open(self.global_list_rels_path, 'r') as f:
            self.global_list_rels = [e.strip() for e in f.readlines()]

        for i, row in tqdm(self.data.iterrows(), total=data_length):
            # Sample negative tail
            head, tail = row['head'], row['tail']
            rel, inv_rel = row['relation'], row['inv_relation']

            # if os.path.exists(os.path.join(self.processed_dir, f'{i}.pt')) and \
            #         os.path.exists(os.path.join(self.processed_dir, f'{i + data_length}.pt')):
            #     continue

            # Load the forward/backward subgraph
            df_fwd_subgraph, fwd_list_ents = load_subgraph(os.path.join(self.subgraph_path, f'{i}.fwd.txt'))
            df_bwd_subgraph, bwd_list_ents = load_subgraph(os.path.join(self.subgraph_path, f'{i}.bwd.txt'))

            # Load the forward/backward embeddings
            fwd_embeddings = load_embeddings(base_path=self.embedding_path,
                                             indices=[self.global_list_ents.index(e) for e in fwd_list_ents])
            bwd_embeddings = load_embeddings(base_path=self.embedding_path,
                                             indices=[self.global_list_ents.index(e) for e in bwd_list_ents])

            # Create hetero data from subgraph
            fwd_negative_indices = sample_negative(anchor_entity=fwd_list_ents.index(head),
                                                   pos_entity=fwd_list_ents.index(tail),
                                                   entities=[fwd_list_ents.index(e) for e in fwd_list_ents])
            for idx in fwd_negative_indices:
                fwd_htr_data = create_htr_data_from_subgraph(df_subgraph=df_fwd_subgraph,
                                                             global_list_rels=self.global_list_rels,
                                                             embeddings=fwd_embeddings,
                                                             relation=rel,
                                                             anchor_idx=fwd_list_ents.index(head),
                                                             pos_idx=fwd_list_ents.index(tail),
                                                             neg_idx=idx)
                torch.save(fwd_htr_data, os.path.join(self.processed_dir, f'{self.incremental}.pt'))
                self.incremental += 1

            bwd_negative_indices = sample_negative(anchor_entity=bwd_list_ents.index(tail),
                                                   pos_entity=fwd_list_ents.index(head),
                                                   entities=[bwd_list_ents.index(e) for e in bwd_list_ents])
            for idx in bwd_negative_indices:
                bwd_htr_data = create_htr_data_from_subgraph(df_subgraph=df_bwd_subgraph,
                                                             global_list_rels=self.global_list_rels,
                                                             embeddings=bwd_embeddings,
                                                             relation=inv_rel,
                                                             anchor_idx=bwd_list_ents.index(tail),
                                                             pos_idx=bwd_list_ents.index(head),
                                                             neg_idx=idx)
                torch.save(bwd_htr_data, os.path.join(self.processed_dir, f'{self.incremental}.pt'))
                self.incremental += 1

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(os.path.join(self.processed_dir, f'{idx}.pt'))
        return data




















