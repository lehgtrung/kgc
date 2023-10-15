
from dataset import KGDataset
import shutil
import os
import glob
from natsort import natsorted
from tqdm import tqdm
import torch
from dataset_utils import load_data_encoded
from torch_geometric.data import DataLoader


def moves_parts_to_single_dir(base_path, new_base_path, num_parts):
    k = 0
    for i in tqdm(range(num_parts)):
        path = base_path + str(i)
        print(f'Part {i} contains: ', len(glob.glob(path + '/*.txt')), ' files')

        for j, file_path in enumerate(natsorted(glob.glob(path + '/*.txt'))):
            file_name = os.path.basename(file_path)

            if 'head' in file_name:
                file_name = f'{k}.fwd.txt'
            else:
                file_name = f'{k}.bwd.txt'

            new_path = os.path.join(new_base_path, file_name)
            if j % 2 == 1:
                k += 1
            shutil.copy(file_path, new_path)


def create_random_embeddings(base_path, num_ents):
    for i in tqdm(range(num_ents)):
        emb = torch.rand(100)
        torch.save(emb, os.path.join(base_path, f'{i}.pt'))


def output_list_rels_and_ents():
    df_train, list_ents, list_rels = load_data_encoded('../WN18RR/train.txt')
    with open('WN18RR_train/list_ents.txt', 'w') as f:
        f.writelines([e + '\n' for e in list_ents])
    with open('WN18RR_train/list_rels.txt', 'w') as f:
        f.writelines([e + '\n' for e in list_rels])


if __name__ == '__main__':
    # moves_parts_to_single_dir(base_path='../reasoning/WN18RR_train_2hops_neo4j/part=',
    #                           new_base_path='WN18RR_train/subgraphs',
    #                           num_parts=1)
    # df_train, list_ents, list_rels = load_data_encoded('../WN18RR/train.txt')
    # create_random_embeddings('WN18RR_train/embeddings', len(list_ents))

    # output_list_rels_and_ents()

    dataset = KGDataset(root='WN18RR_train', data_dir='WN18RR_train')
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch in loader:
        print(batch)




