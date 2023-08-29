from py2neo import Graph
from py2neo import Node, Relationship
import pandas as pd
from tqdm import tqdm
import argparse
import random


def load_dict(path, inv=False):
    dct = {}
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        for line in lines:
            v, k = line.split('\t')
            dct[k] = int(v)
    if inv:
        return {v: k for k, v in dct.items()}
    return dct


def load_data(path):
    data = pd.read_csv(path, sep='\t', dtype=str)
    data.columns = ['head', 'relation', 'tail']
    data['head'] = data['head'].apply(lambda x: 'E' + str(x))
    data['tail'] = data['tail'].apply(lambda x: 'E' + str(x))
    return data


def add_entities(graph, df, dataset_name):
    tx = graph.begin()
    # Find all entities
    head_entities = df['head'].unique().tolist()
    tail_entities = df['tail'].unique().tolist()
    all_entities = list(set(head_entities + tail_entities))

    print('Creating entities ...')
    for entity in tqdm(all_entities):
        node = Node('Entity', name=entity, dtype='train', dt_name=dataset_name)
        tx.create(node)
    graph.commit(tx)


def add_relations(graph, df, dataset_name):
    tx = graph.begin()
    print('Creating relations ...')
    for i, row in tqdm(df.iterrows(), total=len(df)):
        head, relation, tail = row['head'], row['relation'], row['tail']
        inv_relation = '!' + row['relation']
        head_entity = graph.nodes.match('Entity', name=head,
                                        dtype='train', dt_name=dataset_name).first()
        tail_entity = graph.nodes.match('Entity', name=tail,
                                        dtype='train', dt_name=dataset_name).first()

        relationship = Relationship(head_entity, relation, tail_entity)
        inv_relationship = Relationship(tail_entity, inv_relation, head_entity)
        tx.create(relationship)
        tx.create(inv_relationship)
    graph.commit(tx)


def sample_training_data(in_path, out_path, portion=0.2):
    with open(in_path, 'r') as f:
        data = f.readlines()
        data = [line.strip() for line in data]
        print('Data length: ', len(data))
        data = random.sample(data, int(len(data) * portion))
        print('Sample length: ', len(data))
    with open(out_path, 'w') as f:
        f.writelines([line + '\n' for line in data])


if __name__ == '__main__':
    # sudo systemctl start neo4j.service
    # MATCH (n)
    # DETACH DELETE n
    # http://localhost:7474/browser/
    graph = Graph('bolt://localhost:7687')
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Name of dataset", required=True)
    args = parser.parse_args()

    if args.dataset not in ['WN18RR', 'FB15k-237']:
        raise ValueError('Wrong dataset name!!!')
    dataset = args.dataset

    # sample_training_data(f'{dataset}/train.txt',
    #                      f'{dataset}/train_sampled.txt',
    #                      0.1)

    train_data = load_data(f'{dataset}/train.txt')
    add_entities(graph, train_data, dataset)
    add_relations(graph, train_data, dataset)



