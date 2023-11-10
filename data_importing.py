from py2neo import Graph
from py2neo import Node, Relationship
import pandas as pd
from tqdm import tqdm
import argparse
from neo4j import GraphDatabase
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


def normalize_relation(x: str):
    inv_flag = False
    if x.startswith('!'):
        inv_flag = True
        x = x.lstrip('!')
    x = x.lstrip('_')
    if inv_flag:
        return 'inv_' + x
    return x


def load_data(path):
    data = pd.read_csv(path, sep='\t', dtype=str, header=None)
    data.columns = ['head', 'relation', 'tail']
    data['head'] = data['head'].apply(lambda x: 'e' + str(x))
    data['tail'] = data['tail'].apply(lambda x: 'e' + str(x))
    data['relation'] = data['relation'].apply(lambda x: normalize_relation(x))
    data['inv_relation'] = data['relation'].apply(lambda x: normalize_relation('!' + x))
    return data


def import_records(driver, df, dataset_name):
    print('Import records ...')

    # for i, row in tqdm(df.iterrows(), total=len(df)):
    for i, row in df.iterrows():
        head, relation, inv_relation, tail = row['head'], row['relation'], \
                                             row['inv_relation'], row['tail']
        driver.execute_query(
            f'MERGE (head:Entity {{name: "{head}", dt_name: "{dataset_name}"}}) '
            f'MERGE (tail:Entity {{name: "{tail}", dt_name: "{dataset_name}"}}) '
            f'MERGE (head)-[:{relation}]->(tail) '
            f'MERGE (tail)-[:{inv_relation}]->(head)',
        )


def split_list_into_n_sublists(in_path, out_path, n_parts=6):
    with open(in_path, 'r') as f:
        data = f.readlines()
        data = [line.strip() for line in data]
    avg = len(data) // n_parts
    remainder = len(data) % n_parts
    sublists = []
    start = 0

    for i in range(n_parts):
        sublist_size = avg + (1 if i < remainder else 0)
        sublists.append(data[start:start + sublist_size])
        start += sublist_size
        print(f'Part {i} size: {len(sublists[-1])}')

    for i in range(n_parts):
        with open(out_path.format(i=i), 'w') as f:
            f.writelines([line + '\n' for line in sublists[i]])


if __name__ == '__main__':
    # sudo systemctl start neo4j.service
    # MATCH (n)
    # DETACH DELETE n
    # http://localhost:7474/browser/

    URI = "neo4j://localhost:7687"
    global_driver = GraphDatabase.driver(URI)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Name of dataset", required=True)
    args = parser.parse_args()

    if args.dataset not in ['WN18RR', 'FB15k_237']:
        raise ValueError('Wrong dataset name!!!')
    dataset = args.dataset

    train_data = load_data(f'{dataset}/train.txt')
    import_records(global_driver, train_data, dataset)
    global_driver.close()




