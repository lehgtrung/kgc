import argparse
from py2neo import Graph
import pandas as pd
from main import load_data
from collections import Counter
from tqdm import tqdm
from neo4j import GraphDatabase
import time


def query_path(driver, relation, from_entity, to_entity, dataset_name, max_len):
    paths, _, _ = driver.execute_query(f'''
                MATCH (from_entity{{name:"{from_entity}", dt_name:"{dataset_name}"}}), 
                (to_entity{{name:"{to_entity}", dt_name:"{dataset_name}" }}),
                p = (from_entity)-[*1..{max_len}]->(to_entity)
                RETURN DISTINCT relationships(p) AS relations, nodes(p) as nodes;
        ''')
    results = set()
    for j, path in enumerate(paths):
        path = path['relations']
        if len(path) == 1:
            if path[0].type == relation:
                continue
        rep_path = [relation]
        for record in path:
            rep_path.append(record.type)
        results.add(' '.join(rep_path))
    return list(results)


def extract_paths(driver, df:pd.DataFrame, dataset_name, max_len):
    patterns = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        # head -> tail direction
        head, relation, inv_relation, tail = row['head'], row['relation'], row['inv_relation'], row['tail']
        head2tail_paths = query_path(driver=driver,
                                     relation=relation,
                                     from_entity=head,
                                     to_entity=tail,
                                     dataset_name=dataset_name,
                                     max_len=max_len)
        tail2head_paths = query_path(driver=driver,
                                     relation=inv_relation,
                                     from_entity=tail,
                                     to_entity=head,
                                     dataset_name=dataset_name,
                                     max_len=max_len)
        patterns.extend(head2tail_paths)
        patterns.extend(tail2head_paths)
    return sorted(Counter(patterns).most_common(),
                  key=lambda x: (x[0].split()[0], -x[1]))


def count_relation(df:pd.DataFrame):
    rel_count = Counter(df['relation'].tolist() + df['inv_relation'].tolist())
    return rel_count


if __name__ == '__main__':
    # Note: Neo4j is more than 10 times faster then Py2neo
    URI = "neo4j://localhost:7687"
    global_driver = GraphDatabase.driver(URI)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Name of dataset", required=True)
    parser.add_argument("--max_len", help="Max length", required=True, type=int)
    args = parser.parse_args()
    if args.dataset not in ['WN18RR', 'FB15k-237']:
        raise ValueError('Wrong dataset name!!!')
    dataset = args.dataset
    train_data = load_data(f'{dataset}/train.txt')
    relation_count = count_relation(train_data)

    patterns = extract_paths(global_driver, train_data, dataset, args.max_len)

    with open(f'{dataset}/patterns_mxl_{args.max_len}.txt', 'w') as f:
        for i, pat in enumerate(patterns):
            f.write(f'{pat[0]} {pat[1]} {round(pat[1] / relation_count[pat[0].split()[0]], 3)}\n')



