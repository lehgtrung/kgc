import argparse
from py2neo import Graph
from py2neo import Node, Relationship
import pandas as pd
from main import load_dict, load_data
from collections import defaultdict, Counter
from tqdm import tqdm
import json
import re


def extract_record_triplets(record):
    record = str(record).replace(' ', '')
    pattern = r'\((.*?)\)-\[\:(.*?)\{\}\]->\((.*?)\)'
    match = re.search(pattern, record)
    return {
        'head': match.group(1),
        'relation': match.group(2),
        'tail': match.group(3)}


def query_path(graph, relation, from_entity, to_entity, dataset_name, max_len):
    query = graph.run(f'''
            MATCH (from_entity{{name:{from_entity}, dt_name:"{dataset_name}"}}), 
            (to_entity{{name:{to_entity}, dt_name:"{dataset_name}" }}),
            p = (from_entity)-[*1..{max_len}]->(to_entity)
            RETURN DISTINCT relationships(p) AS relations;
    ''')
    results = []
    paths = query.data()
    for j, path in enumerate(paths):
        path = path['relations']
        if len(path) == 1:
            if extract_record_triplets(path[0])['relation'] == relation:
                continue
        rep_path = [relation]
        for record in path:
            rep_path.append(extract_record_triplets(record)['relation'])
        results.append(' '.join(rep_path))
    return results


def inverse_relation(relation: str):
    if relation.startswith('!'):
        return relation.lstrip('!')
    return '!' + relation


def extract_paths(graph:Graph, df:pd.DataFrame, dataset_name, max_len):
    patterns = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        # head -> tail direction
        head, relation, tail = row['head'], row['relation'], row['tail']
        head2tail_paths = query_path(graph=graph,
                                     relation=relation,
                                     from_entity=head,
                                     to_entity=tail,
                                     dataset_name=dataset_name,
                                     max_len=max_len)
        tail2head_paths = query_path(graph=graph,
                                     relation=inverse_relation(relation),
                                     from_entity=tail,
                                     to_entity=head,
                                     dataset_name=dataset_name,
                                     max_len=max_len)
        patterns.extend(head2tail_paths)
        patterns.extend(tail2head_paths)
    return Counter(patterns).most_common()


if __name__ == '__main__':
    graph = Graph('bolt://localhost:7687')
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Name of dataset", required=True)
    parser.add_argument("--max_len", help="Max length", required=True)
    args = parser.parse_args()
    if args.dataset not in ['WN18RR', 'FB15k-237']:
        raise ValueError('Wrong dataset name!!!')
    dataset = args.dataset
    ent_dct = load_dict(f'{dataset}/entities.dict')
    rel_dct = load_dict(f'{dataset}/relations.dict')
    train_data = load_data(f'{dataset}/train.txt', ent_dct)

    patterns = extract_paths(graph, train_data, dataset, args.max_len)

    with open(f'{dataset}/patterns_mxl_{args.max_len}.txt', 'w') as f:
        for i, pat in enumerate(patterns):
            f.write(f'{pat[0]} {pat[1]}\n')


