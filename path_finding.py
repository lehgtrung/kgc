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


def extract_paths(graph:Graph, df:pd.DataFrame, dataset_name, max_len):
    patterns = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        head, relation, tail = row['head'], row['relation'], row['tail']
        # head = 3514
        # tail = 3515
        query = graph.run(f'''
                MATCH (head{{name:{head}, dt_name:"{dataset_name}"}}), 
                (tail{{name:{tail}, dt_name:"{dataset_name}" }}),
                p = (head)-[*1..{max_len}]->(tail)
                RETURN DISTINCT relationships(p) AS relations;
                ''')
        paths = query.data()
        for j, path in enumerate(paths):
            path = path['relations']
            if len(path) == 1:
                if extract_record_triplets(path[0])['relation'] == relation:
                    continue
            rep_path = [relation]
            for record in path:
                rep_path.append(extract_record_triplets(record)['relation'])
            patterns.append(' '.join(rep_path))
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
    max_len = args.max_len

    patterns = extract_paths(graph, train_data, dataset, max_len)

    with open(f'{dataset}/patterns_mxl_{max_len}.txt', 'w') as f:
        for i, pat in enumerate(patterns):
            f.write(f'{pat[0]} {pat[1]}\n')


