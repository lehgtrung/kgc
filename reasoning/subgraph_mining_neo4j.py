import argparse
import ast
import os
import time
from datetime import datetime

from neo4j import GraphDatabase
import re
import pandas as pd
from tqdm import tqdm
from reasoning_utils import *


def extract_subgraph_as_atoms(driver, query):
    # start_time = time.time()
    records, _, _ = driver.execute_query(query)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print("Elapsed Time:", elapsed_time, "seconds")
    triplets = []
    for j, record in enumerate(records):
        # Each record is a path from center to a node
        # center -> n1
        # center -> n1, n1 -> n2
        for path in record['relations']:
            head = path.nodes[0]['name']
            tail = path.nodes[1]['name']
            relation = path.type
            triplets.append(f'{relation}({head},{tail}).')
    return set(triplets)


def compare_with_py2neo(py2neo_path, neo4j_path, top):
    def read(path):
        with open(path, 'r') as f:
            lines = [e.strip() for e in f.readlines()]
        return lines
    for i in range(top):
        py2neo_data = read(py2neo_path.format(i=i))
        neo4j_data = read(neo4j_path.format(i=i))
        assert set(py2neo_data) == set(neo4j_data)


def extract_subgraphs(driver,
                      df: pd.DataFrame,
                      hops,
                      dataset,
                      out_path,
                      start_at):
    # query_template = """
    #     MATCH path = (center:Entity {{name: '{name}', dt_name: '{dataset}'}})-[*..2]-(k:Entity)
    #     RETURN relationships(path) as relation, nodes(path) as source_dest_nodes
    # """
    query_template = """
            MATCH (center:Entity {{name: '{name}', dt_name: '{dataset}'}}),
            path = (center)-[*..{hops}]-(k:Entity)
            RETURN relationships(path) as relations, nodes(path) as nodes
        """
    for i, row in df.iterrows():
        if i < start_at:
            continue
        if i % 100 == 0:
            print(f'At step: {i}, current time: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
        head, relation, inv_relation, tail = row['head'], row['relation'], \
                                             row['inv_relation'], row['tail']
        expected = f'{relation}({head},{tail}).'
        inv_expected = f'{inv_relation}({tail},{head}).'

        query = query_template.format(name=row['head'],
                                      hops=hops,
                                      dataset=dataset)
        atoms = extract_subgraph_as_atoms(driver, query)
        atoms.discard(expected)

        query = query_template.format(name=row['tail'],
                                      hops=hops,
                                      dataset=dataset)
        inv_atoms = extract_subgraph_as_atoms(driver, query)
        inv_atoms.discard(inv_expected)

        with open(out_path.format(idx=f'{i}.head'), 'w') as f:
            f.write('%' + expected + '\n')
            f.writelines([e + '\n' for e in atoms])
        with open(out_path.format(idx=f'{i}.tail'), 'w') as f:
            f.write('%' + inv_expected + '\n')
            f.writelines([e + '\n' for e in inv_atoms])


if __name__ == '__main__':
    # python subgraph_mining_neo4j.py --dataset WN18RR --source train --part 0 --hops 2
    URI = "neo4j://localhost:7687"
    global_driver = GraphDatabase.driver(URI)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Name of dataset", required=True)
    parser.add_argument("--source", help="Train or test", required=True)
    parser.add_argument("--part", help="Part number", type=int, required=True)
    parser.add_argument("--hops", help="Number of hops", type=int, required=True)
    parser.add_argument("--start_at", help="Start at", type=int, required=False, default=0)
    args = parser.parse_args()

    if args.dataset not in ['WN18RR', 'FB15k-237']:
        raise ValueError('Wrong dataset name!!!')
    dataset = args.dataset
    part = args.part
    hops = args.hops
    source = args.source
    start_at = args.start_at

    if source == 'test':
        out_dir = f'WN18RR_{source}_{hops}hops_neo4j'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = out_dir + '/{idx}.txt'
        df = load_data_raw(f'../{dataset}/{source}.txt')
    else:
        df = load_data_raw(f'../WN18RR/splits/part_{part}.txt')
        out_dir = f'WN18RR_train_{hops}hops_neo4j/part={part}'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = out_dir + '/{idx}.txt'
    extract_subgraphs(global_driver, df, hops, 'WN18RR', out_path, start_at)
    global_driver.close()

    # compare_with_py2neo('WN18RR_train_2hops_py2neo/part=0/{i}.train.txt',
    #                     'WN18RR_train_2hops_neo4j/part=0/{i}.train.txt',
    #                     top=100)
    #
    # compare_with_py2neo('WN18RR_train_2hops_py2neo/part=0/{i}.inv.train.txt',
    #                     'WN18RR_train_2hops_neo4j/part=0/{i}.inv.train.txt',
    #                     top=100)











