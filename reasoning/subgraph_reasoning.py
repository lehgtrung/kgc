import argparse
import ast
import os

from py2neo import Graph
from py2neo import Node, Relationship
import re
import pandas as pd
from tqdm import tqdm
from reasoning_utils import *


def extract_record_triplets(record):
    record = str(record).replace(' ', '')
    pattern = r'\((.*?)\)-\[\:(.*?)\{\}\]->\((.*?)\)'
    match = re.search(pattern, record)
    return {
        'head': match.group(1),
        'relation': match.group(2),
        'tail': match.group(3)}


def triplet2atom(triplet, normalize=True):
    head, relation, tail = triplet['head'], triplet['relation'], triplet['tail']
    if normalize:
        relation = normalize_relation(relation)
    return f'{relation}({head},{tail}).'


def extract_subgraph_as_atoms(query):
    paths = graph.run(query).data()
    triplets = []
    for j, path in enumerate(paths):
        path = path['relations']
        for record in path:
            triplets.append(triplet2atom(extract_record_triplets(record)))
    return set(triplets)


def test_subgraph_extraction(graph):
    graph.run("""MATCH (n:Node) DETACH DELETE n""")
    tx = graph.begin()
    pairs = [(2, 3), (3, 4), (1, 2), (7, 1), (6, 5), (5, 2), (8, 4), (9, 8)]
    nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Create a test graph
    for node in nodes:
        node = Node('Node', name=f'Node{node}')
        tx.create(node)
    graph.commit(tx)
    tx = graph.begin()
    for pair in pairs:
        head = graph.nodes.match('Node', name=f'Node{pair[0]}').first()
        tail = graph.nodes.match('Node', name=f'Node{pair[1]}').first()
        relationship = Relationship(head, 'linked_to', tail)
        tx.create(relationship)
    graph.commit(tx)
    query = """
        MATCH (center:Node {name: 'Node2'}), 
        p = (center)-[*..3]-(subgraph)
        RETURN DISTINCT relationships(p) AS relations;
    """
    expected_result = {'linked_to(Node2,Node3).',
                       'linked_to(Node3,Node4).',
                       'linked_to(Node8,Node4).',
                       'linked_to(Node1,Node2).',
                       'linked_to(Node7,Node1).',
                       'linked_to(Node5,Node2).',
                       'linked_to(Node6,Node5).'}
    assert expected_result == extract_subgraph_as_atoms(query)


def extract_subgraphs(df: pd.DataFrame,
                      mode,
                      dataset,
                      out_path):
    """For each triplet, extract subgraph around the head/tail"""
    # Note that usually between each (head,tail) pair, there are 4 relations
    # MATCH (center:Entity {name: 'E01332730'})-[*..1]-(subgraph)
    # RETURN DISTINCT subgraph
    query_template = """
        MATCH (center:Entity {{name: '{name}', dt_name: '{dataset}'}}), 
        p = (center)-[*..2]-(subgraph)
        RETURN DISTINCT relationships(p) AS relations;
    """
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        head, relation, inv_relation, tail = row['head'], row['relation'], \
                                             row['inv_relation'], row['tail']
        expected = f'{relation}({head},{tail}).'
        inv_expected = f'{inv_relation}({tail},{head}).'

        query = query_template.format(name=row['head'], dataset=dataset)
        atoms = extract_subgraph_as_atoms(query)
        atoms.discard(expected)

        query = query_template.format(name=row['tail'], dataset=dataset)
        inv_atoms = extract_subgraph_as_atoms(query)
        inv_atoms.discard(inv_expected)

        with open(out_path.format(idx=f'{i}.{mode}'), 'w') as f:
            f.write('%' + expected + '\n')
            f.writelines([e + '\n' for e in atoms])
        with open(out_path.format(idx=f'{i}.inv.{mode}'), 'w') as f:
            f.write('%' + inv_expected + '\n')
            f.writelines([e + '\n' for e in inv_atoms])


if __name__ == '__main__':
    graph = Graph('bolt://localhost:7687')
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Name of dataset", required=True)
    parser.add_argument("--part", help="Part number", type=int, required=True)
    args = parser.parse_args()

    if args.dataset not in ['WN18RR', 'FB15k-237']:
        raise ValueError('Wrong dataset name!!!')
    dataset = args.dataset
    part = args.part

    df = load_data_raw(f'../WN18RR/splits/part_{part}.txt')
    out_dir = f'WN18RR_train_2hops/part={part}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = out_dir + '/{idx}.txt'
    extract_subgraphs(df, 'train', 'WN18RR', out_path)

    # test_subgraph_extraction(graph=graph)
    # df = load_data_raw('../WN18RR/test.txt')
    # out_path = 'WN18RR_test_2hops/{idx}.txt'
    # extract_subgraphs(df, 'train', 'WN18RR', out_path)










