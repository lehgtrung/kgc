import argparse
import os
import time
import pandas as pd
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt


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


def subgraph_as_atom(subgraph):
    atoms = []
    for source, target, data in subgraph.edges(data=True):
        edge_type = data.get("edge_type", "default_predicate")
        atoms.append(f"{edge_type}({source},{target}).")
    return atoms


def show_subgraph(subgraph, plot=False):
    # Print the graph as triplets
    for source, target, data in subgraph.edges(data=True):
        edge_type = data.get("edge_type", "default_predicate")
        print(f"{edge_type}({source},{target})")

    if plot:
        pos = nx.spring_layout(subgraph)  # Layout algorithm
        nx.draw(subgraph, pos, with_labels=True, node_size=1000, node_color="skyblue",
                font_size=10, font_color="black",
                font_weight="bold", arrowsize=20)
        plt.title("3-Hop Subgraph Around Node B")
        plt.show()


def test_networkx_subgraph():
    kg = nx.MultiDiGraph()
    nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    edges = [(2, 3, {'edge_type': 'linked_to_A'}),
             (3, 4, {'edge_type': 'linked_to_B'}),
             (1, 2, {'edge_type': 'linked_to_C'}),
             (7, 1, {'edge_type': 'linked_to_D'}),
             (6, 5, {'edge_type': 'linked_to_E'}),
             (5, 2, {'edge_type': 'linked_to_F'}),
             (8, 4, {'edge_type': 'linked_to_G'}),
             (9, 8, {'edge_type': 'linked_to_H'})]
    kg.add_nodes_from(nodes)
    kg.add_edges_from(edges)
    subgraph = nx.ego_graph(kg, 2, radius=3, undirected=True)
    show_subgraph(subgraph, True)


def test_edge_overlap_networkx():
    # Create a graph
    knowledge_graph = nx.MultiDiGraph()  # Use MultiDiGraph for directed multigraph

    # Add nodes and edges with attributes to the graph
    knowledge_graph.add_edge("Node A", "Node B", edge_type="connects")
    knowledge_graph.add_edge("Node A", "Node B", edge_type="has_relation")

    # Check the edges between the nodes
    edges = knowledge_graph.edges(data=True)
    for edge in edges:
        print(edge)


def extract_subgraph_networkx(df: pd.DataFrame, out_path):
    start_time = time.time()
    kg = nx.MultiDiGraph()
    head_entities = df['head'].unique().tolist()
    tail_entities = df['tail'].unique().tolist()
    all_entities = list(set(head_entities + tail_entities))
    kg.add_nodes_from(all_entities)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        kg.add_edge(
            row['head'],
            row['tail'],
            edge_type=row['relation']
        )
        kg.add_edge(
            row['tail'],
            row['head'],
            edge_type=row['inv_relation']
        )

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        head, tail = row['head'], row['tail']
        relation, inv_relation = row['relation'], row['inv_relation']
        head_subgraph = nx.ego_graph(kg, head, radius=3, undirected=True)
        tail_subgraph = nx.ego_graph(kg, tail, radius=3, undirected=True)
        head_atoms = set(subgraph_as_atom(head_subgraph))
        tail_atoms = set(subgraph_as_atom(tail_subgraph))

        expected = f'{relation}({head},{tail}).'
        inv_expected = f'{inv_relation}({tail},{head}).'
        head_atoms.remove(expected)
        tail_atoms.remove(inv_expected)

        with open(out_path.format(idx=f'{i}.head'), 'w') as f:
            f.write('%' + expected + '\n')
            f.writelines([e + '\n' for e in head_atoms])
        with open(out_path.format(idx=f'{i}.tail'), 'w') as f:
            f.write('%' + inv_expected + '\n')
            f.writelines([e + '\n' for e in tail_atoms])

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed Time:", elapsed_time, "seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Name of dataset", required=True)
    parser.add_argument("--source", help="Train or test", required=True)
    parser.add_argument("--part", help="Train or test", type=int, required=True)
    args = parser.parse_args()
    dataset = args.dataset
    source = args.source
    part = args.part

    if source == 'test':
        out_dir = f'WN18RR_{source}_3hops'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = out_dir + '/{idx}.txt'
        df = load_data(f'../{dataset}/{source}.txt')
    else:
        out_dir = f'WN18RR_{source}_2hops_networkx/part={part}'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = out_dir + '/{idx}.txt'
        df = load_data(f'../{dataset}/splits/part_{part}.txt')
    extract_subgraph_networkx(df, out_path)





















