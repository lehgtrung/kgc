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


def show_subgraph(subgraph, plot=False):
    # Print the graph as triplets
    for source, target, data in subgraph.edges(data=True):
        edge_type = data.get("edge_type", "default_predicate")
        # print(f"{source} -[{edge_type}]-> {target}")
        print(f"{edge_type}({source},{target})")

    # print("\nTriples for nodes:")
    # for node in subgraph.nodes():
    #     print(f"Node: {node}")

    if plot:
        pos = nx.spring_layout(subgraph)  # Layout algorithm
        nx.draw(subgraph, pos, with_labels=True, node_size=1000, node_color="skyblue", font_size=10, font_color="black",
                font_weight="bold", arrowsize=20)
        plt.title("3-Hop Subgraph Around Node B")
        plt.show()


def test_networkx_subgraph():
    kg = nx.MultiDiGraph()
    nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    edges = [(2, 3, {'edge_type': 'linked_to'}),
             (3, 4, {'edge_type': 'linked_to'}),
             (1, 2, {'edge_type': 'linked_to'}),
             (7, 1, {'edge_type': 'linked_to'}),
             (6, 5, {'edge_type': 'linked_to'}),
             (5, 2, {'edge_type': 'linked_to'}),
             (8, 4, {'edge_type': 'linked_to'}),
             (9, 8, {'edge_type': 'linked_to'})]
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


def import_kg_to_networkx(df: pd.DataFrame):
    start_time = time.time()
    kg = nx.MultiDiGraph()
    head_entities = df['head'].unique().tolist()
    tail_entities = df['tail'].unique().tolist()
    all_entities = list(set(head_entities + tail_entities))
    kg.add_nodes_from(all_entities)
    edges = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        edges.append((
            row['head'],
            row['tail'],
            {'edge_type': row['relation']}
        ))
        edges.append((
            row['tail'],
            row['head'],
            {'edge_type': row['inv_relation']}
        ))
    print(len(edges))
    kg.add_edges_from(edges)
    subgraph = nx.ego_graph(kg, 'e02103406', radius=2, undirected=True)
    show_subgraph(subgraph)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed Time:", elapsed_time, "seconds")


if __name__ == '__main__':
    df = load_data('../WN18RR/train.txt')
    import_kg_to_networkx(df)
    # test_networkx_subgraph()
    # test_edge_overlap_networkx()





















