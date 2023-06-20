from py2neo import Graph
from py2neo import Node, Relationship
import pandas as pd
from tqdm import tqdm


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


def load_data(path, ent_dct):
    data = pd.read_csv(path, sep='\t', dtype=str)
    data.columns = ['head', 'relation', 'tail']
    data['head'] = data['head'].apply(lambda x: ent_dct[x])
    data['tail'] = data['tail'].apply(lambda x: ent_dct[x])
    # data['relation'] = data['relation'].apply(lambda x: rel_dct[x])
    return data


def add_entities(graph, df, data_type, dataset_name):
    tx = graph.begin()
    # Find all entities
    head_entities = df['head'].unique().tolist()
    tail_entities = df['tail'].unique().tolist()
    all_entities = list(set(head_entities + tail_entities))

    print('Creating entities ...')
    for entity in tqdm(all_entities):
        if data_type == 'train':
            node = Node('Entity', name=entity, dtype='train', dt_name=dataset_name)
            tx.create(node)
        else:
            node = graph.nodes.match('Entity', name=entity,
                                     dtype='train', dt_name=dataset_name).first()
            if not node:
                node = Node('Entity', name=entity, dtype='test', dt_name=dataset_name)
                tx.create(node)
    graph.commit(tx)


def add_relations(graph, df, dataset_name):
    tx = graph.begin()
    print('Creating relations ...')
    for i, row in tqdm(df.iterrows(), total=len(df)):
        head, relation, tail = row['head'], row['relation'], row['tail']
        head_entity = graph.nodes.match('Entity', name=head,
                                        dtype='train', dt_name=dataset_name).first()
        if not head_entity:
            head_entity = graph.nodes.match('Entity', name=head,
                                            dtype='test', dt_name=dataset_name).first()
        tail_entity = graph.nodes.match('Entity', name=tail,
                                        dtype='train', dt_name=dataset_name).first()
        if not tail_entity:
            tail_entity = graph.nodes.match('Entity', name=tail,
                                            dtype='test', dt_name=dataset_name).first()

        relationship = Relationship(head_entity, relation, tail_entity)
        tx.create(relationship)
    graph.commit(tx)


# def add_data(_graph, df, data_type):
#     tx = _graph.begin()
#     # Find all entities
#     head_entities = df['head'].unique().tolist()
#     tail_entities = df['tail'].unique().tolist()
#     all_entities = list(set(head_entities + tail_entities))
#
#     print('Creating entities ...')
#     c = 0
#     for entity in tqdm(all_entities):
#         if data_type == 'train':
#             node = Node('Entity', name=entity, dtype='train')
#             tx.create(node)
#         else:
#             node = graph.nodes.match('Entity', name=entity, dtype='train').first()
#             if not node:
#                 print(f'{c}. {entity}')
#                 node = Node('Entity', name=entity, dtype='test')
#                 tx.create(node)
#     _graph.commit(tx)
#
#     tx = graph.begin()
#     print('Creating relations ...')
#     for i, row in tqdm(df.iterrows(), total=len(df)):
#         head, relation, tail = row['head'], row['relation'], row['tail']
#         head_entity = graph.nodes.match('Entity', name=head, dtype='train').first()
#         if not head_entity:
#             head_entity = graph.nodes.match('Entity', name=head, dtype='test').first()
#         tail_entity = graph.nodes.match('Entity', name=tail, dtype='train').first()
#         if not head_entity:
#             tail_entity = graph.nodes.match('Entity', name=tail, dtype='test').first()
#
#         relationship = Relationship(head_entity, relation, tail_entity)
#         tx.create(relationship)
#     _graph.commit(tx)


if __name__ == '__main__':
    graph = Graph('bolt://localhost:7687')
    wn18rr_ent_dct = load_dict('WN18RR/entities.dict')
    wn18rr_rel_dct = load_dict('WN18RR/relations.dict')
    wn18rr_train = load_data('WN18RR/train.txt', wn18rr_ent_dct)
    # wn18rr_test = load_data('WN18RR/test.txt', wn18rr_ent_dct)
    # add_entities(graph, wn18rr_train, 'train', 'wn18rr')
    # add_entities(graph, wn18rr_test, 'test', 'wn18rr')
    add_relations(graph, wn18rr_train, 'wn18rr')
    # add_relations(graph, wn18rr_test, 'wn18rr')

    fb15k_237_ent_dct = load_dict('FB15k-237/entities.dict')
    fb15k_237_rel_dct = load_dict('FB15k-237/relations.dict')
    fb15k_train = load_data('FB15k-237/train.txt', fb15k_237_ent_dct)
    # fb15k_test = load_data('FB15k-237/test.txt')
    add_entities(graph, fb15k_train, 'train', 'fb15k-237')
    # add_entities(graph, fb15k_test, 'test', 'fb15k-237')
    add_relations(graph, fb15k_train, 'fb15k-237')
    # add_relations(graph, fb15k_test, 'fb15k-237')



