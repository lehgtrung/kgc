import argparse

import pandas as pd
import networkx as nx
from tqdm import tqdm
from collections import Counter
from data_importing import load_data


def load_fb15k_237(train_path, valid_path, test_path):
    def _read(path):
        _data = pd.read_csv(path, sep='\t', dtype=str, header=None)
        _data.columns = ['head', 'relation', 'tail']
        _list_ents = set(_data['head'].tolist() + _data['tail'].tolist())
        _list_rels = set(_data['relation'].tolist())
        return _data, _list_ents, _list_rels

    def _apply_ents(_data, _list_ents):
        _data['head'] = _data['head'].apply(lambda x: str(_list_ents.index(x)))
        _data['tail'] = _data['tail'].apply(lambda x: str(_list_ents.index(x)))
        return _data

    def _apply_rels(_data, _list_rels):
        _data['relation'] = _data['relation'].apply(lambda x: 'r' + str(_list_rels.index(x)))
        # _data['inv_relation'] = _data['relation'].apply(lambda x: 'inv_' + x)
        return _data

    data_train, list_ents_train, list_rels_train = _read(train_path)
    data_valid, list_ents_valid, list_rels_valid = _read(valid_path)
    data_test, list_ents_test, list_rels_test = _read(test_path)

    list_ents = list(list_ents_train | list_ents_valid | list_ents_test)
    list_rels = list(list_rels_train | list_rels_valid | list_rels_test)

    data_train = _apply_ents(data_train, list_ents)
    data_train = _apply_rels(data_train, list_rels)

    data_valid = _apply_ents(data_valid, list_ents)
    data_valid = _apply_rels(data_valid, list_rels)

    data_test = _apply_ents(data_test, list_ents)
    data_test = _apply_rels(data_test, list_rels)

    return data_train, data_valid, data_test, list_ents, list_rels


def input_networkx_graph(path):
    df = load_data(path)
    kg = nx.MultiDiGraph()
    head_entities = df['head'].unique().tolist()
    tail_entities = df['tail'].unique().tolist()
    all_entities = list(set(head_entities + tail_entities))
    kg.add_nodes_from(all_entities)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        kg.add_edge(
            row['head'],
            row['tail'],
            key=row['relation']
        )
        kg.add_edge(
            row['tail'],
            row['head'],
            key=row['inv_relation']
        )
    return kg, df


def query_path_networkx(kg, relation, from_entity, to_entity, max_len):
    results = set()
    for path in nx.all_simple_edge_paths(kg, from_entity, to_entity, cutoff=max_len):
        rep_path = [relation]
        if len(path) == 1 and path[0][-1] == relation:
            continue
        for record in path:
            rep_path.append(record[-1])
        results.add(' '.join(rep_path))
    return list(results)


def extract_paths_networkx(kg, df: pd.DataFrame, max_len):
    patterns = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        # head -> tail direction
        head, relation, inv_relation, tail = row['head'], row['relation'], row['inv_relation'], row['tail']
        head2tail_paths = query_path_networkx(kg=kg,
                                              relation=relation,
                                              from_entity=head,
                                              to_entity=tail,
                                              max_len=max_len)
        tail2head_paths = query_path_networkx(kg=kg,
                                              relation=inv_relation,
                                              from_entity=tail,
                                              to_entity=head,
                                              max_len=max_len)
        patterns.extend(head2tail_paths)
        patterns.extend(tail2head_paths)
    return sorted(Counter(patterns).most_common(),
                  key=lambda x: (x[0].split()[0], -x[1]))


def count_relation(df:pd.DataFrame):
    rel_count = Counter(df['relation'].tolist() + df['inv_relation'].tolist())
    return rel_count


if __name__ == '__main__':
    # df_train, df_valid, df_test, list_ents, list_rels = load_fb15k_237('FB15k_237/train.txt.bk',
    #                                                                    'FB15k_237/valid.txt.bk',
    #                                                                    'FB15k_237/test.txt.bk')
    #
    # df_train.to_csv('FB15k_237/train.txt', header=False, sep='\t', index=False)
    # df_valid.to_csv('FB15k_237/valid.txt', header=False, sep='\t', index=False)
    # df_test.to_csv('FB15k_237/test.txt', header=False, sep='\t', index=False)
    #
    # with open('FB15k_237/list_ents.txt', 'w') as f:
    #     f.writelines([e + '\n' for e in list_ents])
    # with open('FB15k_237/list_rels.txt', 'w') as f:
    #     f.writelines([e + '\n' for e in list_rels])

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Name of dataset", required=True)
    parser.add_argument("--max_len", help="Max length", required=True, type=int)
    args = parser.parse_args()
    if args.dataset not in ['WN18RR', 'FB15k_237']:
        raise ValueError('Wrong dataset name!!!')
    dataset = args.dataset
    max_len = args.max_len

    kg, train_data = input_networkx_graph(f'{dataset}/train.txt')
    relation_count = count_relation(train_data)

    patterns = extract_paths_networkx(kg, train_data, max_len)

    with open(f'{dataset}/patterns_mxl_{max_len}_networkx.txt', 'w') as f:
        for i, pat in enumerate(patterns):
            f.write(f'{pat[0]} {pat[1]} {round(pat[1] / relation_count[pat[0].split()[0]], 3)}\n')



















