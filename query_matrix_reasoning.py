import argparse
import numpy as np
from tqdm import tqdm
import torch
import re
import random
import pandas as pd


def normalize_relation(x: str):
    inv_flag = False
    if x.startswith('!'):
        inv_flag = True
        x = x.lstrip('!')
    x = x.lstrip('_')
    if inv_flag:
        return 'inv_' + x
    return x


def load_data_raw(path):
    data = pd.read_csv(path, sep='\t', dtype=str, header=None)
    data.columns = ['head', 'relation', 'tail']
    data['head'] = data['head'].apply(lambda x: 'e' + str(x))
    data['tail'] = data['tail'].apply(lambda x: 'e' + str(x))
    data['relation'] = data['relation'].apply(lambda x: normalize_relation(x))
    data['inv_relation'] = data['relation'].apply(lambda x: normalize_relation('!' + x))
    return data


def encode_rules(rule_path, max_rank):
    with open(rule_path, 'r') as f:
        raw_rule_list = [e.strip().split() for e in f.readlines()]
    rules = {}
    for line in raw_rule_list:
        rule_head = normalize_relation(line[0])
        rule_conf = int(line[-2])
        rule_body = [normalize_relation(e) for e in line[1:-2]]
        if rule_head not in rules:
            rules[rule_head] = []
        if len(rules[rule_head]) < max_rank:
            rules[rule_head].append(
                {
                    'conf': rule_conf,
                    'body': rule_body
                }
            )
    return rules


def compile_data_as_adj_matrix(df: pd.DataFrame,
                               list_ents,
                               list_rels):
    n = len(list_ents)
    global_indices = {}
    global_matrix = {}
    for rel in list_rels:
        if rel.startswith('inv'):
            df_rel = df.query(f'inv_relation == "{rel}"')
        else:
            df_rel = df.query(f'relation == "{rel}"')
        head_indices = df_rel['head_idx'].tolist()
        tail_indices = df_rel['tail_idx'].tolist()
        indices = torch.LongTensor([tail_indices, head_indices])
        sparse_matrix = torch.sparse_coo_tensor(indices,
                                                torch.ones_like(indices[0]),
                                                torch.Size([n, n])).to(torch.float32)
        global_indices[rel] = indices
        global_matrix[rel] = sparse_matrix
    return global_matrix, global_indices


def modify_adj_mat(indices, first2remove_idx, second2remove_idx, n):
    list_indices = indices.clone().numpy().tolist()
    new_x = []
    new_y = []
    for (x, y) in zip(list_indices[0], list_indices[1]):
        if (x, y) == (first2remove_idx, second2remove_idx):
            continue
        new_x.append(x)
        new_y.append(y)

    indices = torch.LongTensor([new_x, new_y])
    sparse_matrix = torch.sparse_coo_tensor(indices,
                                            torch.ones_like(indices[0]),
                                            torch.Size([n, n])).to(torch.float32)
    return sparse_matrix


def matmul_with_customized_matrices(rules, global_matrix, n, idx2predict):
    sub_rules_output = torch.sparse_coo_tensor(torch.Size([n, n])).to(torch.float32)
    for rule in rules:
        conf = rule['conf']
        body = rule['body']

        prod = global_matrix[body[0]]
        for i, rel in enumerate(body[1:]):
            prod = torch.matmul(prod, global_matrix[rel])
        prod = prod * conf
        sub_rules_output += prod
    return sub_rules_output[idx2predict].to_dense().numpy()


def get_rank_at(arr, idx):
    sorted_indices = np.argsort(arr, kind='stable')[::-1]
    # Create an array of ranks based on the sorted indices
    ranks = np.empty(len(arr), int)
    ranks[sorted_indices] = np.arange(len(arr)) + 1  # Adding 1 to start ranks from 1
    return ranks[idx], arr[idx]

def mean_rank(arr):
    return np.mean(arr)


def mean_reciprocal_rank(arr):
    return np.mean([1.0/e for e in arr])


def hit_at(arr, at=10):
    hit = 0
    for rank in arr:
        if rank <= at:
            hit += 1
    return hit / len(arr)


def answer_query(df: pd.DataFrame, complied_rules, global_matrix, global_indices, list_ents):
    ranks = []
    n = len(list_ents)
    for i, row in tqdm(df.iterrows(), total=len(df)):
        head_idx = list_ents.index(row['head'])
        tail_idx = list_ents.index(row['tail'])
        relation = row['relation']
        inv_relation = row['inv_relation']

        global_matrix[relation] = modify_adj_mat(global_indices[relation], head_idx, tail_idx, n)
        global_matrix[inv_relation] = modify_adj_mat(global_indices[inv_relation], tail_idx, head_idx, n)

        # Answer fwd query
        rules = complied_rules[relation]
        fwd_answers = matmul_with_customized_matrices(rules, global_matrix, n, head_idx)
        fwd_rank = get_rank_at(fwd_answers, tail_idx)

        # Answer bwd query
        rules = complied_rules[inv_relation]
        bwd_answers = matmul_with_customized_matrices(rules, global_matrix, n, tail_idx)
        bwd_rank = get_rank_at(bwd_answers, head_idx)

        ranks.extend([fwd_rank, bwd_rank])
    return ranks


def show_results(ranks):
    print('MR: ', mean_rank(ranks))
    print('MRR: ', mean_reciprocal_rank(ranks))
    print('Hit@10: ', hit_at(ranks, 10))
    print('Hit@3: ', hit_at(ranks, 3))
    print('Hit@1: ', hit_at(ranks, 1))


if __name__ == '__main__':
    print('============================================')
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Name of dataset", required=True)
    parser.add_argument("--rule_path", help="Path to rule set", required=True)
    parser.add_argument("--max_rank", help="Max number of rules used", required=False, type=int, default=10)

    args = parser.parse_args()
    dataset = args.dataset
    max_len = 3
    rule_path = args.rule_path
    max_rank = args.max_rank

    df_train = load_data_raw(f'{dataset}/train.txt')
    df_test = load_data_raw(f'{dataset}/test.txt')
    df_valid = load_data_raw(f'{dataset}/valid.txt')

    df_all = pd.concat([df_train, df_valid, df_test], ignore_index=True)
    # df_all = df_test

    list_ents = list(set(df_all['head'].to_list() + df_all['tail'].to_list()))
    list_rels = list(set(df_all['relation'].to_list() + df_all['inv_relation'].to_list()))

    df_all['head_idx'] = df_all['head'].apply(lambda x: list_ents.index(x))
    df_all['tail_idx'] = df_all['tail'].apply(lambda x: list_ents.index(x))

    print(f'Testing on {dataset}...')
    print(f'Maximum rule length {max_len}...')
    print(f'Maximum number of rules {max_rank}...')
    print(f'There are {len(list_ents)} entities in total!')
    print(f'There are {len(list_rels)} relations in total!')

    complied_rules = encode_rules(rule_path, max_rank)
    global_matrix, global_indices = compile_data_as_adj_matrix(df_all, list_ents, list_rels)
    ranks = answer_query(df_all, complied_rules, global_matrix, global_indices, list_ents)
    show_results(ranks)






























