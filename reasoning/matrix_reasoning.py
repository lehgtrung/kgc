
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from reasoning_utils import *
import torch

# Check if CUDA (GPU) is available
cuda_available = torch.cuda.is_available()

if cuda_available:
    print("CUDA (GPU) is available.")
else:
    print("CUDA (GPU) is not available.")


def encode_data_as_adj_mat(df:pd.DataFrame):
    _adj_mat = {}
    _sparse_adj_mat = {}
    _entity_list = list(set(df['head'].to_list() + df['tail'].to_list()))
    _relation_list = list(set(df['relation'].to_list() + df['inv_relation'].to_list()))
    num_entities = len(_entity_list)

    for rel in _relation_list:
        _adj_mat[rel] = np.zeros((num_entities, num_entities))

    for i, row in tqdm(df.iterrows(), total=len(df)):
        head_idx = _entity_list.index(row['head'])
        tail_idx = _entity_list.index(row['tail'])
        relation = row['relation']
        inv_relation = row['inv_relation']

        _adj_mat[relation][head_idx][tail_idx] = 1.0

        _adj_mat[inv_relation][tail_idx][head_idx] = 1.0

    for key in _adj_mat:
        _sparse_adj_mat[key] = torch.tensor(_adj_mat[key])
        # sparse_adj_mat[key] = csr_matrix(adj_mat[key])
        _sparse_adj_mat[key] = _sparse_adj_mat[key].to_sparse()
    return _sparse_adj_mat, _entity_list, _relation_list


def encode_rules(rule_path, max_rank=10):
    with open(rule_path, 'r') as f:
        raw_rule_list = [e.strip() for e in f.readlines()]
    rules = {}
    for line in raw_rule_list:
        line = line.split()
        if float(line[-1]) < 0.1:
            continue
        rule_head = normalize_relation(line[0])
        rule_conf = float(line[-1])
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


def rule_as_mat_mul(adj_mat, rules: dict, n):
    # rules: keys are relations, values are list of tuples
    # with elem 0 rep conf and the rest are relations
    matrix_results = {}
    for query_key in tqdm(rules, total=len(rules)):
        print(query_key)
        sub_rules_output = torch.zeros((n, n))
        for rule in rules[query_key]:
            print(rule)
            conf = rule['conf']
            body = rule['body']
            prod = adj_mat[body[0]]
            for rel in body[1:]:
                prod = torch.matmul(prod, adj_mat[rel])
            prod = prod.to_dense() * conf
            sub_rules_output += prod
        matrix_results[query_key] = sub_rules_output.to('cpu').numpy()
    return matrix_results


def get_rank_at(arr, idx):
    sorted_indices = np.argsort(arr)[::-1]
    # Create an array of ranks based on the sorted indices
    ranks = np.empty(len(arr), int)
    ranks[sorted_indices] = np.arange(len(arr)) + 1  # Adding 1 to start ranks from 1
    return ranks[idx], arr[idx]


def mean_reciprocal_rank(arr):
    mrr = 0
    for rank in arr:
        mrr += 1/rank
    return 1/len(arr) * mrr


def hit_at(arr, at=10):
    hit = 0
    for rank in arr:
        if rank <= at:
            hit += 1
    return hit / len(arr)


def answer_queries(df_test: pd.DataFrame, matrix_results: dict, entity_list: list):
    mrr = []
    values = []
    out_of_dist_count = 0
    for i, row in df_test.iterrows():
        if row['head'] not in entity_list or row['tail'] not in entity_list:
            out_of_dist_count += 1
            continue
        head_idx = entity_list.index(row['head'])
        tail_idx = entity_list.index(row['tail'])
        relation = row['relation']
        inv_relation = row['inv_relation']

        forward_result = matrix_results[relation][head_idx]
        backward_result = matrix_results[inv_relation][tail_idx]

        forward_rank, forward_value = get_rank_at(forward_result, tail_idx)
        backward_rank, backward_value = get_rank_at(backward_result, head_idx)

        mrr.append(forward_rank)
        mrr.append(backward_rank)
        values.append(forward_value)
        values.append(backward_value)
    print('Number of ood = ', out_of_dist_count)
    return mrr, values


if __name__ == '__main__':
    df_train = load_data_raw('../WN18RR/train.txt')
    df_test = load_data_raw('../WN18RR/test.txt')
    sparse_adj_mat, entity_list, relation_list = encode_data_as_adj_mat(df_train)
    print('Finish embedding data as adj matrix')
    rules = encode_rules('../WN18RR/patterns_mxl_3.txt')

    rules_at_mat = rule_as_mat_mul(sparse_adj_mat, rules, len(entity_list))
    print('Finish encoding rules')
    # mrr = answer_queries(df_test, rules_at_mat, entity_list)
    mrr, values = answer_queries(df_train, rules_at_mat, entity_list)
    print(mrr)
    print('--------------------------------------')
    print(values)
    print('MRR: ', mean_reciprocal_rank(mrr))
    print('Hit@10: ', hit_at(mrr, 10))
    print('Hit@3: ', hit_at(mrr, 3))
    print('Hit@1: ', hit_at(mrr, 1))

    # Test data
    # MRR: 0.2603472098878333
    # Hit @ 10: 0.45194938440492477
    # Hit @ 3: 0.3288303693570451
    # Hit @ 1: 0.161593707250342

    # Train data
    # MRR: 0.7011913358847316
    # Hit @ 10: 0.9123970749121898
    # Hit @ 3: 0.798013473829677
    # Hit @ 1: 0.5779812287672021























