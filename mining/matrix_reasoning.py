import argparse
import numpy as np
from tqdm import tqdm
from reasoning_utils import *
import torch
import re
import random
from scipy.stats import rankdata


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Check if CUDA (GPU) is available
cuda_available = torch.cuda.is_available()

if cuda_available:
    print("CUDA (GPU) is available.")
else:
    print("CUDA (GPU) is not available.")


def encode_data_as_adj_mat(df:pd.DataFrame, _relation_list):
    _adj_mat = {}
    _entity_list = list(set(df['head'].to_list() + df['tail'].to_list()))
    num_entities = len(_entity_list)

    for rel in _relation_list:
        _adj_mat[rel] = torch.zeros((num_entities, num_entities), dtype=torch.float)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        head_idx = _entity_list.index(row['head'])
        tail_idx = _entity_list.index(row['tail'])
        relation = row['relation']
        inv_relation = row['inv_relation']

        _adj_mat[relation][head_idx][tail_idx] = 1.0

        _adj_mat[inv_relation][tail_idx][head_idx] = 1.0

    for key in _adj_mat:
        _adj_mat[key] = _adj_mat[key].to_sparse()
    return _adj_mat, _entity_list, _relation_list


def encode_rules(rule_path, max_rank):
    with open(rule_path, 'r') as f:
        raw_rule_list = [e.strip().split() for e in f.readlines()]
    rules = {}
    for line in raw_rule_list:
        # line = line.split()
        rule_head = normalize_relation(line[0])
        # rule_conf = float(line[-1])
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


def rule_as_mat_mul(adj_mat, rules: dict, n, conf_mode='keep'):
    # rules: keys are relations, values are list of tuples
    # with elem 0 rep conf and the rest are relations
    matrix_results = {}
    for query_key in tqdm(rules, total=len(rules)):
        sub_rules_output = torch.zeros((n, n))
        for rule in rules[query_key]:
            conf = rule['conf']
            if conf_mode == 'fixed':
                conf = 0.5
            elif conf_mode == 'random':
                conf = random.gauss(0.5, 0.1)
            body = rule['body']
            prod = adj_mat[body[0]]
            for rel in body[1:]:
                prod = torch.matmul(prod, adj_mat[rel])
            prod = prod.to_dense() * conf
            sub_rules_output += prod
        matrix_results[query_key] = sub_rules_output.to('cpu').numpy()
    return matrix_results


# def get_rank_at(arr, idx):
#     sorted_indices = np.argsort(arr, kind='stable')[::-1]
#     # Create an array of ranks based on the sorted indices
#     ranks = np.empty(len(arr), int)
#     ranks[sorted_indices] = np.arange(len(arr)) + 1  # Adding 1 to start ranks from 1
#     return ranks[idx], arr[idx]


def get_rank_at(arr, idx):
    return rankdata([-e for e in arr], method='ordinal')[idx], arr[idx]


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


def answer_queries(df: pd.DataFrame, matrix_results: dict, entity_list: list):
    mrr = []
    values = []
    out_of_dist_count = 0
    r33_count = 0
    for i, row in df.iterrows():
        if row['relation'] in ['r33']:  # special case for FB15k-237
            r33_count += 1
            continue
        if row['head'] not in entity_list or row['tail'] not in entity_list:
            out_of_dist_count += 1
            continue
        head_idx = entity_list.index(row['head'])
        tail_idx = entity_list.index(row['tail'])
        relation = row['relation']
        inv_relation = row['inv_relation']

        forward_result = matrix_results[relation][head_idx]
        backward_result = matrix_results[inv_relation][tail_idx]

        # forward_rank = get_rank_at(forward_result, tail_idx)
        forward_rank, forward_value = get_rank_at(forward_result, tail_idx)
        # backward_rank = get_rank_at(backward_result, head_idx)
        backward_rank, backward_value = get_rank_at(backward_result, head_idx)

        mrr.append(forward_rank)
        mrr.append(backward_rank)
        values.append(forward_value)
        values.append(backward_value)
    return mrr, values


def check_if_tail_in_subgraph(path):
    with open(path, 'r') as f:
        lines = [e.strip() for e in f.readlines()]
    query = lines.pop(0)
    result = re.search(r'\w+\((\w+),(\w+)\)', query)
    head, tail = result.group(1), result.group(2)
    head_flag = tail_flag = False
    for line in lines:
        if head in line:
            head_flag = True
        if tail in line:
            tail_flag = True
        if head_flag and tail_flag:
            return 1
    return 0


def show_results(mrr, values):
    num_zeros = len([e for e in values if e == 0.0])
    print('Num 0s in values: ', num_zeros)
    print('Percentage 0s in values: ', num_zeros / len(mrr))
    print('MR: ', mean_rank(mrr))
    print('MRR: ', mean_reciprocal_rank(mrr))
    print('Hit@10: ', hit_at(mrr, 10))
    print('Hit@3: ', hit_at(mrr, 3))
    print('Hit@1: ', hit_at(mrr, 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Name of dataset", required=True)
    parser.add_argument("--source", help="Which matrix is used (train/test/all)", required=True)
    parser.add_argument("--conf_mode", help="Mode (keep/fixed/random)", required=False, default='keep')
    parser.add_argument("--max_len", help="Rule max length", required=False, type=int, default=3)
    parser.add_argument("--use_sample", help="Use sample or not", action='store_true')
    parser.add_argument("--max_rank", help="Max number of rules used", required=False, type=int, default=10)
    args = parser.parse_args()
    dataset = args.dataset
    source = args.source
    conf_mode = args.conf_mode
    max_len = args.max_len
    use_sample = args.use_sample
    max_rank = args.max_rank

    if use_sample:
        df_train = load_data_raw(f'../{dataset}/train_sampled.txt')
        df_test = load_data_raw(f'../{dataset}/test_sampled.txt')
        df_valid = load_data_raw(f'../{dataset}/valid_sampled.txt')
    else:
        df_train = load_data_raw(f'../{dataset}/train.txt')
        df_test = load_data_raw(f'../{dataset}/test.txt')
        df_valid = load_data_raw(f'../{dataset}/valid.txt')
    df_all = pd.concat([df_train, df_valid, df_test], ignore_index=True)
    list_ents = list(set(df_all['head'].to_list() + df_all['tail'].to_list()))
    list_rels = list(set(df_all['relation'].to_list() + df_all['inv_relation'].to_list()))
    print(f'Testing on {dataset}...')
    print(f'Source {source}...')
    print(f'Confidence mode {conf_mode}...')
    print(f'Maximum rule length {max_len}...')
    print(f'Maximum number of rules {max_rank}...')
    print(f'There are {len(list_ents)} entities in total!')
    print(f'There are {len(list_rels)} relations in total!')

    if source not in ['train', 'test', 'all']:
        raise ValueError('Wrong source name!!!')

    if source == 'train':
        sparse_adj_mat, entity_list, relation_list = encode_data_as_adj_mat(df_train, list_rels)
    elif source == 'test':
        sparse_adj_mat, entity_list, relation_list = encode_data_as_adj_mat(df_test, list_rels)
    else:
        sparse_adj_mat, entity_list, relation_list = encode_data_as_adj_mat(df_all, list_rels)

    print(f'Finish embedding {source} data as adj matrix')
    rules = encode_rules(f'../{dataset}/patterns_mxl_{max_len}.txt', max_rank)

    rules_at_mat = rule_as_mat_mul(sparse_adj_mat, rules, len(entity_list), conf_mode)
    print('Finish encoding rules')
    mrr_test, values_test = answer_queries(df_test, rules_at_mat, entity_list)
    mrr_train, values_train = answer_queries(df_train, rules_at_mat, entity_list)
    print('Train result')
    show_results(mrr_train, values_train)
    print('Test result')
    show_results(mrr_test, values_test)






















