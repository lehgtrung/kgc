
import pandas as pd
from collections import Counter
import argparse
from data_importing import load_data


def count_relation(df:pd.DataFrame):
    rel_count = Counter(df['relation'].tolist() + df['inv_relation'].tolist())
    return rel_count


def merge_dicts(*paths):
    rules_counter = Counter()
    for path in paths:
        with open(path, 'r') as f:
            lines = [e.strip() for e in f.readlines()]
            for line in lines:
                line = line.split()
                rule = tuple(line[:-2])
                count = int(line[-2])
                rules_counter[rule] += count
    return rules_counter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A program that takes variable-length input.")
    parser.add_argument('paths', nargs='*', help='Variable-length input arguments')
    args = parser.parse_args()

    train_data = load_data(f'FB15k_237/train_50k.txt')
    relation_count = count_relation(train_data)

    rules_counter = merge_dicts(*args.paths)
    with open(f'FB15k_237/patterns_mxl_3.txt', 'w') as f:
        for i, rule in enumerate(rules_counter):
            f.write(f"{' '.join(rule)} {rules_counter[rule]} {round(rules_counter[rule] / relation_count[rule[0]], 3)}\n")


