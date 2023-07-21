import argparse

from py2neo import Graph
from py2neo import Node, Relationship
import pandas as pd
from main import load_dict, load_data
from collections import defaultdict, Counter
from tqdm import tqdm
import json


class Rule:
    def __init__(self, line, with_support):
        line = line.strip('\n').split(' ')
        self.support = 0
        self.head = line[0]
        if with_support:
            self.body = line[1:-1]
            self.support = int(line[-1])
        else:
            self.body = line[1:]

    def to_string(self, with_sup=False):
        if with_sup:
            return f'{self.head} {" ".join(self.body)} {self.support}'
        return f'{self.head} {" ".join(self.body)}'

    def length(self):
        return len(self.body)


def decode_rnn_logic_rules(in_path, dct, out_path):
    text_rules = []
    with open(in_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            rule = line.strip('\n').split(' ')
            rule = [dct[int(e)] for e in rule]
            rule = ' '.join(rule) + '\n'
            text_rules.append(rule)
    text_rules = sorted(text_rules, key=lambda x: x.split(' ')[0])
    with open(out_path, 'w') as f:
        f.writelines(text_rules)


def read_rules(in_path, with_support):
    rules = []
    with open(in_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            rules.append(Rule(line, with_support))
    reg_rules = sorted(rules, key=lambda x: (x.head, -x.support))
    return reg_rules


def reorganize_rules(in_path, out_path, min_sup, with_sup):
    reg_rules = read_rules(in_path, True)
    with open(out_path, 'w') as f:
        for rule in reg_rules:
            if rule.support >= min_sup:
                f.write(rule.to_string(with_sup) + '\n')


def traverse_path(graph, head, dataset_name, body):
    if isinstance(body, str):
        body = body.split(' ')
    body_query = ''
    for i, atom in enumerate(body):
        body_query += f'-[:`{atom}`]->'
        if i != len(body)-1:
            body_query += '()'
    query = f'''
        MATCH (from_entity{{name:{head}, dt_name:"{dataset_name}"}}){body_query}(to_entity)
        RETURN DISTINCT to_entity
    '''
    tails = graph.run(query).data()
    return [e['to_entity']['name'] for e in tails]


def eval_with_rules(graph, df:pd.DataFrame, dataset_name, rules):
    accuracy = 0
    for i, row in tqdm(df.iterrows(), total=len(df)):
        head, relation, tail = row['head'], row['relation'], row['tail']
        for rule in rules:
            if rule.head == relation and rule.length() <= 3:
                # print(rule.to_string())
                if rule.body:
                    prop_tails = traverse_path(graph, head, dataset_name, rule.body)
                    if tail in prop_tails:
                        accuracy += 1
                        break
    print('Accuracy: ', accuracy / len(df))


if __name__ == '__main__':
    graph = Graph('bolt://localhost:7687')

    # reorganize_rules('WN18RR/patterns_mxl_3.txt',
    #                  'WN18RR/reg_patterns_mxl_3.txt',
    #                  5,
    #                  False)

    # print(traverse_path(graph, 1678, 'WN18RR',
    #               '_member_of_domain_usage !_hypernym'))

    # dataset = 'WN18RR'
    # rel_dct = load_dict(f'{dataset}/relations.dict', inv=True)
    # decode_rnn_logic_rules(f'{dataset}/rnnlogic_rules.txt',
    #                        rel_dct,
    #                        f'{dataset}/rnnlogic_rules_decoded.txt')

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Name of dataset", required=True)
    parser.add_argument("--which", help="Which rule", required=True)
    args = parser.parse_args()
    if args.dataset not in ['WN18RR', 'FB15k-237']:
        raise ValueError('Wrong dataset name!!!')
    dataset = args.dataset
    ent_dct = load_dict(f'{dataset}/entities.dict')
    rel_dct = load_dict(f'{dataset}/relations.dict')
    test_data = load_data(f'{dataset}/test.txt', ent_dct)

    algo_rule_path = f'{dataset}/reg_patterns_mxl_3.txt'
    rnnlogic_rule_path = f'{dataset}/rnnlogic_rules_decoded.txt'

    if args.which == 'algo':
        rules = read_rules(algo_rule_path, False)
        print('Evaluate with algorithmic rules:')
        eval_with_rules(graph, test_data, dataset, rules)
    else:
        rules = read_rules(rnnlogic_rule_path, False)
        print('Evaluate with rnn logic rules:')
        eval_with_rules(graph, test_data, dataset, rules)


