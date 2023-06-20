
from py2neo import Graph
from py2neo import Node, Relationship
import pandas as pd
from main import load_dict, load_data
from collections import defaultdict, Counter
from tqdm import tqdm
import json


def extract_rel_types_from_path(path):
    result = []
    for rel in path['r']:
        result.append(str(rel).split(':')[1].split(' ')[0])
    return tuple(result)


def reconstruct_rnn_logic_rules(path, dct):
    rnn_dct = {}
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    for line in lines:
        pattern = [dct[int(e)] for e in line.split(' ')]
        head = pattern[0]
        if head not in rnn_dct:
            rnn_dct[head] = Counter()
        rnn_dct[head][tuple(pattern[1:])] += 1
    return post_process_patterns(rnn_dct)


def post_process_patterns(patterns):
    new_patterns = {}
    for key in patterns:
        total = sum(patterns[key].values())
        new_patterns[key] = {}
        new_patterns[key]['total'] = total
        new_patterns[key]['patterns'] = []
        for each in patterns[key].most_common():
            new_patterns[key]['patterns'].append({
                'key': each[0],
                'count': each[1],
                'pct': float('{:.2f}'.format(each[1] / total * 100))
            })
    return new_patterns


def extract_path_from_train(graph:Graph, df:pd.DataFrame, dataset_name):
    patterns = {}
    for i, row in tqdm(df.iterrows(), total=len(df)):
        head, relation, tail = row['head'], row['relation'], row['tail']
        # head = 3514
        # tail = 3515
        query = graph.run(f'''
        MATCH (head{{name:{head}, dt_name:"{dataset_name}"}}), 
        (tail{{name:{tail}, dt_name:"{dataset_name}" }}),
        p = (head)-[r*..5]->(tail)
        RETURN r ORDER BY length(p);
        ''')
        paths = query.data()
        if relation not in patterns:
            patterns[relation] = Counter()
        for j, path in enumerate(paths):
            if j > 0:
                local_path = extract_rel_types_from_path(path)
                patterns[relation][local_path] += 1
    return post_process_patterns(patterns)


if __name__ == '__main__':
    graph = Graph('bolt://localhost:7687')
    wn18rr_ent_dct = load_dict('WN18RR/entities.dict')
    wn18rr_rel_dct = load_dict('WN18RR/relations.dict', inv=True)
    wn18rr_train = load_data('WN18RR/train.txt', wn18rr_ent_dct)
    wn18rr_test = load_data('WN18RR/test.txt', wn18rr_ent_dct)
    # wn18rr_train_patterns = extract_path_from_train(graph, wn18rr_train, 'wn18rr')
    # wn18rr_test_patterns = extract_path_from_train(graph, wn18rr_test, 'wn18rr')

    # with open('WN18RR/train_patterns.json', 'w') as f:
    #     json.dump(wn18rr_train_patterns, f, indent=4)
    # with open('WN18RR/test_patterns.json', 'w') as f:
    #     json.dump(wn18rr_test_patterns, f, indent=4)

    # with open('WN18RR/train_patterns.json', 'r') as f:
    #     wn18rr_patterns = json.load(f)
    # print(wn18rr_patterns)

    rnn_logic_patterns = reconstruct_rnn_logic_rules('WN18RR/rnnlogic_rules.txt', wn18rr_rel_dct)
    with open('WN18RR/rnn_logic_patterns.json', 'w') as f:
        json.dump(rnn_logic_patterns, f, indent=4)