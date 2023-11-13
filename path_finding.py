import argparse
import pandas as pd
from data_importing import load_data
from collections import Counter
from tqdm import tqdm
from neo4j import GraphDatabase
import time


def query_path(driver, relation, inv_relation, from_entity, to_entity, dataset_name, max_len, to_remove):
    # Remove the edge between 2 entities (edge only)
    driver.execute_query(f'''
    MATCH (from_entity{{name:"{from_entity}", dt_name:"{dataset_name}"}})
    -[r:{relation}]->(to_entity{{name:"{to_entity}", dt_name:"{dataset_name}" }}) 
                    DELETE r;
            ''')
    if to_remove == 'both':
        driver.execute_query(f'''
        MATCH (from_entity{{name:"{to_entity}", dt_name:"{dataset_name}"}})
        -[r:{inv_relation}]->(to_entity{{name:"{from_entity}", dt_name:"{dataset_name}" }}) 
                            DELETE r;
                    ''')
    paths, _, _ = driver.execute_query(f'''
                MATCH (from_entity{{name:"{from_entity}", dt_name:"{dataset_name}"}}), 
                (to_entity{{name:"{to_entity}", dt_name:"{dataset_name}" }}),
                p = (from_entity)-[*1..{max_len}]->(to_entity)
                RETURN DISTINCT relationships(p) AS relations, nodes(p) as nodes;
        ''')
    results = set()
    for j, path in enumerate(paths):
        path = path['relations']
        if len(path) == 1:
            if path[0].type == relation:
                print('This is impossible')
                continue
        rep_path = [relation]
        for record in path:
            rep_path.append(record.type)
        results.add(' '.join(rep_path))

    # Add the edge back
    if to_remove == 'both':
        driver.execute_query(
            f'MERGE (head:Entity {{name: "{from_entity}", dt_name: "{dataset_name}"}}) '
            f'MERGE (tail:Entity {{name: "{to_entity}", dt_name: "{dataset_name}"}}) '
            f'MERGE (head)-[:{relation}]->(tail) '
            f'MERGE (tail)-[:{inv_relation}]->(head)',
        )
    else:
        driver.execute_query(
            f'MERGE (head:Entity {{name: "{from_entity}", dt_name: "{dataset_name}"}}) '
            f'MERGE (tail:Entity {{name: "{to_entity}", dt_name: "{dataset_name}"}}) '
            f'MERGE (head)-[:{relation}]->(tail)',
        )
    return list(results)


# def query_path(driver, relation, from_entity, to_entity, dataset_name, max_len):
#     paths, _, _ = driver.execute_query(f'''
#                 MATCH (from_entity{{name:"{from_entity}", dt_name:"{dataset_name}"}}),
#                 (to_entity{{name:"{to_entity}", dt_name:"{dataset_name}" }}),
#                 p = (from_entity)-[*1..{max_len}]->(to_entity)
#                 RETURN DISTINCT relationships(p) AS relations, nodes(p) as nodes;
#         ''')
#     results = set()
#     for j, path in enumerate(paths):
#         path = path['relations']
#         if len(path) == 1:
#             if path[0].type == relation:
#                 continue
#         rep_path = [relation]
#         for record in path:
#             rep_path.append(record.type)
#         results.add(' '.join(rep_path))
#     return list(results)


def extract_paths(driver, df:pd.DataFrame, dataset_name, max_len, start_at, end_at, to_remove):
    patterns = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if i < start_at or i >= end_at:
            continue
        # head -> tail direction
        head, relation, inv_relation, tail = row['head'], row['relation'], row['inv_relation'], row['tail']
        head2tail_paths = query_path(driver=driver,
                                     relation=relation,
                                     inv_relation=inv_relation,
                                     from_entity=head,
                                     to_entity=tail,
                                     dataset_name=dataset_name,
                                     max_len=max_len,
                                     to_remove=to_remove)
        tail2head_paths = query_path(driver=driver,
                                     relation=inv_relation,
                                     inv_relation=relation,
                                     from_entity=tail,
                                     to_entity=head,
                                     dataset_name=dataset_name,
                                     max_len=max_len,
                                     to_remove=to_remove)
        patterns.extend(head2tail_paths)
        patterns.extend(tail2head_paths)
    return sorted(Counter(patterns).most_common(),
                  key=lambda x: (x[0].split()[0], -x[1]))


def count_relation(df:pd.DataFrame):
    rel_count = Counter(df['relation'].tolist() + df['inv_relation'].tolist())
    return rel_count


if __name__ == '__main__':
    # sudo systemctl start neo4j.service
    # Note: Neo4j is more than 10 times faster then Py2neo
    URI = "neo4j://localhost:7687"
    global_driver = GraphDatabase.driver(URI)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Name of dataset", required=True)
    parser.add_argument("--max_len", help="Max length", required=True, type=int)
    parser.add_argument("--start_at", help="From index", required=False, type=int, default=0)
    parser.add_argument("--end_at", help="To index", required=False, type=int, default=-1)
    parser.add_argument("--to_remove", help="both or fwd_only", required=True)
    parser.add_argument("--use_sample", help="Use sample or not", action='store_true')
    args = parser.parse_args()
    if args.dataset not in ['WN18RR', 'FB15k_237']:
        raise ValueError('Wrong dataset name!!!')
    dataset = args.dataset
    start_at = args.start_at
    end_at = args.end_at
    use_sample = args.use_sample
    to_remove = args.to_remove
    assert to_remove in ['both', 'fwd_only']

    if dataset == 'FB15k_237' and use_sample:
        train_data = load_data(f'{dataset}/train_50k.txt')
    else:
        train_data = load_data(f'{dataset}/train.txt')

    if end_at > len(train_data):
        end_at = len(train_data)

    if end_at < 0:
        train_data = train_data.iloc[start_at:]
    else:
        train_data = train_data.iloc[start_at:end_at]

    relation_count = count_relation(train_data)

    # Sample FB15k-237
    # sample = pd.read_csv(f'{dataset}/train.txt', sep='\t', dtype=str, header=None).sample(n=50000)
    # sample.to_csv(f'{dataset}/train_50k.txt', sep='\t', header=False, index=False)

    patterns = extract_paths(global_driver, train_data, dataset, args.max_len, start_at, end_at, to_remove)

    if use_sample:
        file_name = f'{dataset}/patterns_mxl_{args.max_len}_from_{start_at}_to_{end_at}_sample_50k.txt'
    else:
        file_name = f'{dataset}/patterns_mxl_{args.max_len}_from_{start_at}_to_{end_at}_remove_{to_remove}.txt'

    with open(file_name, 'w') as f:
        for i, pat in enumerate(patterns):
            f.write(f'{pat[0]} {pat[1]} {round(pat[1] / relation_count[pat[0].split()[0]], 3)}\n')



