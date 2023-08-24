import argparse
import ast

import pandas as pd
import subprocess


COMMAND = 'clingo --opt-mode=optN WN18RR_rules.lp WN18RR_{mode}_atoms.lp ' \
          '--outf=0 -V0 --out-atomf=%s. --quiet=1,2,2'


def normalize_relation(x: str):
    inv_flag = False
    if x.startswith('!'):
        inv_flag = True
        x = x.lstrip('!')
    x = x.lstrip('_')
    splits = x.split('_')
    if inv_flag:
        # return 'inv' + ''.join([e.capitalize() for e in splits])
        return 'inv' + ''.join([e for e in splits])
    # return splits[0] + ''.join([e.capitalize() for e in splits[1:]])
    return ''.join([e for e in splits])


def create_inv_relation(x: str):
    return normalize_relation('!' + x)


def load_data_raw(path):
    data = pd.read_csv(path, sep='\t', dtype=str)
    data.columns = ['head', 'relation', 'tail']
    data['head'] = data['head'].apply(lambda x: 'E' + str(x))
    data['tail'] = data['tail'].apply(lambda x: 'E' + str(x))
    data['relation'] = data['relation'].apply(lambda x: normalize_relation(x))
    data['inv_relation'] = data['relation'].apply(lambda x: create_inv_relation(x))
    return data


def convert_data_to_atoms(df):
    atoms = []
    for i, row in df.iterrows():
        atoms.append(f"{row['relation']}(\"{row['head']}\",\"{row['tail']}\").")
        atoms.append(f"{row['inv_relation']}(\"{row['tail']}\",\"{row['head']}\").")
    return atoms


def load_dict(path, num_as_key=False):
    dct = {}
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        for line in lines:
            v, k = line.split('\t')
            dct[k] = int(v)
    if num_as_key:
        return {v: k for k, v in dct.items()}
    return dct


def form_rule(_rule):
    result = f'{_rule[0]}(X,Y) :- {_rule[1]}(X,Z1)'
    k = 1
    for each in _rule[2:-1]:
        result += f',{each}(Z{k},Z{k+1})'
        k += 1
    result += f',{_rule[-1]}(Z{k},Y).'
    return result


def encode_rules(rules_path, dct):
    rules = []
    with open(rules_path, 'r') as f:
        _rules = f.readlines()
        for i, _rule in enumerate(_rules):
            if _rule:
                _rule = _rule.split()
                _rule = [normalize_relation(dct[int(e)]) for e in _rule]
                if len(_rule) > 1:
                    rules.append(form_rule(_rule))
    return rules


def write_things(things, path):
    with open(path, 'w') as f:
        f.writelines([e + '\n' for e in things])


def detect_inferred_atoms(pivot_atoms, atoms):
    return list(set(atoms) - set(pivot_atoms))


def solve(command):
    process = subprocess.Popen(command,
                               shell=True,
                               stdin=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               stdout=subprocess.PIPE)
    output, error = process.communicate()
    answerset = output.decode().split('\n')[:-2]
    atoms = answerset[0]
    atoms = atoms.split(' ')
    return atoms



if __name__ == '__main__':
    df_train = load_data_raw('../WN18RR/train.txt')
    df_test = load_data_raw('../WN18RR/test.txt')
    train_atoms = convert_data_to_atoms(df_train)
    test_atoms = convert_data_to_atoms(df_test)
    dct = load_dict('../WN18RR/relations.dict', True)
    rules = encode_rules('../WN18RR/rnnlogic_rules.txt', dct)
    write_things(rules, 'WN18RR_rules.lp')
    write_things(train_atoms, 'WN18RR_train_atoms.lp')
    write_things(test_atoms, 'WN18RR_test_atoms.lp')

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="Train or test", required=True)
    args = parser.parse_args()
    mode = args.mode

    command = COMMAND.format(mode=mode)
    as_atoms = solve(command)
    pivot_atoms = train_atoms
    if mode == 'test':
        pivot_atoms = test_atoms
    inferred_atoms = detect_inferred_atoms(pivot_atoms, as_atoms)
    print(f'Number of {mode} atoms: ', len(pivot_atoms))
    print('Number of inferred atoms: ', len(inferred_atoms))
    write_things(inferred_atoms, f'WN18RR_{mode}_inferred_atoms.lp')












