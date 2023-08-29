
import pandas as pd


def normalize_relation(x: str):
    inv_flag = False
    if x.startswith('!'):
        inv_flag = True
        x = x.lstrip('!')
    x = x.lstrip('_')
    splits = x.split('_')
    if inv_flag:
        return 'inv' + ''.join([e for e in splits])
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

