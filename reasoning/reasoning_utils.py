
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

