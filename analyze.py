import pandas as pd
from tqdm import tqdm


def load_data(path):
    data = pd.read_csv(path, sep='\t')
    data.columns = ['head', 'relation', 'tail']
    return data


def check_overlap(df_train, df_test):
    train_heads = df_train['head'].unique().tolist()
    train_tails = df_train['tail'].unique().tolist()
    train = set(train_heads + train_tails)

    test_heads = df_test['head'].unique().tolist()
    test_tails = df_test['tail'].unique().tolist()
    test = set(test_heads + test_tails)

    print('Test inside train? = ', test.issubset(train))
    print('Num train = ', len(train))
    print('Num test = ', len(test))
    print('Test - train = ', len(test - train))


if __name__ == '__main__':
    wn18rr_train = load_data('WN18RR/train.txt')
    wn18rr_test = load_data('WN18RR/test.txt')
    check_overlap(wn18rr_train, wn18rr_test)

    fb15k_train = load_data('FB15k-237/train.txt')
    fb15k_test = load_data('FB15k-237/test.txt')
    check_overlap(fb15k_train, fb15k_test)

