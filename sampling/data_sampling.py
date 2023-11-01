
import pandas as pd
from data_importing import load_data


def sample_dataset(path, k=3000):
    df = load_data(path)
    selected_rows = df.sample(n=k)
    sample = pd.DataFrame(selected_rows)
    df = df.drop(selected_rows.index)
    return df, sample


if __name__ == '__main__':
    path = '../WN18RR/train.txt'
    remain_path = '../WN18RR/train_sampled.txt'
    sample_path = '../WN18RR/test_sampled.txt'
    remain, sample = sample_dataset(path)
    remain.to_csv(remain_path, sep='\t', index=False, header=False)
    sample.to_csv(sample_path, sep='\t', index=False, header=False)


