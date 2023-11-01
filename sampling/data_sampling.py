import argparse

import pandas as pd


def sample_dataset(path, k=3000):
    df = pd.read_csv(path, sep='\t', dtype=str, header=None)
    selected_rows = df.sample(n=k)
    sample = pd.DataFrame(selected_rows)
    df = df.drop(selected_rows.index)
    return df, sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", help="Number of samples", required=True)
    args = parser.parse_args()

    path = '../WN18RR/train.txt'
    remain_path = '../WN18RR/train_sampled.txt'
    sample_path = '../WN18RR/test_sampled.txt'
    remain, sample = sample_dataset(path)
    remain.to_csv(remain_path, sep='\t', index=False, header=False)
    sample.to_csv(sample_path, sep='\t', index=False, header=False)


