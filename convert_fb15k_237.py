
import pandas as pd


def load_fb15k_237(train_path, valid_path, test_path):
    def _read(path):
        _data = pd.read_csv(path, sep='\t', dtype=str, header=None)
        _data.columns = ['head', 'relation', 'tail']
        _list_ents = set(_data['head'].tolist() + _data['tail'].tolist())
        _list_rels = set(_data['relation'].tolist())
        return _data, _list_ents, _list_rels

    def _apply_ents(_data, _list_ents):
        _data['head'] = _data['head'].apply(lambda x: 'e' + str(_list_ents.index(x)))
        _data['tail'] = _data['tail'].apply(lambda x: 'e' + str(_list_ents.index(x)))
        return _data

    def _apply_rels(_data, _list_rels):
        _data['relation'] = _data['relation'].apply(lambda x: 'r' + str(_list_rels.index(x)))
        # _data['inv_relation'] = _data['relation'].apply(lambda x: 'inv_' + x)
        return _data

    data_train, list_ents_train, list_rels_train = _read(train_path)
    data_valid, list_ents_valid, list_rels_valid = _read(valid_path)
    data_test, list_ents_test, list_rels_test = _read(test_path)

    list_ents = list(list_ents_train | list_ents_valid | list_ents_test)
    list_rels = list(list_rels_train | list_rels_valid | list_rels_test)

    data_train = _apply_ents(data_train, list_ents)
    data_train = _apply_rels(data_train, list_rels)

    data_valid = _apply_ents(data_valid, list_ents)
    data_valid = _apply_rels(data_valid, list_rels)

    data_test = _apply_ents(data_test, list_ents)
    data_test = _apply_rels(data_test, list_rels)

    return data_train, data_valid, data_test, list_ents, list_rels


if __name__ == '__main__':
    df_train, df_valid, df_test, list_ents, list_rels = load_fb15k_237('FB15k-237/train.txt',
                                                                       'FB15k-237/valid.txt',
                                                                       'FB15k-237/test.txt')

    df_train.to_csv('FB15k-237/train_encoded.txt', header=False, sep='\t', index=False)
    df_valid.to_csv('FB15k-237/valid_encoded.txt', header=False, sep='\t', index=False)
    df_test.to_csv('FB15k-237/test_encoded.txt', header=False, sep='\t', index=False)

    with open('FB15k-237/list_ents.txt', 'w') as f:
        f.writelines([e + '\n' for e in list_ents])
    with open('FB15k-237/list_rels.txt', 'w') as f:
        f.writelines([e + '\n' for e in list_rels])



















