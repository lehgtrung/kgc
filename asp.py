import pandas as pd
from main import load_data, load_dict


def convert_rules_to_asp(rules):
    # each rule has this form: "_hypernym _instance_hypernym _has_part"
    lines = []
    for rule in rules:
        rule = rule.split(' ')
        line = [f'{rule[0]}(X,Z) ', ':- ']
        i = 1
        for atom in rule[1:-1]:
            line.append(f'{atom}(Y{i},Y{i+1}), ')
            i += 1
        line.append(f'{rule[-1]}(Y{i},Z).')
        lines.append(''.join(line))
    return lines


def convert_data_to_asp(df:pd.DataFrame):
    lines = []
    for i, row in df.iterrows():
        head, rel, tail = row['head'], row['relation'], row['tail']
        head = f'E{head}'
        tail = f'E{tail}'
        lines.append(f'{rel}({head},{tail})')
        lines.append(f'!{rel}({tail},{head})')
    return lines


if __name__ == '__main__':
    rules = ['AAA BBB CCC', 'DDD EEE FFF GGG']
    print(convert_rules_to_asp(rules))

    dataset = 'WN18RR'
    ent_dct = load_dict(f'{dataset}/entities.dict')
    train_data = load_data(f'{dataset}/train.txt', ent_dct, apply_dct=False)

    for l in convert_data_to_asp(train_data)[:100]:
        print(l)


