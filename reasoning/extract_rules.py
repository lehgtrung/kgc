

def normalize_relation(x: str):
    inv_flag = False
    if x.startswith('!'):
        inv_flag = True
        x = x.lstrip('!')
    x = x.lstrip('_')
    if inv_flag:
        return 'inv_' + x
    return x


def form_rule(_rule):
    result = f'{_rule[0]}(X,Y) :- {_rule[1]}(X,Z1)'
    k = 1
    for each in _rule[2:-1]:
        result += f',{each}(Z{k},Z{k+1})'
        k += 1
    result += f',{_rule[-1]}(Z{k},Y).'
    return result


def extract_high_conf_rules(in_path, out_path):
    rules = []
    with open(in_path, 'r') as f:
        lines = [e.strip() for e in f.readlines()]
    for line in lines:
        line = line.split()
        rule = [normalize_relation(e) for e in line[:-2]]
        rule_as_txt = form_rule(rule)
        if float(line[-1]) >= 0.1:
            rules.append(rule_as_txt)
    with open(out_path, 'w') as f:
        f.writelines([e + '\n' for e in rules])


if __name__ == '__main__':
    extract_high_conf_rules('../WN18RR/patterns_mxl_3.txt',
                            'WN18RR_rules.txt')










