import json
from proc_bench.evaluator import prefix_accuracy, exact_match

if __name__ == '__main__':
    path = './dataset/combined_dataset.json'
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]

    fm_list, sm_list, pa_list = [], [], []
    for item in data:
        gt = item['label']
        states_true = gt['intermediate'] + [gt['final']]
        ## dummy
        states_pred = gt['intermediate'] + [gt['final']]

        pa = prefix_accuracy(states_true, states_pred)
        sm = pa == 1.0
        fm = exact_match(states_true[-1], states_pred[-1])
        pa_list.append(pa)
        sm_list.append(sm)
        fm_list.append(fm)
    
print(f'prefix accuracy: {sum(pa_list) / len(pa_list)}')
print(f'final match: {sum(fm_list) / len(fm_list)}')
print(f'sequential match: {sum(sm_list) / len(sm_list)}')