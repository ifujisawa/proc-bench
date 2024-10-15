import numpy as np

def match_int(y_true, y_pred):
    y_true_int = int(y_true)
    try:
        y_pred_int = int(y_pred)
    except:
        return 0.0

    return float(y_true_int == y_pred_int)

def match_str(y_true, y_pred):
    y_true_str = str(y_true)
    try:
        y_pred_str = str(y_pred)
    except:
        return 0.0

    return float(y_true_str == y_pred_str)

def match_list(seq_true, seq_pred, match=match_str):
    assert isinstance(seq_true, list)

    if len(seq_true) == 0:
        return 0.0

    try:
        seq_pred == list(seq_pred)
    except:
        return 0.0

    if len(seq_true) != len(seq_pred):
        return 0.0
        
    arr = np.array([match(st, sp) for st, sp in zip(seq_true, seq_pred)])
    return float(arr.all())


def reduce_last_intermediate(y_pred):
    y_pred_ret = y_pred.copy()
    if len(y_pred['intermediate']) == 0:
        return y_pred_ret
    if exact_match(y_pred['final'], y_pred['intermediate'][-1]):
        y_pred_ret['intermediate'] = y_pred['intermediate'][:-1]
    return y_pred_ret

def exact_match(y_true, y_pred):
    type_y_true = type(y_true)
    if type_y_true == str:
        return match_str(y_true, y_pred)
    elif type_y_true == int:
        return match_int(y_true, y_pred)
    elif type_y_true == list:
        return match_list(y_true, y_pred)
    else:
        raise TypeError(f'type of ground truth is {type_y_true} and not supported')

def match_hit(seq_true, seq_pred):
    assert type(seq_true) == list
    if len(seq_true) == 0:
        return np.nan

    hit = 0.0
    for st, sp in zip(seq_true, seq_pred):
        hit += exact_match(st, sp)
    return hit

def prefix_match_length(seq_true, seq_pred):
    assert type(seq_true) == list
    if len(seq_true) == 0:
        return 0.0

    hit = 0.0
    for st, sp in zip(seq_true, seq_pred):
        if exact_match(st, sp):
            hit += 1
        else:
            break
    return hit
    
def match_ratio(seq_true, seq_pred):
    assert type(seq_true) == list
    len_true = len(seq_true)
    len_pred = len(seq_pred)
    len_max = max(len_true, len_pred)
    if len_true == 0:
        return 0.0

    hit = 0.0
    for st, sp in zip(seq_true, seq_pred):
        hit += exact_match(st, sp)
    return hit / len_max

def prefix_accuracy(seq_true, seq_pred):
    assert type(seq_true) == list
    len_true = len(seq_true)
    len_pred = len(seq_pred)
    len_max = max(len_true, len_pred)
    if len_true == 0:
        return 0.0

    hit = 0.0
    for st, sp in zip(seq_true, seq_pred):
        if exact_match(st, sp):
            hit += 1
        else:
            break
    return hit / len_max


if __name__ == '__main__':
    assert match_int(5, '5') == 1.0
    assert match_int(32, '32') == 1.0
    assert match_int('512', 512) == 1.0
    assert match_int(234, (234)) == 1.0

    assert match_int(5, '55') == 0.0
    assert match_int(512, 'a') == 0.0
    assert match_int('512', 'a') == 0.0
    assert match_int(231, [231]) == 0.0
    assert match_int(231, [231, 66]) == 0.0
    assert match_int(65, None) == 0.0
    assert match_int(73, []) == 0.0


    assert match_str('abc', 'abc') == 1.0
    assert match_str('3', '3') == 1.0
    assert match_str('21', 21) == 1.0
    assert match_str('fe a t', 'fe a t') == 1.0

    assert match_str('abc', 'ac') == 0.0
    assert match_str('fe a t', 'feat') == 0.0
    assert match_str('abc', ['abc']) == 0.0
    assert match_str('abc', None) == 0.0
    assert match_str('abc', []) == 0.0


    assert match_list([1], [1]) == 1.0
    assert match_list([1, 3, 5], [1, 3, 5]) == 1.0
    assert match_list(['abc'], ['abc']) == 1.0
    assert match_list(['abc', 'de'], ['abc', 'de']) == 1.0

    assert match_list([], []) == 0.0
    assert match_list([], [1]) == 0.0
    assert match_list([2], [3]) == 0.0
    assert match_list([1], [1, 1]) == 0.0
    assert match_list([1], 1) == 0.0
    assert match_list(['abc', 'de', 'f'], ['abc', 'de']) == 0.0
    assert match_list(['abc', 'de'], ['abc', 'de', 'f']) == 0.0
    assert match_list(['abc', 'de'], ['de', 'abc']) == 0.0
    assert match_list([1], []) == 0.0
    assert match_list([1], None) == 0.0
    assert match_list(['abc', 'de'], [None, 'de']) == 0.0

    print('All tests passed!')