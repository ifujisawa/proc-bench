import numpy as np
import re
from typing import List, Callable, Union

from proc_bench.constants import ALLOWED_DELIMITERS


def loop_for_delimiters(evaluator: Callable, pred: str, label: List[List[str]]):
    outs = []
    for delimiter in ALLOWED_DELIMITERS:
        _label = [delimiter.join(l) for l in label]
        
        outs.append(evaluator(pred, _label))
        
    return max(outs)


def find_all_occurrences(s: str, substr: str) -> List[int]:
    return [i for i in range(len(s)) if s.startswith(substr, i)]


def find_last_occurences(pred: str, label: List[str]) -> List[int]:
    indices = []
    for l in reversed(label): # find from the last
        all_occurs = find_all_occurrences(pred, l)
        found = False
        for index in reversed(all_occurs):
            if index not in indices:
                indices.append(index)
                found = True
                break

        if not found:
            indices.append(-1)

    return list(reversed(indices))


def is_sorted(a: Union[List[int], np.ndarray]) -> bool:
    return np.all(np.array(a[:-1]) <= np.array(a[1:]))


def str_to_list(s: str) -> List[str]:
    """Searches [ and ] in the string from the end, delimits the substring by commas."""
    start, end = s.rfind("["), s.rfind("]")
    if start >= end or start == -1 or end == -1:
        return
    
    s = s[start + 1:end]
    
    return [_s.strip('" ') for _s in s.split(",")]


def str_to_dict(s: str) -> dict[str, List[str]]:
    """Searches { and } in the string from the end, converts the intermediate to dictionary."""
    start, end = s.rfind("{"), s.rfind("}")
    if start >= end or start == -1 or end == -1:
        return dict()

    ret = dict()
    str_list = s[start+1:end].split(":")
    key = re.sub(r"[^a-zA-Z]", "", str_list[0]) # key
    for sub in str_list[1:-1]: # item, key
        sp = sub.rfind(",")
        item = str_to_list(sub[:sp])
        ret[key] = ["0" if it == "" else it for it in item]
        key = re.sub(r"[^a-zA-Z]", "", sub[sp:])
    item = str_to_list(str_list[-1]) # item
    ret[key] = ["0" if it == "" else it for it in item]
    return ret


def find_all_dicts(s: str) -> List[dict]:
    dicts = []
    last = len(s)
    while last >= 0:
        found_dict = str_to_dict(s[:last])
        if len(found_dict.keys()) == 0:
            break
        dicts.append(found_dict)
        last = s[:last].rfind("{")

    return list(reversed(dicts))


def is_dict_matched(pred_dict: dict, label_dict: dict) -> bool:
    total = 0
    for key in label_dict:
        if key in pred_dict:
            if isinstance(pred_dict[key], list) and len(pred_dict[key]) == len(label_dict[key]):
                total += np.all([str(pred_dict[key][i]) == str(label_dict[key][i]) for i in range(len(label_dict[key]))])
    return int(total / len(label_dict.keys())) == 1


def eval_matched_dict(pred_dicts: List[dict], label_dicts: List[dict], require_sorted=True) -> List[bool]:
    last_pred_candidate = len(pred_dicts) - 1
    matched_list = []
    for label_dict in reversed(label_dicts):
        matched_any = False
        for i in range(last_pred_candidate, -1, -1):
            pred_dict = pred_dicts[i]
            matched = is_dict_matched(pred_dict, label_dict)
            matched_any |= matched
            if matched:
                if require_sorted: last_pred_canditate = i - 1
                break
        matched_list.append(matched_any)

    return matched_list