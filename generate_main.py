import json
import os
import sys
from pathlib import Path
import random
import numpy as np
import time
import hydra

from omegaconf import DictConfig

# sys.path.append(os.path.abspath(".."))
from proc_bench.generators import (
        GeneratorSort, GeneratorGather, GeneratorCount, GeneratorSearch,
        GeneratorCopy, GeneratorSubstitute, GeneratorEncode,
        GeneratorBreak, GeneratorBreak2, GeneratorCompose,
        GeneratorDecompose, GeneratorRhythm, GeneratorCompare, 
        GeneratorCountv3, GeneratorDecode, GeneratorStackQueue, GeneratorRotate1dim,
        GeneratorFilling, GeneratorStrdelete, GeneratorDeleteWords, GeneratorCumulate, 
        GeneratorMove, GeneratorCyclicCheck
    )

def to_json(example):
    outs = {}
    outs["intermediate"] = []
    num_states = len(example)
    for i in range(num_states):
        if i == 0:
            outs["init"] = example[i]
        elif i == len(example) - 1:
            if isinstance(example[i], (str, int)):
                outs["final"] = example[i]
            elif isinstance(example[i][0], list):
                outs["final"] = example[i][0]
            else:
                outs["final"] = example[i]
        else:
            if isinstance(example[i], (str, int)):
                outs["intermediate"].append(example[i])
            elif isinstance(example[i][0], list):
                outs["intermediate"].extend(example[i])
            else:
                outs["intermediate"].append(example[i])
    return outs

def save_txt(sp, data):
    sp.parent.mkdir(parents=True, exist_ok=True)
    with open(sp, "w") as f:
        f.write(data)


def save_json(sp, data):
    sp.parent.mkdir(parents=True, exist_ok=True)
    with open(sp, "w") as f:
        json.dump(data, f, sort_keys=True, indent=4)


def save_task(save_dir, cnt, prompt, ex, cfg):
    sp = save_dir / "prompt" / (str(cnt).zfill(4) + ".txt")
    save_txt(sp, prompt)
    sp = save_dir / "label" / (str(cnt).zfill(4) + ".json")
    save_json(sp, to_json(ex))
    sp = save_dir / "config" / (str(cnt).zfill(4) + ".json")
    save_json(sp, cfg)

step_table = {i : 10 for i in range(2, 26)}

template_file_name = [
    "01_sort.txt", "02_gather.txt", "03_count.txt",
    "04_search.txt","05_copy.txt", "06_substitute.txt","07_encode.txt",
    "08_split1.txt","09_split2.txt","10_compose.txt","11_decompose.txt",
    "12_rhythm.txt","13_compare.txt","14_count2.txt","15_decode.txt",
    "16_pushpop.txt","17_rotate.txt","18_fill_word.txt","19_delete_char.txt",
    "20_delete_word.txt","21_cumulate.txt","22_move_cyclic.txt","23_find_cyclic.txt",
]
sentence_file = "./sentence_data/sentences.txt"

num_sentences = 43652

def gen_task01(seed, base_path, template_path):

    num_examples = 1
    gen = GeneratorSort(seed=seed)

    examples = []
    configs = []
    for key in step_table:
        question_count = step_table[key]-1
        while question_count >= 0:
            rand_length = gen.random_state.randint(5, 50)
            gen.set_config(length=rand_length, shuffle=False)
            exs, cfgs = gen.generate(num_examples)

            if len(exs[0]) != key+1:
                print("dropping step count of " + str(len(exs[0])-1) + ", mismatched steps")
                continue
            else:
                examples += exs
                configs += cfgs
                question_count -= 1
                print("step count is " + str(len(exs[0])-1))
                print("adding exs to examples")

    save_dir = base_path/"task01"
    tmpl = Path(template_path/f"{template_file_name[0]}").read_text()

    for cnt, (ex, cfg) in enumerate(zip(examples, configs)):
        txt = "\n[Question]\n"
        txt += "String: "+ ex[0]
        prompt = tmpl + txt
        save_task(save_dir, cnt, prompt, ex, cfg)


def gen_task02(seed, base_path, template_path):

    max_length = 10
    gen = GeneratorGather(seed)
    examples, configs = [], []

    for key in step_table:
        question_count = step_table[key]-1
        while question_count >= 0:
            rand_num_st = gen.random_state.randint(2, 20)
            rand_num_cmd = gen.random_state.randint(1, 30)
            gen.set_config(rand_num_st, rand_num_cmd, max_length)
            ex, cfg = gen.generate_single()
            
            if len(ex) != key+1:
                print("dropping step count of " + str(len(ex)-1) + ", mismatched steps")
                continue
            else:    
                examples.append(ex)
                configs.append(cfg)
                question_count -= 1
                print("step count is " + str(len(ex)-1))
                print("adding exs to examples")

    save_dir = base_path/"task02"
    tmpl = Path(template_path/f"{template_file_name[1]}").read_text()

    for cnt, (ex, cfg) in enumerate(zip(examples, configs)):
        strings, commands = cfg["strings"], cfg["commands"]

        txt = "\n[Question]\n"
        txt += "Strings: ["
        for i, s in enumerate(strings, 1):
            txt += f"{s}, "
        txt = txt[:-2]
        txt += "]\n"
        txt += "Sets of Numbers: ["
        for c in commands:
            txt += f"{c}, "
        txt = txt[:-2]
        txt += "]\n"

        prompt = tmpl + txt
        save_task(save_dir, cnt, prompt, ex, cfg)


def gen_task03(seed, base_path, template_path):

    num_examples = 1
    min_length, max_length = 4, 10
    gen = GeneratorCount(seed)
    examples, configs = [], []
    for key in step_table:
        question_count = step_table[key]-1
        while question_count >= 0:
            num_strings = gen.random_state.randint(1, 30)
            gen.set_config(min_length=min_length, max_length=max_length, num_strings=num_strings)

            exs, cfgs = gen.generate(num_examples)
            if key - 1 != num_strings:
                print("dropping step count of " + str(num_strings-1) + ", mismatched steps " + str(key))
                continue
            else:
                examples += exs
                configs += cfgs
                question_count -= 1
                print("step count is " + str(len(exs[0])-2))
                print("adding exs to examples")

    save_dir = base_path/"task03"
    tmpl = Path(template_path/f"{template_file_name[2]}").read_text()

    for cnt, (ex, cfg) in enumerate(zip(examples, configs)):
        strings = cfg["strings"]
        txt = "\n[Question]\n"
        txt += "Strings:\n"
        txt += str(strings)
        prompt = tmpl + txt
        save_task(save_dir, cnt, prompt, ex, cfg)


def gen_task04(seed, base_path, template_path):

    num_examples = 1
    gen = GeneratorSearch(seed)

    examples, configs = [], []
    for key in step_table:
        if key == 1:
            continue
        question_count = step_table[key]-1
        while question_count >= 0:
            #rand_num_strings = gen.random_state.randint(3, 50)
            num_strings = 10
            gen.set_config(num_strings, key) # num_substrings defines the number of steps directly
            exs, cfgs = gen.generate(num_examples)

            if len(exs[0]) != key+1:
                print("dropping step count of " + str(len(exs[0])-1) + ", mismatched steps")
                continue
            else:
                examples += exs
                configs += cfgs
                question_count -= 1
                print("step count is " + str(len(exs[0])-1))
                print("adding exs to examples")

    save_dir = base_path/"task04"
    tmpl = Path(template_path/f"{template_file_name[3]}").read_text()

    for cnt, (states, config) in enumerate(zip(examples, configs)):
        strings, substrings = config["strings"], config["substrings"]
        txt = '\n[Question]\n'
        txt += "Strings: ["
        txt += ', '.join(strings)
        txt += "]\nSubstrings: ["
        txt += ', '.join(substrings)
        txt += "]\n"
        prompt = tmpl + txt
        save_task(save_dir, cnt, prompt, states, config)


def gen_task05(seed, base_path, template_path):

    num_examples = 1
    max_length = 20
    gen = GeneratorCopy(seed)

    examples, configs = [], []
    for key in step_table:
        question_count = step_table[key]-1
        while question_count >= 0:
            rand_num_strings = gen.random_state.randint(3, 25)
            gen.set_config(rand_num_strings, key, max_length) # num_cmds defines the number of steps directly
            exs, cfgs = gen.generate(num_examples)

            examples += exs
            configs += cfgs
            question_count -= 1

    save_dir = base_path/"task05"
    tmpl = Path(template_path/f"{template_file_name[4]}").read_text()

    for cnt, (ex, cfg) in enumerate(zip(examples, configs)):
        strings, commands = cfg["strings"], cfg["commands"]

        txt = "\n[Question]\n"
        txt += "Strings: ["
        for s in strings:
            txt += f"{s}, "
        txt = txt[:-2]
        txt += "]\n"
        txt += "Indices: ["
        for c in commands:
            txt += f"{c}, "
        txt = txt[:-2]
        txt += "]\n"

        prompt = tmpl + txt
        save_task(save_dir, cnt, prompt, ex, cfg)


def gen_task06(seed, base_path, template_path):

    gen = GeneratorSubstitute(seed)

    examples, configs = [], []
    for key in step_table:
        question_count = step_table[key]
        gen.set_config(key) # length defines the number of steps directly, num_substitution is not used anymore
        exs, cfgs = gen.generate(num_examples=question_count)
        examples += exs
        configs += cfgs

    save_dir = base_path/"task06"
    tmpl = Path(template_path/f"{template_file_name[5]}").read_text()

    for cnt, (ex, cfg) in enumerate(zip(examples, configs)):
        generated_string = cfg["string"]
        substiitution_table = cfg["table"]
        txt = "\n[Question]\n"
        txt += "Pairs:"
        for char1, char2 in substiitution_table.items():
            txt += f"\n({char1}, {char2})"
        txt += "\nString:\n"
        txt += generated_string
        
        prompt = tmpl + txt
        save_task(save_dir, cnt, prompt, ex, cfg)


def gen_task07(seed, base_path, template_path):

    gen = GeneratorEncode(seed)

    examples, configs = [], []
    for key in step_table:
        gen.set_config(key) # num_div defines the number of steps directly
        exs, cfgs = gen.generate(num_examples=step_table[key])

        examples += exs
        configs += cfgs

    save_dir = base_path/"task07"
    tmpl = Path(template_path/f"{template_file_name[6]}").read_text()

    for cnt, (ex, cfg) in enumerate(zip(examples, configs)):
        string = cfg["string"]
        txt = "\n[Question]\n"
        txt += "String:\n"
        txt += string
        prompt = tmpl + txt
        save_task(save_dir, cnt, prompt, ex, cfg)


def gen_task08(seed, base_path, template_path):

    num_examples = 1
    gen = GeneratorBreak(seed)
    
    examples, configs = [], []
    for key in step_table:
        question_count = step_table[key]-1
        while question_count >= 0:
            rand_len_string = gen.random_state.randint(key+1, 30)
            gen.set_config(rand_len_string, key) # num_pos defines the number of steps directly
            exs, cfgs = gen.generate(num_examples)

            examples += exs
            configs += cfgs
            question_count -= 1

    save_dir = base_path/"task08"
    tmpl = Path(template_path/f"{template_file_name[7]}").read_text()

    for cnt, (ex, cfg) in enumerate(zip(examples, configs)):
        string, commands = cfg["string"], cfg["positions"]
        txt = "\n[Question]\n"
        txt += "String:\n"
        txt += string
        txt += "\nPositions:\n"
        for c in commands:
            txt += f"{c}, "
        txt = txt[:-2]

        prompt = tmpl + txt
        save_task(save_dir, cnt, prompt, ex, cfg)


def gen_task09(seed, base_path, template_path):

    num_examples = 1
    gen = GeneratorBreak2(seed)
    
    examples, configs = [], []
    for key in step_table:
        question_count = step_table[key]-1
        while question_count >= 0:
            rand_len_string = gen.random_state.randint(key+1, 30)
            gen.set_config(rand_len_string, key) # num_pos defines the number of steps directly
            exs, cfgs = gen.generate(num_examples)

            examples += exs
            configs += cfgs
            question_count -= 1

    save_dir = base_path/"task09"
    tmpl = Path(template_path/f"{template_file_name[8]}").read_text()

    for cnt, (ex, cfg) in enumerate(zip(examples, configs)):
        string, commands = cfg['strings'], cfg['positions']
        txt = '\n[Question]\n'
        txt += "String:\n"
        txt += string
        txt += "\nIndices:\n"
        for c in commands:
            txt += f"{c}, "
        txt = txt[:-2]

        prompt = tmpl + txt
        save_task(save_dir, cnt, prompt, ex, cfg) 


def gen_task10(seed, base_path, template_path):

    num_examples = 1
    gen = GeneratorCompose(seed=seed)
    
    examples = []
    configs = []
    for key in step_table:
        question_count = step_table[key]-1
        while question_count >= 0:
            len_init_state = gen.random_state.randint(key + 3, 30)
            gen.set_config(len_init_state, key) # num_rules defines the number of steps directly
            print(f'num_steps: {key}, len_init_state: {len_init_state}')
            exs, cfgs = gen.generate(num_examples)
            examples += exs
            configs += cfgs
            question_count -= 1

    save_dir = base_path/"task10"
    tmpl = Path(template_path/f"{template_file_name[9]}").read_text()

    for cnt, (ex, cfg) in enumerate(zip(examples, configs)):
        rules = cfg["rules"]
        txt = "\n[Question]\n"
        txt += "Rules:\n"
        for rule in rules:
            txt += rule + "\n"
        txt += "Initial state:\n"
        txt += ', '.join(ex[0])
        prompt = tmpl + txt
        save_task(save_dir, cnt, prompt, ex, cfg)


def gen_task11(seed, base_path, template_path):
    # P15 # https://github.com/SeanNobel/proc-bench/pull/103 をマージ次第に確認
    num_examples = 1
    gen = GeneratorDecompose(seed=seed)
    
    examples, configs = [], []
    state_list = []
    for key in step_table:
        question_count = step_table[key]-1
        while question_count >= 0:
            state_list.append(gen.random_state.getstate())
            rand_length = gen.random_state.randint(key + 5, key + 15)
            # num_rulesは26を超えられず、num_rulesとlengthが十分近い＆26に近い場合、
            # 全てユニークな値が出現する確率はほぼ0になるため（26!/(26^26)=1e-11)
            # 余裕を持ってrangeを決めてやる必要あり
            rand_num_rules = gen.random_state.randint(1, min(23, key))
            
            if rand_num_rules == 1:
                rand_length = 1
            
            if rand_length < rand_num_rules:
                continue
            gen.set_config(rand_length, rand_num_rules)
            exs, cfgs = gen.generate(num_examples)

            if len(exs[0]) != key+1:
                print("dropping step count of " + str(len(exs[0])-1) + ", mismatched steps. target step = " + str(key+1))
                continue
            else:
                examples += exs
                configs += cfgs
                question_count -= 1
                print("step count is " + str(len(exs[0])-1))
                print("adding exs to examples")

    save_dir = base_path/"task11"
    tmpl = Path(template_path/f"{template_file_name[10]}").read_text()

    for cnt, (ex, cfg) in enumerate(zip(examples, configs)):
        txt = "\n[Question]\n"
        txt += "characters: "
        txt += cfg["characters"]
        txt += "\nrules:"
        for key, value in cfg["rules"].items():
            ans = "".join(value)
            txt += f"\n{key} -> {ans}"
        prompt = tmpl + txt
        save_task(save_dir, cnt, prompt, ex, cfg)


def gen_task12(seed, base_path, template_path):

    gen = GeneratorRhythm(seed)

    examples, configs = [], []
    for key in step_table:
        for _ in range(step_table[key]):
            seq1_pattern = gen.random_state.randrange(4, 7)
            seq2_pattern = gen.random_state.randrange(2, 6)
            gen.set_config(seq1_pattern, seq2_pattern, 8, key)
            exs, cfgs = gen.generate_single()
            examples.append(exs)
            configs.append(cfgs)

    save_dir = base_path/"task12"
    tmpl = Path(template_path/f"{template_file_name[11]}").read_text()
    
    for cnt, (ex, cfg) in enumerate(zip(examples, configs)):
        seq1 = cfg["sequence1"]
        seq2 = cfg["sequence2"]
        N = cfg["N"]
        txt = '\n[Question]\n'
        txt += "Sequence 1: "
        txt += ', '.join([str(num) for num in seq1])
        txt += "\nSequence 2: "
        txt += ', '.join(seq2)
        txt += "\nN: "
        txt += str(N)

        prompt = tmpl + txt
        save_task(save_dir, cnt, prompt, ex, cfg) 


def gen_task13(seed, base_path, template_path):

    gen = GeneratorCompare(seed)
    states_list, config_list = [], []
    for key in step_table:
        question_count = step_table[key] - 1
        while question_count >= 0:
            len_candidate = gen.random_state.randrange(key, key+10)
            gen.set_config(key+1, len_candidate)
            sts, cfg = gen.generate_single()

            if len(sts) != key+1:
                print("dropping step count of " + str(len(sts)) + f", mismatched steps {key}")
                continue
            else:    
                states_list.append(sts)
                config_list.append(cfg)
                question_count -= 1
                print("step count is " + str(len(sts)-1))
                print("adding sts to examples")

    save_dir = base_path/"task13"
    tmpl = Path(template_path/f"{template_file_name[12]}").read_text()

    for cnt, (states, config) in enumerate(zip(states_list, config_list)):
        candidate, target = config["candidate"], config["target"]
        txt = '\n[Question]\n'
        txt += "Target: "
        txt += ''.join(target)
        txt += "\nCandidate: "
        txt += ', '.join(candidate)
        prompt = tmpl + txt

        save_task(save_dir, cnt, prompt, states, config)


def gen_task14(seed, base_path, template_path, sentence_path):

    # 1-5単語の文は abstract_sentences.txt を全文(1711329)検索すると探索可能
    gen = GeneratorCountv3(seed)
    
    examples, configs = [], []
    for key in step_table:
        gen.set_sentence_config(num_sentences, sentence_path)
        for _ in range(step_table[key]):
            gen.set_config(key)
            exs, cfgs = gen.generate_single()
            examples += exs
            configs += cfgs
            
    save_dir = base_path/"task14"
    tmpl = Path(template_path/f"{template_file_name[13]}").read_text()

    for cnt, (ex, cfg) in enumerate(zip(examples, configs)):
        sentence = cfg["sentence"]
        
        txt = "\n\n[Question]\n"
        txt += 'Sentence:\n'
        txt += sentence

        prompt = tmpl + txt
        save_task(save_dir, cnt, prompt, ex, cfg) 


def gen_task15(seed, base_path, template_path):

    gen = GeneratorDecode(seed)
    examples, configs = [], []
    for key in step_table:
        question_count = step_table[key] - 1
        while question_count >= 0:
            len_code = gen.random_state.randrange(1, 30, 1)
            gen.set_config(len_code=len_code)
            sts, cfg = gen.generate_single()
            
            if len(sts[0]) != key+1:
                print("dropping step count of " + str(len(sts[0])-1) + ", mismatched steps")
                continue
            else:
                examples += sts
                configs += cfg
                question_count -= 1
                print("step count is " + str(len(sts[0])-1))
                print("adding exs to examples")
                
    save_dir = base_path/"task15"
    tmpl = Path(template_path/f"{template_file_name[14]}").read_text()

    for cnt, (ex, cfg) in enumerate(zip(examples, configs)):
        txt = "\n[Question]\n" 
        txt += "Sentence: " + cfg["bit_sentence"] + "\n"
        prompt = tmpl + txt
        save_task(save_dir, cnt, prompt, ex, cfg)


def gen_task16(seed, base_path, template_path):
    # P21 # TODO: fixed bugs but should be checked

    gen = GeneratorStackQueue(seed)

    examples, configs = [], []
    for key in step_table:
        question_count = step_table[key]-1
        while question_count >= 0:
            len_number = gen.random_state.randrange(10, key + 15, 5)
            len_number = gen.np_random_state.randint(1, len_number)
            gen.set_config(len_number, key)
            sts, cfg = gen.generate_single()

            if len(sts[0]) != key:
                print("dropping step count of " + str(len(sts[0])-1) + ", mismatched steps")
                continue
            else:
                examples += [sts]
                configs += [cfg]
                question_count -= 1
                print("step count is " + str(len(sts[0])-1))
                print("adding exs to examples")

    save_dir = base_path/"task16"
    tmpl = Path(template_path/f"{template_file_name[15]}").read_text()

    for cnt, (states, config) in enumerate(zip(examples, configs)):
        initial, actions = config["initial"], config["actions"]
        txt = '\n[Question]\n'
        txt += "Initial string: "
        txt += str(initial)
        txt += "\nActions: "
        txt += ', '.join(actions)
        prompt = tmpl + txt

        save_task(save_dir, cnt, prompt, states, config)


def gen_task17(seed, base_path, template_path):

    # パラメータで自動決定される
    gen = GeneratorRotate1dim(seed)

    examples, configs = [], []
    for key in step_table:
        for _ in range(step_table[key]):
            # for length in range(20, 50, 5):
            length = gen.random_state.randrange(5, 15)
            gen.set_config(length, key)
            exs, cfgs = gen.generate(num_examples=1)
            examples += exs
            configs += cfgs

    save_dir = base_path/"task17"
    tmpl = Path(template_path/f"{template_file_name[16]}").read_text()

    for cnt, (ex, cfg) in enumerate(zip(examples, configs)):
        string = cfg["string"]
        combination = cfg['combinations']
        combination = [[p[0]-1, p[1]] for p in combination]
        
        txt = "\n[Question]\n"
        txt += 'String:\n'
        txt += string
        txt += '\nIndices:\n'
        for comb in combination:
            txt += f'({comb[0]}, {comb[1]}), '
        txt = txt[:-2]

        prompt = tmpl + txt
        save_task(save_dir, cnt, prompt, ex, cfg) 


def gen_task18(seed, base_path, template_path, sentence_path):

    gen = GeneratorFilling(seed)
    num_pairs_limit = 5
    iter = 6

    examples, configs = [], []
    gen.set_sentence_config(num_sentences, sentence_path)
    for key in step_table:
        for _ in range(step_table[key]):
            gen.set_config(key)
            exs, cfgs = gen.generate(num_examples=1)
            examples += exs
            configs += cfgs

    save_dir = base_path/"task18"
    tmpl = Path(template_path/f"{template_file_name[17]}").read_text()

    for cnt, (ex, cfg) in enumerate(zip(examples, configs)):
        sentence = cfg["sentence"]
        list = cfg['list']

        txt = "\n[Question]\n"
        txt += 'Sentence:\n'
        txt += sentence
        txt += '\nList:\n'
        txt += '['+ ', '.join([f"({pair[0]}, {pair[1]})" for pair in list]) + ']'

        prompt = tmpl + txt
        save_task(save_dir, cnt, prompt, ex, cfg) 


def gen_task19(seed, base_path, template_path):

    gen = GeneratorStrdelete(seed)
    examples, configs = [], []

    for key in step_table:
        for _ in range(step_table[key]):
            len_string = gen.random_state.randint(key+1, 30)
            gen.set_config(len_string, key)
            sts, cfg = gen.generate_single()
            examples.append(sts)
            configs.append(cfg)

    save_dir = base_path/"task19"
    tmpl = Path(template_path/f"{template_file_name[18]}").read_text()

    for cnt, (ex, cfg) in enumerate(zip(examples, configs)):
        txt = "\n[Question]\n" 
        txt += "String: " + cfg["string"] + "\n"
        txt += "Steps: \n" 
        txt += "\n".join([f"{i+1}. {s}" for i, s in enumerate(cfg["steps"])])
        prompt = tmpl + txt
        save_task(save_dir, cnt, prompt, ex, cfg)


def gen_task20(seed, base_path, template_path, sentence_path):

    num_examples = 1
    gen = GeneratorDeleteWords(seed)
    gen.set_sentence_config(num_sentences, sentence_path)

    examples, configs = [], []
    th_num_words = 5 
    for key in step_table:
        for _ in range(step_table[key]):
            gen.set_config(th_num_words, key)
            exs, cfgs = gen.generate(num_examples=num_examples)
            examples += exs
            configs += cfgs

    save_dir = base_path/"task20"
    tmpl = Path(template_path/f"{template_file_name[19]}").read_text()

    for cnt, (ex, cfg) in enumerate(zip(examples, configs)):
        txt = "\n[Question]\n"
        txt += "Sentence:\n"
        txt += cfg["sentence"] + "\n"
        txt += "Words: ["
        for word in cfg["words"]:
            txt += word + ", "
        txt = txt[:-2]
        txt += "]\n"

        prompt = tmpl + txt
        save_task(save_dir, cnt, prompt, ex, cfg) 


def gen_task21(seed, base_path, template_path):

    num_examples = 1
    gen = GeneratorCumulate(seed=seed)

    examples = []
    configs = []
    for key in step_table:
        for _ in range(step_table[key]):
            gen.set_config(key)
            exs, cfgs = gen.generate(num_examples)
            examples += exs
            configs += cfgs

    save_dir = base_path/"task21"
    tmpl = Path(template_path/f"{template_file_name[20]}").read_text()

    for cnt, (ex, cfg) in enumerate(zip(examples, configs)):
        txt = "\n[Question]\n"
        txt += "N = " + str(ex[0]) + "\n"
        txt += "Operations: ["
        for operation in cfg["operations"]:
            txt += operation + ", "
        txt = txt[:-2]
        txt += "]\n"
        prompt = tmpl + txt
        save_task(save_dir, cnt, prompt, ex, cfg)


def gen_task22(seed, base_path, template_path):

    num_examples = 1
    gen = GeneratorMove(seed=seed, num_dims=1, cyclic=True)

    examples = []
    configs = []
    for key in step_table:
        for i in range(1, step_table[key] + 1):
            gen.set_config(num_operations=key, width=5 * i)
            exs, cfgs = gen.generate(num_examples)
            examples += exs
            configs += cfgs

    save_dir = base_path/"task22"
    tmpl = Path(template_path/f"{template_file_name[21]}").read_text()

    for cnt, (ex, cfg) in enumerate(zip(examples, configs)):
        txt = "\n[Question]\n"
        txt += "Initial array:\n" + ex[0] + "\n"
        txt += "Operations: ["
        for direction, distance in cfg["operations"]:
            txt += f"\n ({direction}, {distance}),"
        txt = txt[:-1] + "\n]\n"
        prompt = tmpl + txt
        save_task(save_dir, cnt, prompt, ex, cfg)


def gen_task23(seed, base_path, template_path):
    num_examples = 1
    gen = GeneratorCyclicCheck(seed=seed)

    examples = []
    configs = []
    for key in step_table:
        question_count = step_table[key] - 1
        while question_count >= 0:
            length = gen.random_state.randrange(1, key+5)
            gen.set_config(length=length)
            exs, cfgs = gen.generate_single()
            if len(exs[0]) != key+1:
                print("dropping step count of " + str(len(exs[0])-1) + ", mismatched steps")
                continue
            else:    
                examples += exs
                configs += cfgs
                # examples.append(exs)
                # configs.append(cfgs)
                question_count -= 1
                print("step count is " + str(len(exs[0])-1))
                print("adding exs to examples")
    
    save_dir = base_path/"task23"
    tmpl = Path(template_path/f"{template_file_name[22]}").read_text()

    for cnt, (ex, cfg) in enumerate(zip(examples, configs)):
        txt = "\n[Question]\n"
        txt += "String: " + cfg["string"] + "\n"
        txt += "Command: " + str(cfg["command"]) + "\n"
        prompt = tmpl + txt
        save_task(save_dir, cnt, prompt, ex, cfg)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    
    base_path = Path(f"experiment/{args.exp_name}/dataset")
    template_path = Path("./templates")

    seed = args.seed
    print(f"seed: {seed}")
    time.sleep(1)
    gen_task01(seed, base_path, template_path)
    gen_task02(seed, base_path, template_path)
    gen_task03(seed, base_path, template_path)
    gen_task04(seed, base_path, template_path)    
    gen_task05(seed, base_path, template_path)
    gen_task06(seed, base_path, template_path)    
    gen_task07(seed, base_path, template_path)    
    gen_task08(seed, base_path, template_path)    
    gen_task09(seed, base_path, template_path)    
    gen_task10(seed, base_path, template_path)    
    gen_task11(seed, base_path, template_path)    
    gen_task12(seed, base_path, template_path)    
    gen_task13(seed, base_path, template_path)    
    gen_task14(seed, base_path, template_path, sentence_file)    
    gen_task15(seed, base_path, template_path)    
    gen_task16(seed, base_path, template_path)    
    gen_task17(seed, base_path, template_path)    
    gen_task18(seed, base_path, template_path, sentence_file)    
    gen_task19(seed, base_path, template_path)    
    gen_task20(seed, base_path, template_path, sentence_file)    
    gen_task21(seed, base_path, template_path)    
    gen_task22(seed, base_path, template_path)    
    gen_task23(seed, base_path, template_path)

if __name__ == "__main__":
    run()
