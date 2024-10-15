import random
import re
import string
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import copy
import sys

import numpy as np


def generate_string(
    length,
    use_digits=False,
    duplicates=True,
    custom_chars=None,
    alpha_ratio=0.5,
    random_state=None,
):
    if random_state is not None:
        random_state = random_state
    else:
        random_state = random.Random()

    # only alphabet as default ( + digits)
    chars = string.ascii_lowercase
    if use_digits:
        chars += string.digits
    else:
        alpha_ratio = 1.0

    # custom character set
    if custom_chars:
        chars = custom_chars

    is_alpha = np.array([c.isalpha() for c in chars])
    is_digit = np.array([c.isdigit() for c in chars])
    weights = alpha_ratio * is_alpha + (1 - alpha_ratio) * is_digit

    if duplicates:
        result = "".join(random_state.choices(chars, weights=weights, k=length))
    else:
        if length > len(chars):
            raise ValueError(
                "The specified length cannot be achieved without duplicates in the generated string."
            )
        result = "".join(random_state.sample(chars, length))

    return result


def generate_sentence(N, sentence_file_path, random_state=None):
    if random_state is not None:
        random_state = random_state
    else:
        random_state = random.Random()
    
    with open(sentence_file_path, 'r') as file:
        lines = file.readlines()
    sentences = [line.strip() for line in lines]
    generated_sentences = random_state.sample(sentences, N)

    return generated_sentences


class BaseGenerator(ABC):

    def __init__(self, seed=None):
        self.seed = seed
        self.random_state = random.Random(seed)
        self.np_random_state = np.random.RandomState(seed)

    @abstractmethod
    def generate_single(self):
        pass

    @abstractmethod
    def set_config(self):
        pass

    def generate(self, num_examples=1):
        examples, configs = [], []
        for _ in range(num_examples):
            ex, cfg = self.generate_single()
            examples.append(ex)
            configs.append(cfg)
        return examples, configs


class GeneratorSort(BaseGenerator):

    def __init__(self, seed=None):
        super().__init__(seed)

    def set_config(self, length, shuffle):
        self.length = length
        self.alphabet = string.ascii_lowercase
        self.shuffle = shuffle
        self.config = {
            "length": length,
            "shuffle": shuffle,
            "alphabet": "".join(self.alphabet),
        }

    def shuffle_alphabet(self):
        self.alphabet = list(self.alphabet)
        self.random_state.shuffle(self.alphabet)
        self.alphabet = "".join(self.alphabet)
        self.config["alphabet"] = self.alphabet

    def exchange(self, string, i, j):
        string_list = list(string)
        dummy = string_list[i]
        string_list[i] = string_list[j]
        string_list[j] = dummy
        return "".join(string_list)

    def generate_single(self):
        string = generate_string(
            self.length,
            use_digits=False,
            duplicates=True,
            random_state=self.random_state,
        )
        if self.shuffle:
            self.shuffle_alphabet()
        focus_num = 0
        outs = []
        tmp_string = string
        outs.append(string)
        for set_char in self.alphabet:
            for i in range(focus_num, len(string)):
                if set_char == string[i]:
                    string = self.exchange(string, i, focus_num)
                    if tmp_string != string:
                        tmp_string = string
                        outs.append(string)
                    focus_num += 1
        return outs, copy(self.config)


class GeneratorGather(BaseGenerator):

    def __init__(self, seed=None):
        super().__init__(seed)

    def set_config(self, num_strings, num_commands, max_length, min_length=5):
        self.num_strings = num_strings
        self.num_commands = num_commands
        self.min_length = min_length
        self.max_length = max_length
        self.config = {}
        self.config["num_strings"] = num_strings
        self.config["num_commands"] = num_commands
        self.config["max_length"] = max_length
        self.config["min_length"] = min_length

    def gather(self, strings, commands):
        extracted_parts = []
        gathered_strings = [""]
        for cmd in commands:
            N, i, n = cmd
            extracted_part = strings[N][i : i + n]
            extracted_parts.append(extracted_part)
            gathered_strings.append(gathered_strings[-1] + extracted_part)
        #return extracted_parts + ["".join(extracted_parts)]
        return gathered_strings

    def generate_single(self):
        len_strings = [
            self.random_state.randint(self.min_length, self.max_length)
            for _ in range(self.num_strings)
        ]
        strings = [
            generate_string(
                len_s, use_digits=True, duplicates=True, random_state=self.random_state
            )
            for len_s in len_strings
        ]

        commands = []
        for i in range(self.num_commands):
            rand_num = self.random_state.randint(0, self.num_strings - 1)
            rand_ini = self.random_state.randint(0, len_strings[rand_num] - 1)
            rand_fin = self.random_state.randint(
                1, len_strings[rand_num] - rand_ini
            )
            commands.append((rand_num, rand_ini, rand_fin))

        self.config["strings"] = strings
        self.config["commands"] = commands

        states = self.gather(strings, commands)

        return states, copy(self.config)


class GeneratorDistribute(BaseGenerator):

    def __init__(self, seed=None):
        super().__init__(seed)

    def set_config(self, num_people, num_items, num_steps):
        self.num_people = num_people
        self.num_items = num_items
        self.num_steps = num_steps
        self.config = {
            "num_people": num_people,
            "num_items": num_items,
            "num_steps": num_steps
        }

    # ランダムな大文字1文字を生成する関数
    # def random_uppercase_letter(self):
    #     return self.random_state.choice(string.ascii_uppercase)
    
    # ランダムな小文字2-10文字を生成する関数
    def random_lowercase_string(self):
        length = self.random_state.randint(2, 10)
        return ''.join(self.random_state.choice(string.ascii_lowercase) for _ in range(length))

    def generate_random_matrix(self):
        # 0で初期化された行列を作成
        matrix = np.zeros((self.num_people, self.num_items), dtype=int)

        # 全体で3つの位置を選び、0以外の値を配置  1ステップにつき、3回の配分があると理解しました。
        total_elements = self.num_people * self.num_items
        non_zero_indices = self.random_state.choices(range(total_elements), k=3)
        values = [self.random_state.randint(1, 3) for _ in range(3)] # 最高でも1つのアイテムにつき、1ステップで一人当たり3つまでの配分
    
        for index, value in zip(non_zero_indices, values):
            row, col = divmod(index, self.num_items)
            matrix[row, col] = value
    
        return matrix.tolist()

    def generate_multiple_matrices(self, num_matrices):
        matrices = []
        for _ in range(num_matrices):
            matrices.append(self.generate_random_matrix())
        return matrices
    
    def number_to_words(self, number):
        words = {1: "one", 2: "two", 3: "three"}
        return words.get(number, str(number))
    
    def generate_distribution_sentence(self, matrix, people_labels, items_labels):
        sentences = []
    
        for i, person in enumerate(people_labels):
            for j, count in enumerate(matrix[i]):
                if count > 0:
                    item = items_labels[j]
                    count_str = f"{self.number_to_words(count)} {item}"
                    sentences.append(f"{count_str} to {person}")
    
        if len(sentences) == 1:
            return f"Distribute {sentences[0]}."
        else:
            return f"Distribute {', '.join(sentences[:-1])}, and {sentences[-1]}."

    # 分配がどのように行われたかを計算する
    def distribute_items(self, states, matrix):
        # 集計用の行列を初期化
        aggregated_counts = [[0] * self.num_items for _ in range(self.num_people)]
    
        # 各行列を処理
        for i in range(self.num_people):
            for j in range(self.num_items):
                aggregated_counts[i][j] = states[i][j] + matrix[i][j]
    
        return aggregated_counts

    def form_belonging_lists(self, item_counts, people, items):
        belonging_lists = dict()
        for person, belonging_list in zip(people, item_counts):
            belonging_lists[person] = belonging_list

        return belonging_lists

    def generate_single(self):
        # num_peopleの数だけ大文字1文字を生成
        # people = [self.random_uppercase_letter() for _ in range(self.num_people)]
        people = self.random_state.sample(string.ascii_uppercase, self.num_people)
        self.config["Names of people"] = people
    
        # num_itemsの数だけ小文字2-5文字を生成
        items = [self.random_lowercase_string() for _ in range(self.num_items)]
        self.config["Items"] = items
        
        # matrixを生成
        distributing_mat = self.generate_multiple_matrices(self.num_steps)
        self.config["Steps"] = [self.generate_distribution_sentence(mat, people, items) for mat in distributing_mat]
        #states = [[[0] * self.num_items for _ in range(self.num_people)]]
        item_counts = [[0] * self.num_items for _ in range(self.num_people)]
        states = []
        states.append(self.form_belonging_lists(item_counts, people, items))
        for i in range(len(distributing_mat)):
            #states.append(self.distribute_items(states[-1], distributing_mat[i]))
            item_counts = self.distribute_items(item_counts, distributing_mat[i])
            states.append(self.form_belonging_lists(item_counts, people, items))
        
        return states, copy(self.config)


class GeneratorCount(BaseGenerator):

    def __init__(self, seed=None):
        super().__init__(seed)

    def set_config(self, min_length, max_length, num_strings):
        self.min_length = min_length
        self.max_length = max_length
        self.num_strings = num_strings
        self.config = {}
        self.config["min_length"] = min_length
        self.config["max_length"] = max_length
        self.config["num_strings"] = num_strings

    def count(self, substrings):
        outputs = []
        answer = 0
        for c in substrings:
            N1 = sum(c_.isalpha() for c_ in c)
            numeric_chars = ''.join(c_ for c_ in c if c_.isdigit())
            N2 = int(numeric_chars) if numeric_chars else 0
            N3 = N1 * N2
            outputs.append([c, str(N1), str(N2), str(N3)])
            answer += N3

        return ['0'] + outputs + [answer]

    def generate_single(self):
        lengths = [rand_len for rand_len
                    in self.np_random_state.randint(self.min_length, self.max_length, self.num_strings)]
        strings = [generate_string(l, use_digits=True, alpha_ratio=0.75, random_state=self.random_state) for l in lengths]
        
        self.config["strings"] = copy(strings)
        states = self.count(strings)
        
        return states, copy(self.config)


class GeneratorSearch(BaseGenerator):

    def __init__(self, seed=None):
        super().__init__(seed)

    def set_config(self, num_strings, num_substrings):
        self.num_strings = num_strings
        self.num_substrings = num_substrings
        self.config = {
            "num_strings": num_strings,
            "num_substrings": num_substrings
        }

    def generate_strs_included_substrs(self, random_strings):
        custom_strings = []
        lengths = self.np_random_state.randint(3, 11, self.num_strings) # 3から10の長さの文字列を生成
        num_inclusions = self.np_random_state.randint(0, 3, self.num_strings) # 0から2個のrandom_strings内の文字を含める

        for i in range(self.num_strings):
            chosen_chars = ''.join(self.random_state.choices(random_strings, k=num_inclusions[i]))  # random_strings内の文字を選択
            remaining_length = lengths[i] - num_inclusions[i]
            random_chars = ''.join(self.random_state.choices(string.ascii_lowercase, k=remaining_length))  # 残りをランダムな小文字で埋める
            insert_position = self.random_state.randint(0, remaining_length)
            final_string = random_chars[:insert_position] + chosen_chars + random_chars[insert_position:]
            custom_strings.append(final_string)
        return custom_strings

    def generate_single(self):
        substrings = [generate_string(2, use_digits=False, random_state=self.random_state) for _ in range(self.num_substrings)]
        self.config["substrings"] = substrings
        strings = self.generate_strs_included_substrs(substrings)
        self.config["strings"] = strings
        states = [[0] * len(strings)]
        for substring in substrings:
            states.append(copy(states[-1]))
            for i, string in enumerate(strings):
                count = string.count(substring)
                states[-1][i] += count

        return states, copy(self.config)


class GeneratorCopy(BaseGenerator):

    def __init__(self, seed=None):
        super().__init__(seed)

    def set_config(self, num_strings, num_commands, max_length, min_length=5):
        self.num_strings = num_strings
        self.num_commands = num_commands
        self.min_length = min_length
        self.max_length = max_length
        self.config = {}
        self.config["num_strings"] = num_strings
        self.config["num_commands"] = num_commands
        self.config["max_length"] = max_length
        self.config["min_length"] = min_length

    def copy(self, strings, commands):
        copied_parts = [""] # initial state
        for i in range(len(commands)):
            part = "".join([strings[N] for N in commands[: i + 1]])
            copied_parts.append(part)
        return copied_parts

    def generate_single(self):
        len_strings = [
            self.random_state.randint(self.min_length, self.max_length)
            for _ in range(self.num_strings)
        ]
        strings = [
            generate_string(
                len_s, use_digits=True, duplicates=True, random_state=self.random_state
            )
            for len_s in len_strings
        ]

        commands = []
        for i in range(self.num_commands):
            rand_num = self.random_state.randint(0, self.num_strings-1)
            commands.append(rand_num)

        self.config["strings"] = strings
        self.config["commands"] = commands

        states = self.copy(strings, commands)

        return states, copy(self.config)


class GeneratorSubstitute(BaseGenerator):
  
    def __init__(self, seed=None):
        super().__init__(seed)
  
    def set_config(self, length):
        self.length = length
        self.num_substitutions = length
        self.config = {}
        self.config["length"] = length
        self.config["num_substitutions"] = self.num_substitutions

    def generate_random_string_and_table(self):
        _string = generate_string(self.length, use_digits=True, random_state=self.random_state)
        available_chars = list(set(_string))
        
        char_pool = string.ascii_lowercase + string.digits

        table = {}
        num = len(available_chars)
        substitution_targets = self.random_state.sample(available_chars, num)

        for char in substitution_targets:
            char_pool = char_pool.replace(char, '')
            replacement_char = self.random_state.sample(char_pool, len(char_pool))
            replacement_char = replacement_char[0]

            table[char] = replacement_char
        return _string, table

    def substitute_string(self, _string, table):
        result = list(_string)
        intermediate_states = []

        for i, char in enumerate(result):
            if char in table.keys():
                result[i] = table[char]
            intermediate_states.append(''.join(result))

        return [_string] + intermediate_states

    def generate_single(self):
        _string, table = self.generate_random_string_and_table()
        states = self.substitute_string(_string, table)

        self.config["string"] = _string
        self.config["table"] = table

        return states, copy(self.config)


class GeneratorEncode(BaseGenerator):

    def __init__(self, seed=None):
        super().__init__(seed)

    def set_config(self, num_divisions):
        self.num_divisions = num_divisions
        self.config = {
            "num_divisions": num_divisions,
        }

    def encode_sequence(self, string):
        encoded = []
        states = []
        i = 0

        while i < len(string):
            count = 1
            while (
                i + 1 < len(string) and string[i] == string[i + 1]
            ):
                i += 1
                count += 1
            encoded.append(f"{string[i]}_{count}")
            i += 1
            states.append(copy(encoded))
        return encoded, states

    def generate_single(self):
        rand_num_div = self.np_random_state.randint(1, 10, self.num_divisions)
        start_bin = self.random_state.randint(0, 1)
        string = ""
        for cnt, num_div in enumerate(rand_num_div):
            if cnt % 2 == 0:
                string += str(start_bin) * num_div
            else:
                string += str((1 - start_bin) % 2) * num_div
        encoded, states = self.encode_sequence(string)
        self.config["string"] = string
        return [""] + states, copy(self.config)
    

class GeneratorRotate(BaseGenerator):

    def __init__(self, seed=None):
        super().__init__(seed)

    def set_config(self, num_commands):
        initial_array = [list("abc"), list("def"), list("ghi")]
        self.initial_array = initial_array
        self.num_commands = num_commands
        self.config = {
            "initial_array": initial_array,
            "num_commands": num_commands,
        }

    def rotate_row_left(self, matrix, row_index):
        row = matrix[row_index]
        matrix[row_index] = row[1:] + row[:1]

    def rotate_column_up(self, matrix, col_index):
        col = [matrix[row][col_index] for row in range(len(matrix))]
        col_rotated = col[1:] + col[:1]
        for row in range(len(matrix)):
            matrix[row][col_index] = col_rotated[row]

    def flatten(self, matrix):
        return "".join("".join(row) for row in matrix)

    def perform_rotations(self, matrix, sequence):
        states = []
        for step in sequence:
            if step == 1:
                self.rotate_row_left(matrix, 0)
            elif step == 2:
                self.rotate_row_left(matrix, 1)
            elif step == 3:
                self.rotate_row_left(matrix, 2)
            elif step == 4:
                self.rotate_column_up(matrix, 0)
            elif step == 5:
                self.rotate_column_up(matrix, 1)
            elif step == 6:
                self.rotate_column_up(matrix, 2)
            states.append(self.flatten(matrix))
        return states

    def generate_single(self):
        commands = [self.random_state.randint(1, 6) for _ in range(self.num_commands)]
        matrix = [list(row) for row in self.initial_array]
        initial_state = self.flatten(matrix)
        states = self.perform_rotations(matrix, commands)
        self.config["commands"] = commands
        return [initial_state] + states, copy(self.config)


class GeneratorBreak(BaseGenerator):

    def __init__(self, seed=None):
        super().__init__(seed)

    def set_config(self, length, num_positions):
        self.length = length
        self.num_positions = num_positions
        self.config = {
            "length": length,
            "num_positions": num_positions,
        }

    def split(self, positions):
        segments_list = []
        for i in range(len(positions) + 1):
            segments = []
            pos = sorted([0] + positions[:i] + [len(self.initial_string)])
            for j in range(len(pos) - 1):
                segments.append(self.initial_string[pos[j] : pos[j + 1]])
            segments_list.append(segments)
        return segments_list

    def generate_single(self):
        self.initial_string = generate_string(
            self.length,
            use_digits=False,
            duplicates=True,
            random_state=self.random_state,
        )
        self.positions = self.random_state.sample(
            range(1, self.length), self.num_positions
        )
        states = self.split(self.positions)
        self.config["string"] = self.initial_string
        self.config["positions"] = self.positions
        return states, copy(self.config)


class GeneratorBreak2(BaseGenerator):
    
    def __init__(self, seed=None):
        super().__init__(seed)

    def set_config(self, length, num_positions):
        self.length = length
        self.num_positions = num_positions
        self.config = {
            "length": self.length,
            "num_posisions": self.num_positions,
        }
    def split(self, strings_list, positions):
        tmp = strings_list.pop(positions[0])
        tmp1, tmp2 = tmp[:positions[1]], tmp[positions[1]:]
        strings_list.insert(positions[0], tmp1)
        strings_list.insert(positions[0]+1, tmp2)
        return strings_list

    def generate_positions(self, len_string, num_positions):
        positions_list = []

        # define list stored number of each senment.
        split_len_strings = [len_string]
        for _ in range(num_positions):
            # get segment index whose segment length larger than one.
            seg_pos = self.random_state.choice(
                [idx for idx, length in enumerate(split_len_strings) if length > 1]
            ) 
            # select segment position ramdomly.
            string_pos = self.random_state.randint(1, split_len_strings[seg_pos] - 1)
            tmp = split_len_strings.pop(seg_pos)

            # update list stored number of each senment.
            split_len_strings.insert(seg_pos, string_pos)
            split_len_strings.insert(seg_pos + 1, tmp - string_pos)
            positions_list.append((seg_pos, string_pos))

        return positions_list

    def generate_single(self):
        initial_string = generate_string(
            self.length,
            use_digits=False,
            duplicates=True,
            random_state=self.random_state
        )
        positions = self.generate_positions(
            self.length,
            self.num_positions
            )
        
        strings = copy(initial_string)
        states = [[copy(strings)]]
        sample_strings_list = [copy(strings)]
        self.config["strings"] = strings
        self.config["positions"] = positions
        for pos in positions:
            tmp = sample_strings_list.pop(pos[0])
            tmp1, tmp2 = tmp[:pos[1]], tmp[pos[1]:]
            sample_strings_list.insert(pos[0], tmp1)
            sample_strings_list.insert(pos[0]+1, tmp2)
            states.append(copy(sample_strings_list))

        return states, copy(self.config)


class GeneratorCompose(BaseGenerator):
  
    def __init__(self, seed=None):
        super().__init__(seed)

    def set_config(self, length, num_steps):
        if length - 1 < num_steps:
            raise ValueError("The length of the string must be greater than the number of steps.")
        self.length = length
        self.num_steps = num_steps

        self.config = {}
        self.config["length"] = self.length
        self.config["num_steps"] = self.num_steps

    def generate_rules(self, init_state, rough_num_steps):
        left_rules = [1, 1, 2] # dummy for while loop
        while len(np.unique(left_rules)) != len(left_rules):
            rules = []
            state = init_state
            states = [init_state]
            for _ in range(rough_num_steps):
                idx = self.random_state.randint(0, len(state) - 2)
                rule_left = state[idx:idx+2]
                rule_right = generate_string(1, random_state=self.random_state)
                rules.append([rule_left, rule_right])
                state = state[:idx] + rule_right + state[idx+2:]
                states.append(state)
            # dummy rules
            for _ in range(rough_num_steps // 7 + 1):
                rules.append([generate_string(2, random_state=self.random_state), generate_string(1, random_state=self.random_state)])
            left_rules = [lr for lr, _ in rules]
        self.random_state.shuffle(rules)
        return rules

    def generate_states(self, init_state, rules):
        state = init_state
        old_state = ''
        states = [init_state]
        while len(state) != len(old_state):
            old_state = state
            for idx in range(2, len(state) + 1)[::-1]:
                substr = state[idx - 2:idx]
                for left_rule, right_rule in rules:
                    if substr == left_rule:
                        break
                else:
                    continue
                state = state[:idx - 2] + right_rule + state[idx:]
                break
            states.append(state)
        return states[:-1]
    
    def generate_single(self):
        num_steps = -1
        while num_steps != self.num_steps:
            rough_num_steps = self.random_state.randint(max(1, self.num_steps - 5),
                                                        min(self.length - 2, self.num_steps + 5))
            init_state = generate_string(self.length, random_state=self.random_state)
            rules = self.generate_rules(init_state, rough_num_steps)
            states = self.generate_states(init_state, rules)
            num_steps = len(states) - 1
            rules = [lr[0] + ', ' + lr[1] + f" -> {rr}" for lr, rr in rules]
            states = [list(state) for state in states]

            self.config['initial_state'] = init_state
            self.config['rules'] = rules
            self.config['num_steps'] = num_steps
            self.config['num_rules'] = len(rules)
        return states, copy(self.config)


class GeneratorDecompose(BaseGenerator):
    
    def __init__(self, seed):
        super().__init__(seed)

    def set_config(self, length, num_rules):
        self.length = length
        self.num_rules = num_rules
        self.config = {
            "length" : length,
            "num_rules": num_rules
        }

    def generate_rules(self, chars):
        rules = {}
        unique_chars = set(chars)
        # available_charsの長さがself.num_rulesより＋２大きくないと成り立たない
        #available_chars = set(string.ascii_lowercase) - unique_chars
        available_chars = set(string.ascii_lowercase)
        next_unique_char = set({})
        key_chars = self.random_state.sample(unique_chars, self.num_rules)
        for key_char in key_chars:
            # Create a list of possible replacements for the character
            if len(list(available_chars)) < 2:
                print(available_chars)
            # key_char = self.random_state.choice(list(unique_chars))
            available_chars = available_chars - set(key_chars)
            replacement_chars = self.random_state.choices(list(available_chars), k=2)

            # change each set
            next_unique_char.update(set(replacement_chars))
            unique_chars.remove(key_char)

            '''
            # update unique chars with ones in previous steps
            if len(unique_chars) < (len(chars) * 0.3):
                unique_chars.update(next_unique_char)
                available_chars -= next_unique_char
                next_unique_char = set({})
            if key_char in available_chars:
                available_chars.remove(key_char)
            '''
            available_chars = set(string.ascii_lowercase)
            rules[key_char] = replacement_chars

        return rules
    
    def generate_single(self):

        while True:
            chars = generate_string(self.length, random_state=self.random_state)
            unique_chars = set(chars)
            # available_charsの長さがself.num_rulesより＋２大きくないと成り立たない
            available_chars = set(string.ascii_lowercase) - unique_chars
            print("serch chars.", f"{len(unique_chars)}, {self.num_rules}, {self.length}")
            if len(unique_chars) >= self.num_rules:
                break
 
        rules = self.generate_rules(chars)
        self.config["characters"] = chars
        self.config["rules"] = rules
        states = []
        string_ = copy(chars)
        
        itv = 0
        for i in range(len(string_)):
            if len(chars) == 3:
                pass
            decomposed_string = []
            replaced = False
            for ii, char in enumerate(string_):
                if (itv != 0) and (itv > ii):
                    decomposed_string += [char]
                    continue
                
                if char in rules.keys() and not replaced:
                    decomposed_string += rules[char][0] +  rules[char][1]
                    replaced = True
                else:
                    decomposed_string += [char]
            if replaced:
                itv += 2
            else:
                break
            decomposed_string = "".join(decomposed_string)
            states.append(copy(decomposed_string))
            string_ = decomposed_string
            
        return [chars] + states, copy(self.config)


class GeneratorRhythm(BaseGenerator):
  
    def __init__(self, seed=None):
        super().__init__(seed)
        
    def set_config(self, sequence1_pattern, sequence2_variety, sequence2_length, N):
        self.sequence1_pattern = sequence1_pattern
        self.sequence2_variety = sequence2_variety
        self.sequence2_length = sequence2_length
        self.N = N
        self.config = {}
        self.config["sequence1_pattern"] = sequence1_pattern
        self.config["sequence2_variety"] = sequence2_variety
        self.config["sequence2_length"] = sequence2_length
        self.config["N"] = N

    def generate_sequences(self):
        sequence1 = list([self.random_state.randint(0, 9) for _ in range(self.sequence1_pattern)])
        sequence2 = self.random_state.choices(string.ascii_lowercase[:self.sequence2_variety], k=self.sequence2_length)
        return sequence1, sequence2
    
    def make_rhythm(self, sequence1, sequence2, N):
        len1, len2 = len(sequence1), len(sequence2)
        intermediates = []
        combined_sequence = ""
        for i in range(self.N):
            elem1 = sequence1[i % len1]
            elem2 = sequence2[i % len2]
            combined_sequence += f"{elem1}{elem2}"
            intermediates.append(combined_sequence)
        return [""] + intermediates

    def generate_single(self):
        sequence1, sequence2 = self.generate_sequences()
        states = self.make_rhythm(sequence1, sequence2, self.config["N"])
        self.config["sequence1"] = sequence1
        self.config["sequence2"] = sequence2

        return states, copy(self.config)


class GeneratorCompare(BaseGenerator):
    def __init__(self, seed=None):
        super().__init__(seed)

    def set_config(self, length, num_candidate):
        self.length = length
        self.num_candidate = num_candidate
        self.config = {
            "length": length,
            "num_candidate": num_candidate,
        }

    def generate_candidate(self, target):
        candidate = []
        candidate.append(target)

        for _ in range(self.num_candidate):
            chg_len_str = self.random_state.randint(0, self.length)
            chg_str = generate_string(chg_len_str, random_state=self.random_state)
            chg_str = "".join(chg_str)
            tmp = copy(target)
            diff = self.length - chg_len_str
            tmp = tmp.replace(tmp[diff:], chg_str)
            candidate.append(target[0] + tmp[1:])

        self.random_state.shuffle(candidate)

        return candidate
    
    def compare(self, target, candidate):
        intermediates = []
        matches = []
        for i, cand in enumerate(candidate):
            out = ""
            for t, c in zip(target, cand):
                if t == c:
                    out += t
                else:
                    break
            matches.append(out)
            if len(out) == len(target):
                intermediates.append([matches])
                break
            intermediates.append(copy(matches)[-1])
            
        return [[""]] + intermediates
    
    def generate_single(self):
        target = generate_string(
            self.length,
            random_state=self.random_state
        )
        candidate = self.generate_candidate(target)

        states = self.compare(target, candidate)

        self.config['target'] = target
        self.config['candidate'] = candidate

        return states, copy(self.config)


class GeneratorRecomposition(BaseGenerator):
  
    def __init__(self, seed=None):
        super().__init__(seed)

    def set_sentence_config(self, sentence_file_path):
        self.sentence_file_path = sentence_file_path
        self.config = {'sentence_file_path': sentence_file_path}
        
    def set_config(self, num_sentences, num_instructions):
        # Generate multiple sentences per task
        self.num_sentences = num_sentences
        self.num_instructions = num_instructions
        self.sentences = generate_sentence(num_sentences, self.sentence_file_path, random_state=self.random_state)
        self.config['num_sentences'] = num_sentences
        self.config['sentences'] = self.sentences

    def generate_nandm(self, sentences, num_instructions):
        pairs = []
        for i, sentence in enumerate(sentences):
            n = i + 1
            words = re.findall(r'\b\w+\b', sentence)
            m = self.random_state.randint(1, len(words))
            # 30% chance of adding m
            for _ in range(5):
                if self.random_state.randint(1, 10) < 4:
                    m += len(words)
                else:
                    break
            pairs.append((n, m))
            if len(pairs) >= num_instructions:
                break
        return pairs

    def get_word(self, sentence, position):
        words = re.findall(r'\b\w+\b', sentence)
        N = len(words)
        m = position
        while m > N:
            m -= N
        return words[m - 1]

    def generate_single(self):
        self.pairs = self.generate_nandm(self.sentences, self.num_instructions)
        intermediates = []
        new_sentence = []

        for n, m in self.pairs:
            word = self.get_word(self.sentences[n - 1], m)
            new_sentence.append(word)
            intermediates.append(' '.join(new_sentence))

        self.config["list"] = self.pairs

        return [None] + intermediates, copy(self.config)


class GeneratorCountv3(BaseGenerator):

    def __init__(self, seed=None):
        super().__init__(seed)

    def set_sentence_config(self, num_sentences, sentence_file_path):
        self.num_sentences = num_sentences
        self.sentence_file_path = sentence_file_path
        self.sentences = generate_sentence(num_sentences, sentence_file_path, random_state=self.random_state)
        self.iter_num = 0
        self.config = {'num_sentences': num_sentences, 'sentence_file_path': sentence_file_path}

    def set_config(self, num_words):
        self.num_words = num_words
        self.config['num_words'] = num_words

    def count_alphabets_in_word(self, word, counts):
        for char in word:
            if char.isalpha():
                counts[char.lower()] += 1
        return counts

    def generate_single(self):
        empty_list_zero = [0] * 26 # total count of alphabets
        while True:
            
            #if self.num_words == 1:
            #    print("There is no sentence consisted of just 1 word, skipping (this is not an error)")
            #    self.config['sentence'] = ""
            #    return [empty_list_zero], [copy(self.config)]
            if self.iter_num >= len(self.sentences): # this should technically throw an error as it should not happen
                print("No more sentence is available. Generate enough sentences at initialization.")
                print(f"Requiring {self.iter_num}-th sentence.")
                self.config['sentence'] = ""
                return [empty_list_zero], [copy(self.config)]

            sentence = self.sentences[self.iter_num]
            self.iter_num += 1
            sentence = re.findall(r'\b\w+\b', sentence)
            words = list(set(sentence))
            
            if self.num_words == len(words):
                break
        words = list(sentence)

        intermediates = []
        self.alphabet_counts = defaultdict(int, {char: 0 for char in string.ascii_lowercase})
        intermediates.append(list(self.alphabet_counts.values()))
        
        for word in words:
            self.alphabet_counts = self.count_alphabets_in_word(word, self.alphabet_counts)
            intermediates.append(list(self.alphabet_counts.values()))
        self.config['sentence'] = " ".join(words)

        return [intermediates], [copy(self.config)]


class GeneratorDecode(BaseGenerator):

    def __init__(self, seed):
        super().__init__(seed)

    def set_config(self, len_code):
        self.len_code = len_code
        self.bit_sentence = self.generate_bit_sentence()
        self.config = {
            "len_code": self.len_code,
            "bit_sentence": self.bit_sentence
        }
    
    def generate_bit_sentence(self):
        pattern = []
        previous_bit = 0 if 0.5 > self.np_random_state.rand() else 1
        for _ in range(self.len_code):
            length = self.random_state.randint(1, 9)
            if previous_bit == 0:
                n = 1
            else:
                n = 0
            pattern.append(f"{n}x{length}")
            previous_bit = n
            
        return ' '.join(pattern)
    
    def decode(self):
        decode = ""
        decodes = [""]
        parts = self.bit_sentence.split()
    
        for part in parts:
            char, count = part.split('x')
            decode += char * int(count)
            decodes.append(decode)
    
        return decodes
    
    def generate_single(self):
        intermediate = self.decode()

        return [intermediate], [copy(self.config)]

    
class GeneratorStackQueue(BaseGenerator):

    def __init__(self, seed):
        super().__init__(seed)

    def set_config(self, len_str, len_action, max_value=20):
        self.len_str = len_str
        if len_str == 1:
            self.len_action = len_str
        else:
            self.len_action = len_action
        self.strings = string.ascii_lowercase
        self.initial_str = generate_string(self.len_str, random_state=self.random_state)
        self.actions = self.generate_random_actions()
        self.config = {
            "len_string": self.len_str,
            "len_action": self.len_action,
            "initial": self.initial_str,
            "actions": self.actions
        }

    def generate_random_actions(self):
        actions = []
        for _ in range(self.len_action):
            action_type = self.random_state.choice(["push_left", "push_right", "pop_right", "pop_left"])
            if action_type in ["pop_left", "pop_right"]:
                actions.append(action_type)
            else:
                value = self.random_state.choice(self.strings)
                actions.append(f"{action_type}({value})")
        return actions

    def str_stack_queue(self, str, actions):
        intermediate = []
        
        intermediate.append(copy(str))
        sts = str
        for action in actions:
            if action == "pop_left":
                sts = sts[1:]
            elif action == "pop_right":
                sts = sts[:-1]
            elif action.startswith("push_right("):
                value = action[len("push_right("):-1]
                sts = sts + value
            elif action.startswith("push_left("):
                value = action[len("push_left("):-1]
                sts = value + sts
            intermediate.append(sts)
            
        return intermediate
    
    def generate_single(self):
        intermediates = self.str_stack_queue(self.initial_str, self.actions)

        return intermediates, copy(self.config)
 

class GeneratorRotate1dim(BaseGenerator):

    def __init__(self, seed=None):
        super().__init__(seed)

    def set_config(self, length, num_pairs):
        self.length = length
        self.num_pairs = num_pairs
        self.config = {"length": self.length, 'num_pairs': num_pairs}

    def rotate_substring(self, s, n, m):
        if n < 1 or m > len(s) or n > m:
            raise ValueError("Invalid n or m values")
        
        part_to_rotate = s[(n-1):(m-1)]
        rotated_part = s[m-1] + part_to_rotate
        return s[:n-1] + rotated_part + s[m:]
    
    def generate_pair(self, input_string):
        pairs = []
        for _ in range(self.num_pairs):
            m = self.np_random_state.randint(1, len(input_string))
            if m == 1:
                n = 0
            else:
                n = self.np_random_state.randint(0, m - 1)
            pairs.append((n + 1, m + 1))
        return pairs

    def generate_single(self):
        input_string = generate_string(self.length, random_state=self.random_state)
        pairs = self.generate_pair(input_string)
        intermediate_states = []
        current_string = input_string

        for pair in pairs:
            current_string = self.rotate_substring(current_string, pair[0], pair[1])
            intermediate_states.append(current_string)

        self.config["string"] = input_string
        self.config["combinations"] = pairs

        return [input_string] + intermediate_states, copy(self.config)


class GeneratorFilling(BaseGenerator):

    def __init__(self, seed=None):
        super().__init__(seed)

    def set_sentence_config(self, num_sentences, sentence_file_path):
        self.num_sentences = num_sentences
        self.sentence_file_path = sentence_file_path
        self.sentences = generate_sentence(num_sentences, sentence_file_path, random_state=self.random_state)
        self.iter_num = 0
        self.config = {'num_sentences': num_sentences, 'sentence_file_path': sentence_file_path}
        
    def set_config(self, num_pairs):
        self.num_pairs = num_pairs
        self.config['num_pairs'] = num_pairs

    def replace_number_with_word(self, sentence, num, word):
        return sentence.replace(f"[{num}]", word, 1)

    def generate_sentence_and_list(self, num_pairs, sentence):
        pattern = r'\b\w+\b|[.,\'!?()\[\]{}]-'
        tokens = re.findall(pattern, sentence)
        words_indices = {}
        words = []
        words_count = 0
        for index, token in enumerate(tokens):
            if re.match(r'\b\w+\b', token):
                words.append(token)
                words_indices[words_count] = index
                words_count += 1
                
        words_index_list = list(range(len(words)))
        pairs = []
        
        for i in range(num_pairs):
            selected_numbers = self.random_state.choice(words_index_list)
            replace_num = i + 1
            pairs.append((replace_num, words[selected_numbers]))
            replace_box = f'[{replace_num}]'
            tokens[words_indices[selected_numbers]] = replace_box
            words_index_list.remove(selected_numbers)
        
        reconstructed_text = ''
        word_index = 0

        for i, token in enumerate(tokens):
            # if token is word.
            if re.match(r'\b\w+\b', token):
                reconstructed_text += words[word_index] 
                reconstructed_text += ' '
                word_index += 1

            # if token is [num]
            elif token.startswith('[') and token.endswith(']'):
                        
                reconstructed_text += token
                reconstructed_text += ' '
                word_index += 1

            # if token is symbol
            else:
                # last word of sentence process
                if (i == (len(tokens) - 1)) & (reconstructed_text[-1] == ' '):
                    reconstructed_text = reconstructed_text[:-1] + token
                
                else:
                    reconstructed_text += token

        return pairs, reconstructed_text

    def generate_single(self):
        self.iter_num = self.random_state.sample(range(len(self.sentences)), k=1)[0]
        while True:
            if self.iter_num >= len(self.sentences):
                print("No more sentence is available. Generate enough sentences at initialization.")
                print(f"Requiring {self.iter_num}-th sentence.")
                return [], copy(self.config)

            sentence = self.sentences[self.iter_num]
            words = list(set(re.findall(r'\b\w+\b', sentence)))
            if len(words) < self.num_pairs:
                self.iter_num += 1
            else:
                break

        pairs, recon_sentence = self.generate_sentence_and_list(self.num_pairs, sentence)

        if recon_sentence[-1] == " ":
            recon_sentence = recon_sentence[:-1]

        intermediates = []

        current_sentence = copy(recon_sentence)
        for num, word in pairs:
            current_sentence = self.replace_number_with_word(current_sentence, num, word)
            intermediates.append(current_sentence)

        self.config['original_sentence'] = sentence
        self.config["sentence"] = recon_sentence
        self.config["list"] = pairs

        return [recon_sentence] + intermediates, copy(self.config)


class GeneratorStrdelete(BaseGenerator):

    def __init__(self, seed):
        super().__init__(seed)

    def set_config(self, len_string, len_step):
        self.len_string = len_string
        self.len_step = len_step
        self.config = {
            "len_string": self.len_string,
            "len_step": self.len_step,
        }
    
    def generate_step(self, string):
        # return self.random_state.sample(list(set(string)), self.len_step) 
        return self.random_state.sample(list(string), self.len_step) 
    
    def string_delete(self, string, steps):
        current_string = string
        strings = []
        strings.append(current_string)
        for char in steps:
            idx = current_string.find(char)
            current_string = current_string[:idx] + current_string[idx+1:]
            strings.append(current_string)

        return strings
    
    def generate_single(self):
        string = generate_string(self.len_string, random_state=self.random_state)
        steps = self.generate_step(string)
        self.config['string'] = string
        self.config['steps'] = steps
        intermadiate = self.string_delete(string, steps)

        return intermadiate, copy(self.config)
 

class GeneratorDeleteWords(BaseGenerator):

    def __init__(self, seed=None):
        super().__init__(seed)

    def set_sentence_config(self, num_sentences, sentence_file_path):
        self.sentence_file_path = sentence_file_path
        self.sentences = generate_sentence(num_sentences, sentence_file_path, random_state=self.random_state)
        self.num_sentences = num_sentences
        self.iter_num = 0
        self.config = {}
        self.config['num_sentences'] = self.num_sentences
        self.config['sentence_file_path'] = self.sentence_file_path

    def set_config(self, th_num_words, num_remove_word):
        # set_sentence_config must be called before calling this
        self.th_num_words = th_num_words
        self.num_remove_word = num_remove_word
        self.config["th_num_words"] = self.th_num_words
        self.config["num_remove_word"] = num_remove_word

    def generate_single(self):
        while True:
            if self.iter_num >= len(self.sentences):
                print("No more sentence is available. Generate enough sentences at initialization.")
                print(f"Requiring {self.iter_num}-th sentence.")
                return [], copy(self.config)

            sentence = self.sentences[self.iter_num]
            sentence = re.findall(r'\b\w+\b', sentence)
            words = list(set(sentence))
            sentence = " ".join(sentence)

            if (len(words) < self.th_num_words) or (len(words) <= self.num_remove_word):
                self.iter_num += 1
            else:
                break

        assert set(sentence) <= set(string.ascii_lowercase + string.ascii_uppercase + string.digits + " "), print(f"There is an unnecessary string included. {words}")
 
        sentence = sentence.replace("  ", " ")
        deleted_words = self.random_state.sample(words, self.num_remove_word)
        self.config["sentence"] = sentence
        self.config["words"] = deleted_words
        
        states = [copy(sentence)]
        for word in deleted_words:
            sentence = sentence.split(" ")
            idx = sentence.index(word)
            sentence.pop(idx)
            sentence = " ".join(sentence)
            sentence = sentence.replace("  ", " ")
            states.append(copy(sentence))

        self.iter_num += 1

        return states, copy(self.config)


class GeneratorCumulate(BaseGenerator):
  
    def __init__(self, seed=None):
        super().__init__(seed)
        self.operators = ["Add", "Multiply"]

    def set_config(self, num_operations):
        self.num_operations = num_operations

        self.config = {}
        self.config["num_operations"] = self.num_operations

    def generate_operations(self, num_operations):
        operations = []
        operations_str = []
        operators = self.random_state.choices(self.operators, k=num_operations)
        for operator in operators:
            if operator == "Multiply":
                x = self.random_state.randint(1, 5)
            else:
                x = self.random_state.randint(0, 9)

            operations.append((operator, x))
            operations_str.append(f'{operator} {x}')
        return operations, operations_str

    def generate_single(self):
        operations, operations_str = self.generate_operations(self.num_operations)
        N = self.random_state.randint(1, 9)
        self.config["operations"] = operations_str

        states = [copy(N)]
        for operation in operations:
            N = self.apply(N, operation)
            states.append(copy(N))
        
        return states, copy(self.config)

    def apply(self, N, operation):
        if operation[0] == "Add":
            return N + operation[1]
        elif operation[0] == "Multiply":
            return N * operation[1]
        else:
            return N


class GeneratorMove(BaseGenerator):

    def __init__(self, seed=None, cyclic=True, num_dims=1):
        super().__init__(seed)
        self.cyclic = cyclic
        self.num_dims = num_dims
        self.directions = ["right", "left"]
        if self.num_dims == 2:
            self.directions += ["up", "down"]
        self.target = "x"
        self.background = "-"

    def set_config(self, num_operations, width, height=1):
        self.num_operations = num_operations
        self.width = width
        self.height = height

        self.config = {}
        self.config['num_operations'] = self.num_operations
        self.config['width'] = self.width
        self.config['height'] = self.height

    def generate_initial_state(self, width, height):
        size = self.width * self.height
        initial_state = [self.background for _ in range(size)]
        initial_state[self.random_state.randint(0, size - 1)] = self.target
        initial_state =  np.array(initial_state).reshape(height, width).tolist()
        return initial_state

    def generate_operations(self, num_operations, width, height):
        operations = []
        for i in range(num_operations): 
            operation = self.generate_operation(width, height)
            operations.append(operation)
        return operations

    def generate_operation(self, width, height):
        direction = self.random_state.choice(self.directions)
        if direction in ["right", "left"]:
            distance = self.random_state.randint(1, width)
        else:
            distance = self.random_state.randint(1, height)

        return (direction, distance)

    def generate_single(self):
        initial_state = self.generate_initial_state(self.width, self.height)
        operations = self.generate_operations(self.num_operations, self.width, self.height)
        self.config["operations"] = operations

        state = initial_state
        states = [self.to_string(state)]
        for operation in operations:
            state = self.apply(copy(state), operation)
            states.append(self.to_string(state))

        return states, copy(self.config)

    def apply(self, state, operation):
        if operation[0] == "right":
            return self.move_right(state, operation[1])
        elif operation[0] == "left":
            return self.move_left(state, operation[1])
        elif operation[0] == "up":
            return self.move_up(state, operation[1])
        elif operation[0] == "down":
            return self.move_down(state, operation[1])
        return state

    def move_right(self, state, distance):
        row = self.get_target_row(state)
        index = row.index(self.target)
        new_index = index + distance 
        if new_index >= len(row): #overflow 
            if self.cyclic:
                new_index = new_index - len(row)
            else:
                new_index = len(row) - 1
        row[index] = self.background
        row[new_index] = self.target
        return state

    def move_left(self, state, distance):
        row = self.get_target_row(state)
        index = row.index(self.target)
        new_index = index - distance 
        if new_index < 0: #underflow 
            if self.cyclic:
                new_index = len(row) + new_index
            else:
                new_index = 0
        row[index] = self.background
        row[new_index] = self.target
        return state

    def move_up(self, state, distance):
        state_trans = np.array(state).transpose(1, 0).tolist()
        state_trans = self.move_left(state_trans, distance)
        return np.array(state_trans).transpose(1, 0).tolist()

    def move_down(self, state, distance):
        state_trans = np.array(state).transpose(1, 0).tolist()
        state_trans = self.move_right(state_trans, distance)
        return np.array(state_trans).transpose(1, 0).tolist()

    def get_target_row(self, state):
        for row in state:
            if self.target in row: return row
        return None

    def to_string(self, input_list):
        ret_list = []
        for row in input_list:
            ret = ""
            for el in row:
                ret += el
            ret_list.append(ret)
            
        if len(ret_list) == 1:
            ret_list = ret_list[0]
        return ret_list


class GeneratorSortedWords(BaseGenerator):

    def __init__(self, seed=None):
        super().__init__(seed)

    def set_sentence_config(self, num_sentences, sentence_file_path):
        self.num_sentences = num_sentences
        self.sentence_file_path = sentence_file_path
        self.sentences = generate_sentence(num_sentences, sentence_file_path, random_state=self.random_state)
        self.iter_num = 0
        self.config = {'num_sentences': num_sentences, 'sentence_file_path': sentence_file_path}

    def set_config(self, num_words):
        self.num_words = num_words
        self.config['num_words'] = num_words
    
    def generate_single(self):
        while True:
            empty_list_zero = [0] * 26 # total count of alphabets
            if self.num_words == 1:
                print("There is no sentence consisted of just 1 word, skipping (this is not an error)")
                self.config['sentence'] = ""
                return [empty_list_zero], [copy(self.config)]
            if self.iter_num >= len(self.sentences):
                print("No more sentence is available. Generate enough sentences at initialization.")
                print(f"Requiring {self.iter_num}-th sentence.")
                return [empty_list_zero], [copy(self.config)]

            sentence = self.sentences[self.iter_num]
            self.iter_num += 1

            words = list(set(re.findall(r'\b\w+\b', sentence)))
            if self.num_words == len(words):
                break

        words_dict = {}
        for word in words:
            first_letter = word[0].lower()
            if first_letter not in words_dict:
                words_dict[first_letter] = []
            words_dict[first_letter].append(word)

        sorted_letters = sorted(words_dict.keys())
        sorted_list = []
        intermediate_states = []

        letters_list = []

        for letter in sorted_letters:
            for each_word in words_dict[letter]:
                sorted_list.append(each_word)
                letters_list.append(letter)
                intermediate_states.append([sorted_list.copy(), letters_list.copy()])

        self.config['sentence'] = sentence

        return [intermediate_states], [copy(self.config)]

class GeneratorCyclicCheck(BaseGenerator):
  
    def __init__(self, seed=None):
        super().__init__(seed)

    def set_config(self, length):
        self.length = length

        self.config = {}
        self.config["length"] = self.length

    def generate_command(self, input_string):
        letter = self.random_state.choice(input_string)
        number = self.random_state.randint(1, len(input_string))
        return letter, number

    def generate_single(self):
        input_string = generate_string(length=self.length, duplicates=False, use_digits=True, random_state=self.random_state)
        letter, number = self.generate_command(input_string)
        self.config["string"] = input_string
        self.config["command"] = self.to_str(letter, number)

        states = [self.to_str(letter, number)]
        index = input_string.find(letter)
        for i in range(number):
            index = index + 1
            if index >= len(input_string): index = 0
            letter = input_string[index]
            number = number - 1
            if number == 0:
                states.append(letter)
            else:
                states.append(self.to_str(letter, number))
        
        return [states], [copy(self.config)]

    def to_str(self, letter, number):
        return eval(f'["{letter}", "{number}"]')


class GeneratorTagsystem(BaseGenerator):

    def __init__(self, seed):
        super().__init__(seed)

    def set_config(self, m, c_length, string_length):
        self.m = m
        self.c_length = c_length
        self.string_length = string_length
        self.config = {
            "m": self.m,
            "c_length": self.c_length,
            "string_length": self.string_length
        }

    def generate_single(self):
        # 全体の文字列をリストとして初期化
        string = [''] * self.string_length
        a_b_pos = self.random_state.choice(range(self.string_length // self.m))
        string[a_b_pos] = self.random_state.choice(['a', 'b'])
    
        # 残りの位置にランダムにa, b, cを設定
        for i in range(self.string_length):
            if string[i] == '':
                string[i] = self.random_state.choice(['a', 'b', 'c'])
    
        string = ''.join(string)
        self.config["string"] = string
        a_rule = ''.join([self.random_state.choice(["a", "b", "c"]) for _ in range(2*self.m)] + ["H"])
        b_rule = ''.join([self.random_state.choice(["a", "b", "c"]) for _ in range(self.m)] + ["a"])
        c_rule = ''.join([self.random_state.choice(["a", "b", "c"]) for _ in range(self.c_length)])
        rules = {
            "a" : a_rule,
            "b" : b_rule,
            "c" : c_rule
        }
        self.config["rules"] = rules

        results = []
        current_string = string
        while True:
            results.append(current_string)
        
            # 左からm文字以内に'H'が含まれているかチェック
            if 'H' in current_string[:self.m]:
                break
        
            # 文字列の一番最初の文字に対して変換規則を適用
            first_char = current_string[0]
            if first_char in rules.keys():
                transformed = ''.join(rules[first_char])
            else:
                transformed = first_char
        
            current_string += transformed
        
            current_string = current_string[self.m:]
        
        return results, copy(self.config)
