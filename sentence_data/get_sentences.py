import gc
import os
import random
import string

import nltk
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

nltk.download('punkt')
nltk.download('punkt_tab')


def is_first_letter_lowercase(s):
    for char in s:
        if char.isalpha():
            return char.islower()
    return False


def contains_only_english(text, allowed_chars):
    for char in text:
        if char not in allowed_chars:
            return False
    return True


def split_text_into_sentences(dataset_text, output_file):
    allowed_chars = set(string.ascii_letters + string.digits + '..,!?\'"()-' + ' ')
    with open(output_file, 'w', encoding='utf-8') as file:
        for text in tqdm(dataset_text):
            sentences = sent_tokenize(text)
            remain_sentence = ''
            total_sentences = 0

            for sentence in sentences:
                if contains_only_english(sentence, allowed_chars):
                    if (is_first_letter_lowercase(sentence)) & (total_sentences > 0):
                        remain_sentence += ' '
                        remain_sentence += sentence
                    
                    else:
                        if total_sentences > 0:
                            file.write(remain_sentence + '\n')
                        total_sentences += 1
                        remain_sentence = ''
                        remain_sentence += sentence
                

if __name__ == '__main__':
    output_file = os.path.dirname(os.path.abspath(__file__)) + '/sentences.txt'
    seed = 42
    num_articles = 3000

    random.seed(seed)

    ds = load_dataset("izumi-lab/wikipedia-en-20230720")
    dataset_text = random.sample(ds['train']['text'], num_articles)
    
    del ds
    gc.collect()

    split_text_into_sentences(dataset_text, output_file)



