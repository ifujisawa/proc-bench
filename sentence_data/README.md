# Sentence Extractor

This script makes text file from wiki-en articles for generating the experiment datasets.

The script uses the [izumi-lab/wikipedia-en-20230720](https://huggingface.co/datasets/izumi-lab/wikipedia-en-20230720) dataset from hugging face. 

Text data of some articles is obtained from the dataset.
It is preprocessed to remove sentences with non-English text, and saved as a text file.

## Prerequisites

Ensure you have the following Python packages installed:

- `nltk`
- `datasets`
- `tqdm`

You can install them using pip:

```sh
pip install nltk datasets tqdm
```

## Configuration

The script has several configurable parameters:

- output_file: The path to the output file where the extracted sentences will be saved.
- seed: The random seed for reproducibility.
- num_articles: The number of articles obtained from wiki-en dataset.

## Run

Execute the script by running:

```sh
python get_sentences.py
```