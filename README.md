# Proc Bench

This repository is for reproducing the experiments in the paper

**ProcBench: Benchmark for Multi-Step Reasoning and Following Procedure** (https://arxiv.org/abs/2410.03117).

If you wish to generate a new dataset, you can refer to the generate_main.py script for guidance on how to do so.

## Requirement

python 3.10.15

```bash
pip install -r requirements.txt
```

## Run

```bash
# Generate the dataset
PYTHONHASHSEED=42 python generate_main.py -cn main
# Split the dataset into short, medium, and long categories
# and generate parquet files for saving on Hugging Face
python preprocess.py -cn main
# Call the API to make predictions
export $(grep -v '^#' .env | xargs) && python predict.py -cn main_o1mini
export $(grep -v '^#' .env | xargs) && python predict.py -cn main_o1preview
export $(grep -v '^#' .env | xargs) && python predict.py -cn main_4o
export $(grep -v '^#' .env | xargs) && python predict.py -cn main_geminiclaude
export $(grep -v '^#' .env | xargs) && python predict.py -cn main_mistral
# Extract data and format it into JSON using GPT-4o
export $(grep -v '^#' .env | xargs) && python extractor.py -cn main
# Evaluate the prediction
python evaluate.py
```
