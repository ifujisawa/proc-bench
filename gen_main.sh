PYTHONHASHSEED=42 python generate_main.py -cn main
python preprocess.py -cn main
export $(grep -v '^#' .env | xargs) && python predict.py -cn main_o1mini
export $(grep -v '^#' .env | xargs) && python predict.py -cn main_o1preview
export $(grep -v '^#' .env | xargs) && python predict.py -cn main_4o
export $(grep -v '^#' .env | xargs) && python predict.py -cn main_geminiclaude
export $(grep -v '^#' .env | xargs) && python predict.py -cn main_mistral
export $(grep -v '^#' .env | xargs) && python extractor.py -cn main