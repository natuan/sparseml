#!/bin/bash

python scripts/model_management.py create --source-path src/dvc/nlp-masked_language_modeling/obert-small/pytorch-huggingface/wikipedia_bookcorpus/base-none-bk/ --domain nlp --sub-domain masked_language_modeling --architecture obert --sub-architecture small --framework pytorch --repo huggingface --dataset wikipedia_bookcorpus --optim-name base --optim-category none --log-level info
