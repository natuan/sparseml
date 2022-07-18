#!/bin/bash

GPU=0

M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-token_classification/bert-large/pytorch-huggingface/conll2003/base-none-org
M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-token_classification/bert-large/pytorch-huggingface/conll2003/pruned80_quant-none-vnni-org

M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-token_classification/obert-base/pytorch-huggingface/conll2003/base-none
M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-token_classification/obert-base/pytorch-huggingface/conll2003/pruned90_quant-none

M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-token_classification/bert-large/pytorch-huggingface/conll2003/pruned80_quant-none-vnni/framework

M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-token_classification/bert-large/pytorch-huggingface/conll2003/pruned90_quant-none/framework

CUDA_VISIBLE_DEVICES=$GPU python /hdd/src/natuan/sparseml/src/sparseml/transformers/export.py \
    --model_path $M \
    --task ner \
    --sequence_length 512 \
    --onnx_file_name model.onnx

# CUDA_VISIBLE_DEVICES=$GPU python /hdd/src/natuan/sparseml/src/sparseml/transformers/export.py \
#     --model_path $M \
#     --task glue \
#     --sequence_length 128 \
#     --no_convert_qat \
#     --onnx_file_name model_no_convert_qat.onnx
