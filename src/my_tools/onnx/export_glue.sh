#!/bin/bash

GPU=0

MODEL_DIR=/hdd/models/intel/bert_large
MODEL_NAME=MNLI@bert-large-uncased-sparse-80-1x4-block-pruneofa@sparse_transfer_quant_decay_glue_13@BS64@EP13@HARD1.0@WD0.01@LR9.5e-5@ID9437
MODEL_NAME=QQP@bert-large-uncased-sparse-80-1x4-block-pruneofa@sparse_transfer_quant_glue_6@BS32@EP6@HARD1.0@LR1e-4@ID29298/framework

M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/bert-large/pytorch-huggingface/mnli/base_none

M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/bert-large/pytorch-huggingface/sst2/24layer_pruned80_quant-none-vnni

M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/bert-large/pytorch-huggingface/sst2/base-none

M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-masked_language_modeling/bert-large/pytorch-huggingface/wikipedia_bookcorpus/24layer_pruned80_quant-none-vnni

M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/obert-base/pytorch-huggingface/qqp/base-none

M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/obert-base/pytorch-huggingface/qqp/12layer_pruned90-none

M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/obert-base/pytorch-huggingface/qqp/12layer_pruned80_quant-none-vnni

M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/obert-base/pytorch-huggingface/mnli/base-none

M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/obert-base/pytorch-huggingface/mnli/12layer_pruned90-none

M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/obert-base/pytorch-huggingface/mnli/12layer_pruned80_quant-none-vnni

M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/obert-base/pytorch-huggingface/mnli/base-none

M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/obert-base/pytorch-huggingface/qqp/pruned90_quant-none
M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/bert-large/pytorch-huggingface/qqp/pruned90_quant-none
M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-sentiment_analysis/obert-base/pytorch-huggingface/sst2
M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-sentiment_analysis/obert-base/pytorch-huggingface/sst2/pruned90_quant-none
M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/obert-base/pytorch-huggingface/mnli/pruned90_quant-none

M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/obert-base/pytorch-huggingface/mnli/pruned90_quant-none/framework
M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/bert-large/pytorch-huggingface/mnli/pruned90_quant-none
M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/bert-large/pytorch-huggingface/qqp/pruned90_quant-none/framework
M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/bert-large/pytorch-huggingface/mnli/pruned80_quant-none-vnni/framework
M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-sentiment_analysis/bert-large/pytorch-huggingface/sst2/pruned80_quant-none-vnni/framework


CUDA_VISIBLE_DEVICES=$GPU python /hdd/src/natuan/sparseml/src/sparseml/transformers/export.py \
    --model_path $M \
    --task glue \
    --sequence_length 128 \
    --onnx_file_name model.onnx

# CUDA_VISIBLE_DEVICES=$GPU python /hdd/src/natuan/sparseml/src/sparseml/transformers/export.py \
#     --model_path $M \
#     --task glue \
#     --sequence_length 128 \
#     --no_convert_qat \
#     --onnx_file_name model_no_convert_qat.onnx
