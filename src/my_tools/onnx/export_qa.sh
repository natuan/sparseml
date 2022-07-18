#!/bin/bash
GPU=0

MODEL_DIR=/hdd/models/intel/bert_large
MODEL_NAME=SQuAD@bert-large-uncased-sparse-80-1x4-block-pruneofa@sparse_transfer_quant_decay_squad_13@EP13@HARD1.0@BS32@WD0.01@LR1.5e-4@ID23504

M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-question_answering/obert-base/pytorch-huggingface/squad/pruned90_quant-none
M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-question_answering/bert-large/pytorch-huggingface/squad/pruned90_quant-none

M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-question_answering/bert-large/pytorch-huggingface/squad/pruned90_quant-none/framework
M=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-question_answering/bert-large/pytorch-huggingface/squad/pruned80_quant-none-vnni/framework02

CUDA_VISIBLE_DEVICES=$GPU python /hdd/src/neuralmagic/sparseml/src/sparseml/transformers/export.py \
  --model_path $M \
  --task qa \
  --onnx_file_name model.onnx

# CUDA_VISIBLE_DEVICES=$GPU python /hdd/src/neuralmagic/sparseml/src/sparseml/transformers/export.py \
#   --model_path $M \
#   --task qa \
#   --sequence_length 384 \
#   --onnx_file_name model.onnx


# CUDA_VISIBLE_DEVICES=$GPU python /hdd/src/neuralmagic/sparseml/src/sparseml/transformers/export.py \
#   --model_path $M \
#   --task qa \
#   --sequence_length 384 \
#   --no_convert_qat \
#   --onnx_file_name model_no_convert.onnx
