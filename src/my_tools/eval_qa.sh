#!/bin/bash

GPU=0

ROOT=/hdd/src/natuan/sparseml/src/sparseml/transformers

MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-question_answering/bert-base/pytorch-huggingface/squad/12layer_pruned80_quant-none-vnni

CUDA_VISIBLE_DEVICES=$GPU python $ROOT/question_answering.py \
  --model_name_or_path $MODEL_DIR \
  --dataset_name squad \
  --max_eval_samples 128 \
  --do_eval \
  --per_device_eval_batch_size 16 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $HOME/tmp
