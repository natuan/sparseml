#!/bin/bash

ZOOMODELS_DIR=/hdd/src/neuralmagic/zoomodels

MODEL_DIR=${ZOOMODELS_DIR}/src/dvc/nlp-question_answering/bert-base/pytorch-huggingface/squad/12layer_pruned90-none
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-question_answering/bert-base/pytorch-huggingface/squad/12layer_pruned80_quant-none-vnni

CMD=${CMD}

cd ${ZOOMODELS_DIR}; python ${ZOOMODELS_DIR}/scripts/model_management.py dvc-pull --model-paths ${MODEL_DIR}

cd ${CMD}
