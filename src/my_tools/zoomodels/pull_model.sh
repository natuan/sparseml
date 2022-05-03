#!/bin/bash

ZOOMODELS_DIR=/hdd/src/neuralmagic/zoomodels

MODEL_DIR=${ZOOMODELS_DIR}/src/dvc/nlp-text_classification/distilbert-none/pytorch-huggingface/qqp/base-none/

CMD=${CMD}

cd ${ZOOMODELS_DIR}; python ${ZOOMODELS_DIR}/scripts/model_management.py dvc-pull --model-paths ${MODEL_DIR}

cd ${CMD}
