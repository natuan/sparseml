#!/bin/bash

# Model folder
MODEL_DIR=/hdd/models/intel/bert_large/QQP@bert-large-uncased@BS32@W0.0@LR2e-5WD0.01@EP3@ID6176
MODEL_DIR=/hdd/models/intel/bert_large/SQuAD@bert-large-uncased@BS32@W0.1@LR5e-5@EP3@ID6626
MODEL_DIR=/hdd/models/intel/bert_large/QQP@bert-large-uncased-sparse-80-1x4-block-pruneofa@sparse_transfer_quant_glue_6@BS32@EP6@HARD1.0@LR1e-4@ID29298
MODEL_DIR=/hdd/models/intel/bert_large/SQuAD@bert-large-uncased-sparse-80-1x4-block-pruneofa@sparse_transfer_quant_decay_squad_13@EP13@HARD1.0@BS32@WD0.01@LR1.5e-4@ID23504
MODEL_DIR=/hdd/models/intel/bert_large/MNLI@bert-large-uncased-sparse-80-1x4-block-pruneofa@sparse_transfer_quant_decay_glue_13@BS64@EP13@HARD1.0@WD0.01@LR9.5e-5@ID9437
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/bert-large/pytorch-huggingface/mnli/base_none
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/bert-large/pytorch-huggingface/sst2/24layer_pruned80_quant-none-vnni
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/bert-large/pytorch-huggingface/sst2/base-none
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-masked_language_modeling/bert-large/pytorch-huggingface/wikipedia_bookcorpus/24layer_pruned80_quant-none-vnni
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-masked_language_modeling/obert-base/pytorch-huggingface/wikipedia_bookcorpus/base-none
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-masked_language_modeling/obert-base/pytorch-huggingface/wikipedia_bookcorpus/3layer_base-none
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-masked_language_modeling/obert-base/pytorch-huggingface/wikipedia_bookcorpus/6layer_base-none
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-masked_language_modeling/obert-base/pytorch-huggingface/wikipedia_bookcorpus/12layer_pruned90-none
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-masked_language_modeling/obert-base/pytorch-huggingface/wikipedia_bookcorpus/12layer_pruned80-none-vnni
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/obert-base/pytorch-huggingface/qqp/base-none
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/obert-base/pytorch-huggingface/qqp/12layer_pruned90-none
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/obert-base/pytorch-huggingface/qqp/12layer_pruned80_quant-none-vnni
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-masked_language_modeling/bert-large/pytorch-huggingface/wikipedia_bookcorpus/base-none
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/obert-base/pytorch-huggingface/mnli/base-none
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/obert-base/pytorch-huggingface/mnli/12layer_pruned90-none
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/obert-base/pytorch-huggingface/mnli/12layer_pruned80_quant-none-vnni
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-token_classification/bert-large/pytorch-huggingface/conll2003/base-none-org
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-token_classification/bert-large/pytorch-huggingface/conll2003/pruned80_quant-none-vnni-org
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-token_classification/obert-base/pytorch-huggingface/conll2003/base-none
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-token_classification/obert-base/pytorch-huggingface/conll2003/pruned90_quant-none
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-token_classification/obert-base/pytorch-huggingface/conll2003/pruned80_quant-none-vnni
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-question_answering/obert-base/pytorch-huggingface/squad/pruned90_quant-none
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/obert-base/pytorch-huggingface/qqp/pruned90_quant-none
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-question_answering/bert-large/pytorch-huggingface/squad/pruned90_quant-none
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/bert-large/pytorch-huggingface/qqp/pruned90_quant-none
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-sentiment_analysis/obert-base/pytorch-huggingface/sst2/pruned90_quant-none
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/obert-base/pytorch-huggingface/mnli/pruned90_quant-none
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/bert-large/pytorch-huggingface/mnli/pruned90_quant-none
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-token_classification/bert-large/pytorch-huggingface/conll2003/pruned90_quant-none
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-masked_language_modeling/bert-large/pytorch-huggingface/wikipedia_bookcorpus/pruned90-none
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-sentiment_analysis/bert-large/pytorch-huggingface/sst2/pruned90_quant-none
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-document_classification/obert-base/pytorch-huggingface/imdb/base-none_ID23809
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-document_classification/obert-base/pytorch-huggingface/imdb/pruned80_quant-none-vnni
MODEL_DIR=/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-document_classification/obert-base/pytorch-huggingface/imdb/pruned90_quant-none

# Step 0: exporting model to onnx
ONNX_FILE=${MODEL_DIR}/model.onnx
[ -f ${ONNX_FILE} ] && echo "ONNX model exists" || echo "ONNX model does not exist"

# Step 1: export sample inputs, outputs
INPUT_DIR=${MODEL_DIR}/sample-inputs
[ -d ${INPUT_DIR} ] && echo "Sample inputs exist" || { echo "Sample inputs not exist"; exit 1; }

OUTPUT_DIR=${MODEL_DIR}/sample-outputs
[ -d ${OUTPUT_DIR} ] && echo "Sample outputs exist" || { echo "Sample outputs not exist"; exit 1; }

# Step 3: create framework folder
CMD=${CMD}
cd ${MODEL_DIR}
if [ ! -d framework ]; then
    echo "Creating framework folder"
    mkdir framework
    framework_files="all_results.json eval_results.json config.json pytorch_model.bin special_tokens_map.json tokenizer_config.json tokenizer.json trainer_state.json training_args.bin train_results.json vocab.txt eval_nbest_predictions.json eval_predictions.json recipe.yaml"
    for f in ${framework_files}; do
	mv ${f} framework
    done
else
    echo "Folder framework already exists. Do nothing."
fi
cd ${CMD}

# Step 4: create recipes folder, add the training recipe into it

# Step 5: create and edit the model card

# Step 6: Create training folder, add validation metric file into it

# Step 7: Create dvc files

# Step 8: push: dvc push ./*.dvc
