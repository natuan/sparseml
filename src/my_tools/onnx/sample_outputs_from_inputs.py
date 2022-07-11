import os
import onnxruntime
import numpy as np

# Folder to get sample inputs in npz
src_sample_inputs_dir = "/hdd/models/intel/bert_large/QQP@bert-large-uncased-sparse-80-1x4-block-pruneofa@sparse_transfer_quant_glue_6@BS32@EP6@HARD1.0@LR1e-4@ID29298/sample-inputs"
src_sample_inputs_dir = "/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-masked_language_modeling/bert-base/pytorch-huggingface/wikipedia_bookcorpus/12layer_pruned90-none/sample-inputs"
src_sample_inputs_dir = "/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/bert-base/pytorch-huggingface/qqp/base-none/sample-inputs"
src_sample_inputs_dir = "/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/bert-base/pytorch-huggingface/qqp/base-none/sample-inputs"
src_sample_inputs_dir = "/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/bert-large/pytorch-huggingface/qqp/base-none/sample-inputs"
src_sample_inputs_dir = "/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-token_classification/bert-base/pytorch-huggingface/conll2003/base-none/sample-inputs"
src_sample_inputs_dir = "/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-question_answering/bert-base/pytorch-huggingface/squad/12layer_pruned90-none/sample-inputs"

assert os.path.exists(src_sample_inputs_dir)

# Destination model folder
model_dir = "/hdd/models/intel/bert_large/SQuAD@bert-large-uncased-sparse-80-1x4-block-pruneofa@sparse_transfer_quant_decay_squad_13@EP13@HARD1.0@BS32@WD0.01@LR1.5e-4@ID23504"
model_dir = "/hdd/models/intel/bert_large/SQuAD@bert-large-uncased@BS32@W0.1@LR5e-5@EP3@ID6626"
model_dir = "/hdd/models/intel/bert_large/QQP@bert-large-uncased-sparse-80-1x4-block-pruneofa@sparse_transfer_quant_glue_6@BS32@EP6@HARD1.0@LR1e-4@ID29298"
model_dir = "/hdd/models/intel/bert_large/MNLI@bert-large-uncased-sparse-80-1x4-block-pruneofa@sparse_transfer_quant_decay_glue_13@BS64@EP13@HARD1.0@WD0.01@LR9.5e-5@ID9437"
model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/bert-large/pytorch-huggingface/mnli/base_none"
model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/bert-large/pytorch-huggingface/sst2/24layer_pruned80_quant-none-vnni"
model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/bert-large/pytorch-huggingface/sst2/base-none"
model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-masked_language_modeling/bert-large/pytorch-huggingface/wikipedia_bookcorpus/24layer_pruned80_quant-none-vnni"
model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-masked_language_modeling/obert-base/pytorch-huggingface/wikipedia_bookcorpus/base-none"
model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-masked_language_modeling/obert-base/pytorch-huggingface/wikipedia_bookcorpus/6layer_base-none"
model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-masked_language_modeling/obert-base/pytorch-huggingface/wikipedia_bookcorpus/12layer_pruned90-none"
model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-masked_language_modeling/obert-base/pytorch-huggingface/wikipedia_bookcorpus/12layer_pruned80-none-vnni"
model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/obert-base/pytorch-huggingface/qqp/base-none"
model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/obert-base/pytorch-huggingface/qqp/12layer_pruned90-none"
model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/obert-base/pytorch-huggingface/qqp/12layer_pruned80_quant-none-vnni"
model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-masked_language_modeling/bert-large/pytorch-huggingface/wikipedia_bookcorpus/base-none"
model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/obert-base/pytorch-huggingface/qqp/12layer_pruned80-none-vnni_from_Mark"
model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/obert-base/pytorch-huggingface/mnli/base-none"
model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/obert-base/pytorch-huggingface/mnli/12layer_pruned90-none"
model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/obert-base/pytorch-huggingface/mnli/12layer_pruned80_quant-none-vnni"
model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-token_classification/bert-large/pytorch-huggingface/conll2003/base-none-org"
model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-token_classification/bert-large/pytorch-huggingface/conll2003/pruned80_quant-none-vnni-org"
model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-token_classification/obert-base/pytorch-huggingface/conll2003/base-none"
model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-token_classification/obert-base/pytorch-huggingface/conll2003/pruned90_quant-none"
model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-token_classification/obert-base/pytorch-huggingface/conll2003/pruned80_quant-none-vnni"
model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-question_answering/obert-base/pytorch-huggingface/squad/pruned90_quant-none"
model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/obert-base/pytorch-huggingface/qqp/pruned90_quant-none"
model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-question_answering/bert-large/pytorch-huggingface/squad/pruned90_quant-none"
model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/bert-large/pytorch-huggingface/qqp/pruned90_quant-none"
model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-sentiment_analysis/obert-base/pytorch-huggingface/sst2/pruned90_quant-none"
model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/obert-base/pytorch-huggingface/mnli/pruned90_quant-none"
model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-text_classification/bert-large/pytorch-huggingface/mnli/pruned90_quant-none"
model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-token_classification/bert-large/pytorch-huggingface/conll2003/pruned90_quant-none"

model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-question_answering/obert-base/pytorch-huggingface/squad/base-none"

model_dir="/hdd/src/neuralmagic/zoomodels/src/dvc/nlp-question_answering/bert-large/pytorch-huggingface/squad/pruned90_quant-none"

# Folder to write inputs/outputs in npz
sample_inputs_dir = os.path.join(model_dir, "sample-inputs")
sample_outputs_dir = os.path.join(model_dir, "sample-outputs")

assert not os.path.exists(sample_inputs_dir), f"{sample_inputs_dir} already exists"
assert not os.path.exists(sample_outputs_dir), f"{sample_outputs_dir} already exists"
os.makedirs(sample_inputs_dir)
os.makedirs(sample_outputs_dir)

# The onnx model to use
onnx_file_path = os.path.join(model_dir, "model.onnx")

sess = onnxruntime.InferenceSession(onnx_file_path)

input_names = [i.name for i in sess.get_inputs()]
output_names = [o.name for o in sess.get_outputs()]

input_shapes = {i.name: i.shape for i in sess.get_inputs()}

# Sometimes the sample input folder contain npz with different names
# from the model. The following map will be used to fix that
# Assign to None to ignore it
input_name_map = {
    "input_ids": "input_0",
    "attention_mask": "input_1",
    "token_type_ids": "input_2"
}
#input_name_map = None

input_file_names = sorted([f for f in os.listdir(src_sample_inputs_dir) if f.endswith(".npz")])
for idx, f in enumerate(input_file_names):
    print(f"Processing input file: {f}")
    file_idx = f"{idx}".zfill(4)
    assert f == f"inp-{file_idx}.npz"
    arr = np.load(os.path.join(src_sample_inputs_dir, f))

    input_dict_to_model = {}
    input_dict_to_save = {}
    for input_name in input_names:
        input_shape = input_shapes[input_name]
        k = input_name_map[input_name] if input_name_map else input_name
        input_dict_to_save[input_name] = arr[k]
        input_dict_to_model[input_name] = arr[k].reshape(input_shape)

    output_vals = sess.run(output_names, input_dict_to_model)
    output_dict = {
        name: np.squeeze(val) for name, val in zip(output_names, output_vals)
    }
    input_file_path = os.path.join(sample_inputs_dir, f"inp-{file_idx}.npz")
    output_file_path = os.path.join(sample_outputs_dir, f"out-{file_idx}.npz")
    np.savez(input_file_path, **input_dict_to_save)
    np.savez(output_file_path, **output_dict)

print("Done")
