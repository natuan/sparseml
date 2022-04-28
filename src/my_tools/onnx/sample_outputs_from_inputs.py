import os
import onnxruntime
import numpy as np

# Folder to get sample inputs in npz
sample_inputs_dir = "/hdd/models/intel/bert_large/QQP@bert-large-uncased@BS32@W0.0@LR2e-5WD0.01@EP3@ID6176/sample-inputs"

# Folder to write outputs in npz
sample_outputs_dir = "/hdd/models/intel/bert_large/QQP@bert-large-uncased@BS32@W0.0@LR2e-5WD0.01@EP3@ID6176/sample-outputs"

# The onnx model to use
onnx_file_path = "/hdd/models/intel/bert_large/QQP@bert-large-uncased@BS32@W0.0@LR2e-5WD0.01@EP3@ID6176/model.onnx"


sess = onnxruntime.InferenceSession(onnx_file_path)

input_names = [i.name for i in sess.get_inputs()]
output_names = [o.name for o in sess.get_outputs()]

input_shapes = {i.name: i.shape for i in sess.get_inputs()}

os.makedirs(sample_outputs_dir, exist_ok=True)

input_file_names = sorted([f for f in os.listdir(sample_inputs_dir) if f.endswith(".npz")])
for idx, f in enumerate(input_file_names):
    print(f"Processing input file: {f}")
    file_idx = f"{idx}".zfill(4)
    assert f == f"inp-{file_idx}.npz"
    arr = np.load(os.path.join(sample_inputs_dir, f))
    output_vals = sess.run(
        output_names,
        {
            k: arr[k].reshape(input_shapes[k]) for k in arr.files if k in input_names
        }
    )
    output_dict = {
        name: np.squeeze(val) for name, val in zip(output_names, output_vals)
    }
    output_file_path = os.path.join(sample_outputs_dir, f"out-{file_idx}.npz")
    np.savez(output_file_path, **output_dict)

print("Done")
