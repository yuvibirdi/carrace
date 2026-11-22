#!/bin/sh
# Copy the code and recordings to the GPU machine and run a training script there:
#     host/runpod.sh dreamer.py [args...]      (or train.py)
# Results (model files, logs) are copied back into data/.
cd "$(dirname "$0")/.."
rsync -az --include='*/' --include='host/*.py' --include='data/ep_*.mjpeg' --include='data/other/ep_*.mjpeg' \
    --include='data/model.pt' --exclude='*' ./ runpod:carrace/ || exit 1
# The machine's own Python has PyTorch but not the onnx package the model export needs,
# so run from an environment that adds it on top.
ssh runpod "cd carrace && { [ -x ~/carrace-env/bin/python ] || { uv venv -q --system-site-packages ~/carrace-env && uv pip install -q --python ~/carrace-env/bin/python onnx; }; } && ~/carrace-env/bin/python -u host/$*" || exit 1
rsync -az --include='*.pt' --include='*.onnx' --include='*.csv' --exclude='*' runpod:carrace/data/ data/
