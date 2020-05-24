#!/bin/bash

PYTHON_NAME="python3"  # change this to Python3's command in your PC

# compile vz-projector
cd tensorboard
bazel build //tensorboard/plugins/projector/vz_projector:devserver
cp ./bazel-bin/tensorboard/plugins/projector/vz_projector/devserver.html ../index.html

# compute embeddings
cd ..
$PYTHON_NAME compute-embeddings.py

echo DONE!
