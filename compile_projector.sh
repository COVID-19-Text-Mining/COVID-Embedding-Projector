#!/bin/bash

cd tensorboard
bazel build //tensorboard/plugins/projector/vz_projector:devserver
cp ./bazel-bin/tensorboard/plugins/projector/vz_projector/devserver.html ../index.html
echo OK

