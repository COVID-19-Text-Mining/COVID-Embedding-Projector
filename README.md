# COVID-19 Embedding Projector
This is a project forked from Google's [embedding projector](https://github.com/tensorflow/embedding-projector-standalone).

## Demo
You can find a demo on [https://projector.yuxingfei.top](https://projector.yuxingfei.top)

## Update Embeddings
Here we have an automatic script `update_projector.py` to update the embeddings with `gensim.models.FastText` model.

First, you should change the dir to the root of this repo. Then change the variable `MODEL_PATH` to the path to the `FastText` model and `NUM_OF_WORDS`. Finally,  run the script `update_projector.py`.