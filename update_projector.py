import json
import re
from gensim.models import FastText
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


# change this to where the model is stored
# modify these two variables
MODEL_PATH = r"F:\fulltext_covid19_model\fulltext_covid19_model_fasttext_optimized.model"
NUM_OF_WORDS = 10_000

# get words
words = []
model = FastText.load(MODEL_PATH)
for each in list(model.wv.vocab.keys()):
    each = re.sub(r"_", " ", each)
    # do some filter here

    words.append(each)
        
    if len(words) >= NUM_OF_WORDS:
        break

# get vectors
vecs = []
for word in words:
    vec = model.wv.get_vector(word)
    vecs.append(vec.tolist())
    
vecs = np.array(vecs, dtype=np.float32)
dim = vecs.shape[1]

# save vectors and labels
with open(r"rsc/labels.tsv", "w", encoding="utf-8") as f, \
    open(r"rsc/tensor.bytes", "wb") as g:
    f.write("\n".join(words)+"\n")
    g.write(vecs.tobytes())


# # Get all projectors precomputed
# ## PCA
pca = PCA(n_components=3)
pca.fit(vecs)
print(pca.explained_variance_ratio_)
ratio = pca.explained_variance_ratio_
ratio.shape = (1, -1)
print(pca.singular_values_)

PCA_tensor = pca.transform(vecs)
PCA_tensor_ = np.concatenate((ratio, PCA_tensor), axis=0)
PCA_tensor_.shape

# save PCA result
with open("rsc/pca.bytes", "wb") as f:
    f.write(PCA_tensor_.tobytes())


# ## UMAP
umap = UMAP(n_neighbors=5, n_components=3, metric="cosine")
UMAP_tensor = umap.fit_transform(vecs)

# save UMAP
with open("rsc/umap.bytes", "wb") as f:
    f.write(UMAP_tensor.tobytes())


# ## t-SNE
tsne = TSNE(n_components=3, learning_rate=10, n_iter=1000, metric="cosine", verbose=2)
TSNE_tensor = tsne.fit_transform(vecs)

# save tsne
with open("rsc/tsne.bytes", "wb") as f:
    f.write(TSNE_tensor.tobytes())


# ##config file
config = {
  "embeddings": [
    {
      "tensorName": "COVID-19 Word Embedding",
      "tensorShape": [NUM_OF_WORDS, dim],
      "tensorPath": "rsc/tensor.bytes",
      "metadataPath": "rsc/labels.tsv"
    }
  ],
  "modelCheckpointPath": "COVID Dataset"
}

# save config
with open(r"rsc/oss_demo_projector_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)