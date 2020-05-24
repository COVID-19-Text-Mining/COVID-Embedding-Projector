#!/usr/bin/env python
# coding: utf-8

# # Get all embeddings precomputed
# ## Load word embeddings

# In[1]:


import json
from bson import json_util
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import requests

TENSOR_URL = r"https://projector.yuxingfei.top/rsc/tensor.bytes"


# In[2]:


a = requests.get(TENSOR_URL).content
vec = np.frombuffer(a, dtype=np.float32)
vec.shape = (-1, 300)


# ## PCA

# In[3]:


from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(vec)


# In[4]:


print(pca.explained_variance_ratio_)
ratio = pca.explained_variance_ratio_
ratio.shape = (1, -1)
print(pca.singular_values_)


# In[5]:


PCA_tensor = pca.transform(vec)


# In[6]:


PCA_tensor_ = np.concatenate((ratio, PCA_tensor), axis=0)
PCA_tensor_.shape


# In[7]:


with open("pca.bytes", "wb") as f:
    f.write(PCA_tensor_.tobytes())


# ## UMAP

# In[8]:


from umap import UMAP
from umap import UMAP


# In[9]:


umap = UMAP(n_neighbors=15, n_components=3, metric="cosine")
UMAP_tensor = umap.fit_transform(vec)


# In[10]:


UMAP_tensor


# In[11]:


with open("umap.bytes", "wb") as f:
    f.write(UMAP_tensor.tobytes())


# ## t-SNE

# In[12]:


from sklearn.manifold import TSNE


# In[16]:


tsne = TSNE(n_components=3, learning_rate=100, n_iter=1000, metric="cosine", verbose=2)
TSNE_tensor = tsne.fit_transform(vec)


# In[14]:


TSNE_tensor


# In[15]:


with open("tsne.bytes", "wb") as f:
    f.write(TSNE_tensor.tobytes())

