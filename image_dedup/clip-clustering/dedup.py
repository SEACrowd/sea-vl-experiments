from PIL import Image
from matplotlib import pyplot as plt
import requests

from transformers import CLIPProcessor, CLIPModel

# if you are having trouble loading this specific model, ensure you don't have a 
# local directory containing the same model, if you do, you could use that 
# instead and put the path to that model as the value in the model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

import torch
from pathlib import Path

demo_directory = Path("./seacrowd-sample") #replace Path value to directory containing your images
images_to_paths = {image_path.stem: image_path for image_path in demo_directory.iterdir()}

images = [Image.open(path) for path in images_to_paths.values()]
inputs = processor(images=images, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model.get_image_features(**inputs)

images_to_embeddings = {image_id: tensor_embedding.detach().numpy() for image_id, tensor_embedding in zip(images_to_paths.keys(), outputs)}

import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict

# tune eps to fit your needs (6 is the sweet spot for me)
clustering = DBSCAN(min_samples=2, eps=6).fit(np.stack(images_to_embeddings.values()))

# postprocess cluster labels into groups of similar images
image_id_communities = defaultdict(set)
independent_image_ids = set()

for image_id, cluster_idx in zip(images_to_paths.keys(), clustering.labels_):
    cluster_idx = int(cluster_idx)
    if cluster_idx == -1:
        independent_image_ids.add(image_id)
        continue

    image_id_communities[cluster_idx].add(image_id)

len(independent_image_ids) # = 10

image_id_communities

### Just shows the images considered as duplicates
# for image_id_community in image_id_communities.values():
#     for image_id in image_id_community:
#         plt.figure()
#         plt.imshow(Image.open(images_to_paths[image_id]))
#         plt.show()

### Just shows the images that aren't considered as duplicates
# # images that have not got a cluster with similar images assigned to them
# for image_id in independent_image_ids:
#     plt.figure()
#     plt.imshow(Image.open(images_to_paths[image_id]))
#     plt.show()

# print file names of images considered as duplicates
print('image id communities', image_id_communities)

# print file names of images not considered as duplicates
# print('independent image ids', independent_image_ids)