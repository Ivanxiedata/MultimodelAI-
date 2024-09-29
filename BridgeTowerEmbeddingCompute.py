import json
import os
import numpy as np
from numpy.linalg import norm
import cv2
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from tqdm import tqdm
from utils import encode_image
from utils import bt_embedding_from_prediction_guard as bt_embeddings
from embedding import img1, img2, img3


embeddings = []
for img in [img1, img2, img3]:
    img_path = img['image_path']
    caption = img['caption']
    base64_img = encode_image(img_path)
    embedding = bt_embeddings(caption, base64_img)
    embeddings.append(embedding)

print(len(embeddings[0]))