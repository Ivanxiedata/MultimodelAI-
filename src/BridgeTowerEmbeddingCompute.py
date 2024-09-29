import os
import logging
from embedding import img1, img2, img3
from utils import prepare_dataset_for_umap_visualization as data_prep
from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning
from utils import get_embedding, encode_image
import pandas as pd
from tqdm import tqdm
import numpy as np
from numpy.linalg import norm
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Set environment variable to avoid HuggingFace parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Function to load the model and processor
def load_model_and_processor():
    logging.info("Loading the BridgeTower processor and model")
    modelprocessor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
    model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
    return model, modelprocessor


# Function to compute embedding for a single image
def compute_embedding(model, modelprocessor, img_path, caption):
    logging.info(f"Computing embedding for image: {img_path}, caption: {caption}")
    return get_embedding(model, modelprocessor, img_path, caption)


# Function to compute cosine similarity
def cosine_similarity(vec1, vec2):
    similarity = np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    return similarity


# Function to prepare dataset and compute embeddings
def process_dataset(model, modelprocessor, dataset_name, label, test_size):
    logging.info(f"Preparing image-text pairs for '{dataset_name}' dataset")
    img_txt_pairs = data_prep(dataset_name, label, test_size=test_size)
    embeddings = []

    for img_txt_pair in tqdm(img_txt_pairs, total=len(img_txt_pairs)):
        pil_img = img_txt_pair['pil_img']
        caption = img_txt_pair['caption']
        base64_img = encode_image(pil_img)
        embedding = get_embedding(model, modelprocessor, base64_img, caption)
        embeddings.append(embedding)

    return embeddings, img_txt_pairs


# Function to perform dimensionality reduction
def dimensionality_reduction(embed_arr, labels):
    logging.info("Performing dimensionality reduction using UMAP")
    X_scaled = MinMaxScaler().fit_transform(embed_arr)
    logging.info(f"Scaled embedding array: {X_scaled}")
    mapper = umap.UMAP(n_components=2, metric="cosine").fit(X_scaled)
    df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
    df_emb["label"] = labels
    logging.info(f"Reduced dimensionality embeddings: {df_emb}")
    return df_emb


# Main function to run the entire process
def main():
    # Load model and processor
    model, modelprocessor = load_model_and_processor()

    # Process predefined images
    logging.info("Processing predefined images for embedding")
    embeddings = []
    for img in [img1, img2, img3]:
        img_path = img['image_path']
        caption = img['caption']
        embedding = compute_embedding(model, modelprocessor, img_path, caption)
        embeddings.append(embedding)

    logging.info(f"Length of the first embedding: {len(embeddings[0])}")

    # Compute cosine similarity between embeddings
    logging.info("Computing cosine similarity between embeddings")
    sim_ex1_ex2 = cosine_similarity(np.array(embeddings[0]), np.array(embeddings[1]))
    sim_ex1_ex3 = cosine_similarity(np.array(embeddings[0]), np.array(embeddings[2]))

    logging.info(f"Similarity between Example 1 and Example 2: {sim_ex1_ex2}")
    logging.info(f"Similarity between Example 1 and Example 3: {sim_ex1_ex3}")

    # Process datasets for cats and cars
    logging.info("Processing 'yashikota/cat-image-dataset' for cat embeddings")
    cat_embeddings, cat_img_txt_pairs = process_dataset(
        model, modelprocessor, "yashikota/cat-image-dataset", "cat", 50
    )

    logging.info("Processing 'tanganke/stanford_cars' for car embeddings")
    car_embeddings, car_img_txt_pairs = process_dataset(
        model, modelprocessor, "tanganke/stanford_cars", "car", 50
    )

    # Show the first cat and car images and captions
    logging.info(f"First cat caption: {cat_img_txt_pairs[0]['caption']}")
    cat_img_txt_pairs[0]['pil_img'].show()

    logging.info(f"First car caption: {car_img_txt_pairs[0]['caption']}")
    car_img_txt_pairs[0]['pil_img'].show()

    # Combine cat and car embeddings and perform dimensionality reduction
    all_embeddings = np.concatenate([cat_embeddings, car_embeddings])
    labels = ['cat'] * len(cat_embeddings) + ['car'] * len(car_embeddings)

    logging.info("Performing dimensionality reduction for combined embeddings")
    reduced_dim_emb = dimensionality_reduction(all_embeddings, labels)

    # Plot the centroids against the cluster
    fig, ax = plt.subplots(figsize=(8, 6))  # Set figsize

    sns.set_style("whitegrid", {'axes.grid': False})
    sns.scatterplot(data=reduced_dim_emb,
                    x=reduced_dim_emb['X'],
                    y=reduced_dim_emb['Y'],
                    hue='label',
                    palette='bright')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.title('Scatter plot of images of cats and cars using UMAP')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()




    logging.info("Process completed successfully")



# Run the main function
if __name__ == "__main__":
    reduced_dim_emb = main()
