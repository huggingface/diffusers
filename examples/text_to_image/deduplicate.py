import torch
from PIL import Image
import open_clip
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def pairwise_cosine_distance(matrix):
    """
    Computes pairwise cosine distances for an n x m numpy matrix, where n is the number of embeddings
    and m is the embedding size.

    :param matrix: A numpy array of shape (n, m)
    :return: A numpy array of shape (n, n) containing pairwise cosine distances
    """

    # Normalize the matrix along the embedding size axis (axis 1)
    normalized_matrix = matrix / np.linalg.norm(matrix, axis=1)[:, np.newaxis]

    # Compute the pairwise cosine similarities
    similarities = np.matmul(normalized_matrix, normalized_matrix.T)

    # Clip the similarities to the range [-1, 1] to avoid potential floating point errors
    similarities = np.clip(similarities, -1, 1)

    # Get the upper triangular indices (excluding the diagonal)
    upper_tri_indices = np.triu_indices(similarities.shape[0], k=1)

    # Create a mask with True values only for the upper triangular part (excluding the diagonal)
    mask = np.zeros_like(similarities, dtype=bool)
    mask[upper_tri_indices] = True

    # Set the lower triangular part to zero, keeping only the upper triangular part
    similarities = np.where(mask, similarities, 0)

    return similarities

#add parser function
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, help='path to images directory', required=True)
    parser.add_argument('--histogram_name', type=str, help='name of histogram', required=True)
    return parser.parse_args()

def get_image_features(model, preprocess, image_dir, device):
    
    file_list = os.listdir(image_dir)

    all_vecs = []

    for img in tqdm(file_list):
        image = preprocess(Image.open(os.path.join(image_dir, img))).unsqueeze(0).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image)
            all_vecs.append(image_features.cpu().numpy())

    all_vecs = np.concatenate(all_vecs, axis=0)
    return all_vecs

if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Directory: ", args.images_dir)
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
    model = model.to(device)

    all_vecs = get_image_features(model, preprocess, "/scratch/mp5847/diffusers_generated_datasets/kilian_eng_multi/train", device)
    distances = pairwise_cosine_distance(all_vecs)
    distances = distances[np.triu_indices(distances.shape[0], k=1)]
    
    #plot histogram of distances
    plt.hist(distances, label='kilian')

    all_vecs = get_image_features(model, preprocess, "/scratch/mp5847/diffusers_generated_datasets/van_gogh_multi/train", device)
    distances = pairwise_cosine_distance(all_vecs)
    distances = distances[np.triu_indices(distances.shape[0], k=1)]
    
    #plot histogram of distances
    plt.hist(distances, label='van_gogh')
    plt.legend(loc='upper right')

    plt.show()
    # plt.savefig(args.histogram_name)
    plt.savefig("deduplication_histogram.png")
    
    
    
