import csv
import itertools
import os
import time

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


def find_duplicate_images(folder, threshold=0.9):
    start_time = time.time()
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor
    processor = AutoImageProcessor.from_pretrained(
        "nomic-ai/nomic-embed-vision-v1",
    )
    vision_model = AutoModel.from_pretrained(
        "nomic-ai/nomic-embed-vision-v1",
        trust_remote_code=True,
    ).to(device)
    vision_model.eval()

    # Get all image paths
    image_paths = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Process images and get embeddings
    embeddings = []
    valid_images = []

    for img_path in tqdm(
        image_paths,
        desc=f"Processing images in {os.path.basename(folder)}",
    ):
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = vision_model(**inputs)
                img_emb = outputs.last_hidden_state
                embedding = F.normalize(img_emb[:, 0], p=2, dim=1)

            embeddings.append(
                embedding.cpu(),
            )  # Store on CPU to save GPU memory
            valid_images.append(img_path)
        except Exception as e:
            print(f"Skipping {img_path} due to error: {str(e)}")

    if not embeddings:
        return []

    # Convert to tensor and move to GPU
    embeddings_tensor = torch.cat(embeddings, dim=0).to(device)

    # Compute similarity matrix for all pairs
    similarity_matrix = torch.mm(embeddings_tensor, embeddings_tensor.T)

    # Find matches above threshold, excluding self-comparisons
    duplicates = []
    num_images = len(valid_images)

    # Only consider upper triangle of similarity matrix to avoid duplicate pairs
    for i in range(num_images):
        for j in range(i + 1, num_images):
            if similarity_matrix[i, j] >= threshold:
                duplicates.append((valid_images[i], valid_images[j]))

    print(f"\nExecution Time: {time.time() - start_time:.2f} seconds")
    return duplicates


if __name__ == "__main__":
    folder = "/home/joko/Desktop/dedup/sea-vl-image-collection-main/data"  # Replace with your folder path
    threshold = 0.94

    duplicates = find_duplicate_images(folder, threshold)

    # Print results
    print(f"\nFound {len(duplicates)} duplicate pairs:")
    for pair in duplicates:
        print(f"Reference: {pair[0]}, Comparison: {pair[1]}")

    # Save to CSV
    output_file = f"duplicate_pairs_threshold_{threshold}.csv"
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Reference", "Comparison"])
        writer.writerows(duplicates)

    print(f"\nResults saved to {output_file}")
