# Nomic

A Python tool that efficiently detects duplicate images within a folder using the Nomic Vision embedding model.

## Overview

This script analyzes images using the `nomic-embed-vision-v1` model to generate high-quality visual embeddings, then compares these embeddings to find duplicate or highly similar images. It uses cosine similarity to measure the similarity between images and can be configured with a threshold to control how strict the matching should be.

## Features

- GPU acceleration for faster processing (falls back to CPU if GPU not available)
- Progress bar showing processing status
- Configurable similarity threshold
- CSV output with all detected duplicate pairs
- Works with common image formats (PNG, JPG, JPEG)

## Requirements

- Python 3.6+
- PyTorch
- Transformers
- Pillow
- tqdm
- einops

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/image-duplicate-finder.git
   cd image-duplicate-finder
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Edit the script to specify your image folder and desired threshold:
   ```python
   folder = "/path/to/your/image/folder"  # Replace with your folder path
   threshold = 0.94  # Adjust as needed (higher = more strict matching)
   ```

2. Run the script:
   ```
   python nomic.py
   ```

3. View results in the generated CSV file (`duplicate_pairs_threshold_X.XX.csv`).

## How It Works

1. The script loads each image from the specified folder
2. Images are processed through the Nomic Vision model to obtain embeddings
3. A similarity matrix is computed to compare all image pairs
4. Image pairs with similarity scores above the threshold are identified as duplicates
5. Results are saved to a CSV file with reference and comparison image paths

## Performance Considerations

- Processing time depends on the number of images and your hardware
- GPU acceleration significantly improves performance
- For large image collections, consider batch processing

## License

MIT

## Acknowledgements

This tool utilizes the [Nomic Embed Vision model](https://huggingface.co/nomic-ai/nomic-embed-vision-v1) developed by Nomic AI.