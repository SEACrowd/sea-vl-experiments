# Image Deduplicator

A Python library for finding duplicate and similar images using perceptual hashing techniques.

The proposed image deduplication methodology can be described as follows:

1. **Hashing**  
   Each image I is transformed into a hash h(I) using a hash function H:

   h(I) = H(I)

   where H is a hash function, such as average hash, phash, dhash, etc.

2. **Hamming Distance**  
   The similarity between two images I₁ and I₂ is determined by the Hamming distance D(h(I₁), h(I₂)) between their hashes:

   D(h(I₁), h(I₂)) = Hamming distance(h(I₁), h(I₂)) = Σ|hᵢ(I₁) - hᵢ(I₂)|

   where hᵢ(I) is the i-th bit of the hash h(I), and n is the length of the hash.

3. **Thresholding for Grouping**  
   If the Hamming distance between two images is below a predefined threshold T, the images are considered similar:

   D(h(I₁), h(I₂)) ≤ T → Images I₁ and I₂ are grouped together

## Setup

### Prerequisites
- Python 3.8+
- pip
- git

### Installation

1. Clone the repository.

2. Create and activate virtual environment:
```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command-Line Arguments
```
python main.py path/to/folder --hash-type phash --threshold 8 --csv path/to/output_file.csv --print --remove
```

`folder` (required): Path to the folder containing images to check for duplicates.

`--hash-type`: Type of hash function to use. Options are:
   - `ahash`: Average hash
   - `phash`: Perceptual hash (default)
   - `dhash`: Difference hash
   - `whash-haar`: Wavelet hash using Haar wavelets
   - `whash-db4`: Wavelet hash using DB4
   - `colorhash`: Color-based hash

`--threshold`: Integer threshold for hash difference to consider images as similar. Default is 8.

`--csv`: Path to save the CSV file with duplicate image groups. Default is `None`.

`--remove`: If set, duplicate images will be removed, keeping only one copy.

`--print`: If set, the duplicate image groups will be printed to the console.

`--all`: If set, it will print, save, and remove duplicate image groups in a single command.

### Example Usage in Terminal:
```
python main.py path/to/images --hash-type phash --threshold 8 --csv duplicates.csv --print
```

This will find duplicate images in the specified folder, print the groups of duplicates, and save the results in a CSV file.
