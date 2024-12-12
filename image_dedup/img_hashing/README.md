# Image Deduplicator

A Python library for finding duplicate and similar images using perceptual hashing techniques.

The proposed image deduplication methodology can be described as follows:

1. **Hashing**:  
   Each image \( I \) is transformed into a hash \( h(I) \) using a hash function \( H \):
   \[
   h(I) = H(I)
   \]
   where \( H \) is a perceptual hash function, such as average hash, phash, dhash, etc.

2. **Hamming Distance**:  
   The similarity between two images \( I_1 \) and \( I_2 \) is determined by the Hamming distance \( D(h(I_1), h(I_2)) \) between their perceptual hashes:
   \[
   D(h(I_1), h(I_2)) = \text{Hamming distance}(h(I_1), h(I_2)) = \sum_{i=1}^n |h_i(I_1) - h_i(I_2)|
   \]
   where \( h_i(I) \) is the \( i \)-th bit of the perceptual hash \( h(I) \), and \( n \) is the length of the hash.

3. **Thresholding for Grouping**:  
   If the Hamming distance between two images is below a predefined threshold \( T \), the images are considered similar:
   \[
   D(h(I_1), h(I_2)) \leq T \quad \Rightarrow \quad \text{Images } I_1 \text{ and } I_2 \text{ are grouped together}
   \]

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
