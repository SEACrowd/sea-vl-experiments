# SEA-VL Image Generation Module

## Installation
### Clone the project:
```bash
git clone https://github.com/SEACrowd/sea-vl-experiments.git
```
### Create a new environment and install dependencies:
```bash
conda create --name image_generation python==3.10.14
conda activate image_generation
cd sea-vl-experiments/image_gen
pip install -r requirements.txt
```
### Clone required dependencies:
```bash
# For Janus-Pro:
git clone https://github.com/deepseek-ai/Janus.git
cd Janus
pip install -e .
```
    
## How to Use 
### Run the Corresponding Script
To generate specific outputs, use the following commands:
#### For **Flux (.1-dev)** Models:
```bash
cd sea-vl-experiments/image_gen
python flux/gen_cultures.py      # Generate cultures
python flux/gen_food.py         # Generate food
python flux/gen_landmarks.py    # Generate landmarks
```
#### For **Stable Diffusion 2 (SD2)** Models:
```bash
cd sea-vl-experiments/image_gen
python sd2/gen_cultures.py      # Generate cultures
python sd2/gen_food.py         # Generate food
python sd2/gen_landmarks.py    # Generate landmarks
```
#### For **Stable Diffusion 3.5 (SD3.5 Large)** Models:
```bash
cd sea-vl-experiments/image_gen
python sd3.5/gen_cultures.py      # Generate cultures
python sd3.5/gen_food.py         # Generate food
python sd3.5/gen_landmarks.py    # Generate landmarks
```
#### For **Janus Pro (7B) Models** Models:
```bash
cd sea-vl-experiments/image_gen
python janus_pro/gen_cultures.py      # Generate cultures
python janus_pro/gen_food.py         # Generate food
python janus_pro/gen_landmarks.py    # Generate landmarks
```


