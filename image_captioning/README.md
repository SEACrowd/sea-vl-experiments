# SEA-VL Image Captioning Module

## Installation
### Clone the project:
```bash
git clone https://github.com/SEACrowd/sea-vl-experiments.git
```
### Create a new environment and install dependencies:
```bash
conda create --name image_captioning python==3.10.14
conda activate image_captioning
cd sea-vl-experiments/image_captioning
pip install -r requirements.txt
```
### Clone required dependencies:
```bash
# For the Maya model:
git clone https://github.com/nahidalam/maya
# For the Pangea model:
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
```

## How to Use
### Configure the project:
> [!IMPORTANT]
> 1. Open `Config/settings.py` and update the following settings with absolute paths as needed:
>    - Replace `Maya_path` with the **absolute path** of the cloned Maya repository.
>    - Replace `LLaVA-NeXT_path` with the **absolute path** of the cloned LLaVA-NeXT repository.
>    - Replace `RESULTS_DIR` with the **absolute path** where you want to save the results.
>    - Adjust other settings based on your configuration, you can leave it as default also.
### Run the script:
```bash
cd sea-vl-experiments/image_captioning/src
python main.py
```
- Available arguments:
    - --dataset: Currently supports ["seavqa", "worldcuisines"]
    - --model: Currently supports ["maya", "paligemma2", "pangea", "qwenvl2"]
    - --prompt: Currently supports:
        - Location-Agnostic Prompts
            - en_location_agnostic_prompt
            - th_location_agnostic_prompt
            - ms_location_agnostic_prompt
            - tl_location_agnostic_prompt
            - id_location_agnostic_prompt
            - vi_location_agnostic_prompt
        - Location-Aware Prompts
            - en_location_aware_prompt
            - th_location_aware_prompt
            - ms_location_aware_prompt
            - tl_location_aware_prompt
            - id_location_aware_prompt
            - vi_location_aware_prompt 
    - Other generation parameters:
        > [!TIP]
        > - If you don't pass these parameters, the defaults (greedy decoding) specified in your configuration file will be used.
        - max_new_tokens 
        - do_sample
        - num_beams
        - use_cache
        - temperature
        - top_p
        - top_k 

