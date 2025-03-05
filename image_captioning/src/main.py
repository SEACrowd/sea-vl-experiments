import os
import sys
import argparse
import torch
from tqdm import tqdm
from utils.file_io import ResultSaver
from config.settings import Config

def change_working_directory(args):
    if args.model == "pangea":
        llava_path = Config.PANGEA_CONFIGURATION["LLaVA-NeXT_path"]
        sys.path.insert(0, llava_path)
        os.chdir(llava_path)
    elif args.model == "maya":
        maya_path = Config.MAYA_CONFIGURATION["Maya_path"]
        sys.path.insert(0, maya_path)
        os.chdir(maya_path)
    print("Working in:", os.getcwd())

def parse_arguments():
    parser = argparse.ArgumentParser(description="Multimodal Caption Generation Pipeline")
    
    # Required arguments
    parser.add_argument("--dataset", choices=["seavqa", "worldcuisines"], required=True)
    parser.add_argument("--model", choices=["maya", "paligemma2", "pangea", "qwenvl2"], required=True)
   
    
    # Generation parameters
    parser.add_argument("--no_use_cache", action="store_false", dest="use_cache",
                        default=Config.DEFAULT_GENERATION_CONFIG["use_cache"])
    parser.add_argument("--do_sample", action="store_true",
                        default=Config.DEFAULT_GENERATION_CONFIG["do_sample"])
    parser.add_argument("--max_new_tokens", type=int, default=Config.DEFAULT_GENERATION_CONFIG["max_new_tokens"])
    parser.add_argument("--num_beams", type=int, default=Config.DEFAULT_GENERATION_CONFIG["num_beams"])
    parser.add_argument("--temperature", type=float, default=Config.DEFAULT_GENERATION_CONFIG["temperature"])
    parser.add_argument("--top_p", type=float, default=Config.DEFAULT_GENERATION_CONFIG["top_p"])
    parser.add_argument("--top_k", type=float, default=Config.DEFAULT_GENERATION_CONFIG["top_k"])

    # Prompt parameters
    parser.add_argument(
        "--prompt", 
        choices=list(Config.PROMPT_CONFIGURATION.keys()), 
        default="en_location_agnostic_prompt"
    )
    

    parser.add_argument("--save_interval", type=int, default=10)
    
    return parser.parse_args()

def run_pipeline(args):
    from data_modules import SEAVQADataset, WorldCuisinesDataset
    from models import MayaModel, PaliGemma2Model, PangeaModel, QwenVL2Model
    dataset_map = {
        "seavqa": SEAVQADataset,
        "worldcuisines": WorldCuisinesDataset
    }
    
    model_map = {
        "maya": MayaModel,
        "paligemma2": PaliGemma2Model,
        "pangea": PangeaModel,
        "qwenvl2": QwenVL2Model
    }
    
    dataset = dataset_map[args.dataset]()
    model = model_map[args.model]()
    saver = ResultSaver()
    
    model.load()
    
    gen_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "num_beams": args.num_beams,
        "use_cache": args.use_cache,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }

    
    results = []
    try:
        dataset_list = list(dataset)
        result_filename = f"{args.model}_{args.dataset}_{args.prompt}_result"
        for idx, item in enumerate(tqdm(dataset_list, desc=f"Processing {args.dataset}")):
            prompt = Config.PROMPT_CONFIGURATION[args.prompt]
            if args.prompt == "en_location_aware_prompt":
                if isinstance(dataset, SEAVQADataset):
                    prompt = prompt.format(Location=item['country'])
                elif isinstance(dataset, WorldCuisinesDataset):
                    prompt = prompt.format(Location=item['data']['countries'])
                    
            try:
                caption = model.generate_caption(
                    image=item["image"],
                    prompt=prompt,
                    **gen_config
                )

                if isinstance(dataset, SEAVQADataset):
                    results.append({
                        "name":item['data']["culture_name"],
                        "country":item['country'],
                        "image_url":item['data']["image_path"],
                        "gt_caption":item['data']["gt_caption"],
                        "caption":caption
                    })
                elif isinstance(dataset, WorldCuisinesDataset):
                    results.append({
                        "name":item['data']['name'],
                        "country":item['data']['countries'],
                        "image_url":item['image_url'].replace("?download", ""),
                        "gt_caption":item['data']["text_description"],
                        "caption":caption
                    })
                
                if (idx + 1) % args.save_interval == 0:
                    saver.append_results(results, result_filename)
                    results = []
                    
            except Exception as e:
                print(f"Error processing item {idx}: {str(e)}")
                continue
                
    finally:
        if results:
            saver.append_results(results, result_filename)
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    args = parse_arguments()
    change_working_directory(args)
    run_pipeline(args)