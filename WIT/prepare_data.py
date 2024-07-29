from tqdm import tqdm
from collections import Counter

import datasets
import pandas as pd
import seacrowd as sc
import tldextract
import wikipediaapi

country_list = {'kh','th','la','id','mm','ph','vn','tl','bn','my','sg'}
SEA_FLAGS = [
    "Southeast Asia", "Burmese", "Cambodian", "Indonesian", "Laotian", "Malay", "Philippine", "Khmer", "Thai", "Vietnamese", "Bruneian", "East Timor", "Singapore",
    "Brunei", "Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar", "Philippines", "Thailand", "Vietnam", "Timor Leste",
]
splits = ['test', 'validation', 'train']
wiki_wiki = wikipediaapi.Wikipedia('SEA VL', 'en')

def main():
    dset = sc.load_dataset_by_config_name("wit_source")

    for split in splits:

        print("===== Processing", split, "=====")

        sea_related_dict = {
            "language": [],
            "page_url": [],
            "page_title": [],
            "section_title": [],
            "hierarchical_section_title": [],
            "image_url": [],
            "caption_reference_description": [],
            "caption_attribution_description": [],
            "caption_alt_text_description": [],
            "mime_type": [],
            "image/jpeg": [],
            "original_height": [],
            "original_width": [],
            "is_main_image": [],
            "attribution_passes_lang_id": [],
            "page_changed_recently": [],
            "context_page_description": [],
        }
        for i in tqdm(range(len(dset[split]))):
            page_title = dset[split][i]["page_title"]
            wiki_page = wiki_wiki.page(page_title)

            if wiki_page:
                is_sea_related = False
                for cat_title in wiki_page.categories:
                    if any(flag in cat_title for flag in SEA_FLAGS):
                        is_sea_related = True
                        break

                if is_sea_related:
                    sea_related_dict["language"].append(dset[split][i]["language"])
                    sea_related_dict["page_url"].append(dset[split][i]["page_url"])
                    sea_related_dict["page_title"].append(dset[split][i]["page_title"])
                    sea_related_dict["section_title"].append(dset[split][i]["section_title"])
                    sea_related_dict["hierarchical_section_title"].append(dset[split][i]["hierarchical_section_title"])
                    sea_related_dict["image_url"].append(dset[split][i]["image_url"])
                    sea_related_dict["caption_reference_description"].append(dset[split][i]["caption_reference_description"])
                    sea_related_dict["caption_attribution_description"].append(dset[split][i]["caption_attribution_description"])
                    sea_related_dict["caption_alt_text_description"].append(dset[split][i]["caption_alt_text_description"])
                    sea_related_dict["mime_type"].append(dset[split][i]["mime_type"])
                    sea_related_dict["original_height"].append(dset[split][i]["original_height"])
                    sea_related_dict["original_width"].append(dset[split][i]["original_width"])
                    sea_related_dict["is_main_image"].append(dset[split][i]["is_main_image"])
                    sea_related_dict["attribution_passes_lang_id"].append(dset[split][i]["attribution_passes_lang_id"])
                    sea_related_dict["page_changed_recently"].append(dset[split][i]["page_changed_recently"])
                    sea_related_dict["context_page_description"].append(dset[split][i]["context_page_description"])
            
        sea_related_df = pd.DataFrame(sea_related_dict)
        sea_related_df.to_csv(
            f"./out/wit_sea_{split}_{len(sea_related_dict["image_url"])}_from_{len(dset[split])}.csv")

if __name__ == "__main__":
    main()