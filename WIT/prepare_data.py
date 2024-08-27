#%%
import os
import sys
import logging
# from tqdm import tqdm
from collections import Counter
from itertools import chain
from datasets import load_dataset
import pandas as pd
import seacrowd as sc
# import tldextract
import wikipediaapi

#%%
country_list = {'kh','th','la','id','mm','ph','vn','tl','bn','my','sg'}
SEA_FLAGS = [
    "Southeast Asia", "Burmese", "Cambodian", "Indonesian", "Laotian", "Malay", "Philippine", "Khmer", "Thai", "Vietnamese", "Bruneian", "East Timor", "Singapore",
    "Brunei", "Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar", "Philippines", "Thailand", "Vietnam", "Timor Leste",
]
splits = os.getenv("SEAVL_WIT_SPLITS", "test validation train").split(' ')
# --- wit_source count ---
# test: 7320
# validation: 8393
# train: 1327201    =>  chunk to 201 * 6603 (remainder = 2)
wiki_wiki = wikipediaapi.Wikipedia('SEA VL', 'en')

#%%
def _get_sea_flags_in_categories(wiki_page):
    return list(filter(
        lambda flag: any(
            flag in cat_title
            for cat_title in wiki_page.categories
        ),
        SEA_FLAGS
    ))


def _is_geoloc_in_sea(wiki_page):
    raise NotImplementedError


def izit_sea_related(i):
    wiki_page = wiki_wiki.page(i["page_title"])

    if not wiki_page:
        return False
    if not _get_sea_flags_in_categories(wiki_page):
        return False
    # if not _is_geoloc_in_sea(wiki_page):
    #     return False

    return True


def get_sea_related_stats(i):
    wiki_page = wiki_wiki.page(i["page_title"])
    if not wiki_page:
        return {
            "no_page": True,
            "no_sea_flags": False,
            "sea_related": [],
        }
    sea_flags_in_categories = _get_sea_flags_in_categories(wiki_page)
    return {
        "no_page": False,
        "no_sea_flags": (False if sea_flags_in_categories else True),
        "sea_related": sea_flags_in_categories,
    }


#%%
def main():
    if int(os.getenv("SEAVL_WIT_DONT_USE_SEACROWD_PKG", 0)):
        dset = load_dataset("SEACrowd/wit", name="wit_source")
    else:
        dset = sc.load_dataset_by_config_name("wit_source")

    # %%
    for split in splits:
        print("===== Processing", split, "=====")

        # Features of `wit_source`
        # sea_related_dict = {
        #     "language": [],
        #     "page_url": [],
        #     "page_title": [],
        #     "section_title": [],
        #     "hierarchical_section_title": [],
        #     "image_url": [],
        #     "caption_reference_description": [],
        #     "caption_attribution_description": [],
        #     "caption_alt_text_description": [],
        #     "mime_type": [],
        #     "original_height": [],
        #     "original_width": [],
        #     "is_main_image": [],
        #     "attribution_passes_lang_id": [],
        #     "page_changed_recently": [],
        #     "context_page_description": [],
        # }

        # ===== meta =====
        ds = dset[split].remove_columns(["context_section_description"])  # following the previous commit
        if os.getenv("SEAVL_WIT_IDX_ST", None) is not None:
            assert os.getenv("SEAVL_WIT_IDX_ED", None) is not None, "Please set env `SEAVL_WIT_IDX_ED` !!"
            idx_st, idx_ed = int(os.environ["SEAVL_WIT_IDX_ST"]), min(int(os.environ["SEAVL_WIT_IDX_ED"]), len(ds))
            ds = ds.select(range(idx_st, idx_ed))
            fp_out = f"./out/wit_sea_{split}_idx_{idx_st}_to_{idx_ed}"
        else:
            idx_st, idx_ed = 0, len(ds)
            fp_out = f"./out/wit_sea_{split}_all_idx_{idx_st}_to_{idx_ed}"

        logging.getLogger().handlers.clear()
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.getLogger().addHandler(logging.FileHandler(f"{fp_out}.txt"))
        logging.getLogger().setLevel(21)

        logging.log(21, f"subsets: dset['{split}'].select(range({idx_st}, {idx_ed}))")
        logging.log(21, f"data count: {len(ds)}")
        logging.log(21, f"uniq title: {len(set(ds['page_title']))}")

        # ===== terra =====
        num_proc = int(os.getenv("SEAVL_WIT_PROC", 1))  # multi-proc is currently not working :')
        batch_size = int(os.getenv("SEAVL_WIT_BS", 8))
        ds_stat = ds.map(
            get_sea_related_stats, num_proc=num_proc, batch_size=batch_size,
        )
        logging.log(21, f"no_page #: {sum(ds_stat['no_page'])}")
        logging.log(21, f"no_sea_flags #: {sum(ds_stat['no_sea_flags'])}")
        logging.log(21, f"sea_related (flag multi-match): {Counter(chain(*ds_stat['sea_related']))}")

        # ===== ozma =====
        sea_related_ds = ds_stat.filter(
            lambda i: not i["no_page"] and not i["no_sea_flags"],  # izit_sea_related,
            num_proc=num_proc, batch_size=batch_size,
        ).remove_columns(["no_page", "no_sea_flags", "sea_related"])
        logging.log(21, f"sea_related #: {len(sea_related_ds)}")
        pd.DataFrame(sea_related_ds).to_csv(f"{fp_out}.csv")


#%%
if __name__ == "__main__":
    main()
