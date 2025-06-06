import os, sys
import pandas as pd
import numpy as np

from datasets import load_dataset, load_from_disk, concatenate_datasets
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

import os
import requests
from tqdm import tqdm
import pickle
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from glob import glob
import time

from itertools import chain

import warnings

warnings.filterwarnings('ignore')

if os.getenv('SEAVL_COYO_PREDOWNLOAD_ONLY', None):
    print("... Download images then exit ...")

else:
    print("Loading model ..")
    model = SentenceTransformer("sentence-transformers/clip-ViT-B-32", device='cuda')
    print("Model loaded.")


####################
# SEA-VQA Emb.
####################
fp_ds_emb = "sea_vqa.pkl"
if not os.path.isfile(fp_ds_emb):
    print(f"Generating Encoding: {fp_ds_emb}.")
    sea_vqa_dataset = load_dataset('wit543/sea-vqa')
    sea_vqa_images_filt = []
    sea_vqa_images_embed = []
    sea_vqa_caption = []
    sea_vqa_culture = []
    for key in sea_vqa_dataset.keys():
        for row in tqdm(sea_vqa_dataset[key]):
            try:
                img_opened = Image.open(requests.get(row['image_path'], stream=True).raw)
                sea_vqa_images_embed.append(model.encode(img_opened))
                sea_vqa_images_filt.append(img_opened)
                if row['correct_answer'] in ['a', 'b', 'c', 'd']:
                    sea_vqa_caption.append(row['question'] + " " + row['choice_' + row['correct_answer']])
                else:
                    sea_vqa_caption.append(row['question'])
                sea_vqa_culture.append(key)
            except:
                print(row)
    pickle.dump((sea_vqa_images_filt, sea_vqa_images_embed, sea_vqa_caption, sea_vqa_culture), open(fp_ds_emb, 'wb'))

if not os.getenv('SEAVL_COYO_PREDOWNLOAD_ONLY', None):
    print(f"Loading Encoding: {fp_ds_emb}.")
    (sea_vqa_images_filt, sea_vqa_images_embed, sea_vqa_caption, sea_vqa_culture) = pickle.load(open(fp_ds_emb, 'rb'))


####################
# CVQA Emb.
####################
# print("CVQA encoding")
fp_ds_emb = "cvqa_category_all.pkl"
if not os.path.isfile(fp_ds_emb):
    print(f"Generating Encoding: {fp_ds_emb}")
    cvqa_dataset = load_dataset('afaji/cvqa')
    cvqa_sea_subsets = [
        "('Indonesian', 'Indonesia')",
        "('Malay', 'Malaysia')",
        "('Javanese', 'Indonesia')",
        "('Minangkabau', 'Indonesia')",
        "('Sundanese', 'Indonesia')",
        "('Chinese', 'Singapore')"
    ]
    cvqa_dataset_filt = cvqa_dataset['test'].filter(lambda x: str(x['Subset']) in cvqa_sea_subsets, num_proc=8)

    #cvqa_images_filt = []
    #cvqa_images_embed = []
    #cvqa_caption = []
    #cvqa_culture = []
    #for row in tqdm(cvqa_dataset_filt):
    #    try:
    #        cvqa_images_embed.append(model.encode(row['image']))
    #        cvqa_images_filt.append(row['image'])
    #        cvqa_caption.append(row['Translated Question'] + " " + ', '.join(row['Translated Options']))
    #        cvqa_culture.append(eval(row['Subset'])[0])
    #    except:
    #        print(row)

    cvqa_images_embed = []
    cvqa_caption = []
    cvqa_culture = []
    cvqa_category = []
    for row in tqdm(cvqa_dataset['test']):
        try:
            cvqa_images_embed.append(model.encode(row['image']))
            cvqa_caption.append(row['Translated Question'] + " " + ', '.join(row['Translated Options']))
            cvqa_culture.append(eval(row['Subset'])[0])
            cvqa_category.append(row['Category'])
        except:
            print(row)
    pickle.dump((cvqa_images_embed, cvqa_caption, cvqa_culture, cvqa_category), open(fp_ds_emb, 'wb'))

if not os.getenv('SEAVL_COYO_PREDOWNLOAD_ONLY', None):
    print(f"Loading Encoding: {fp_ds_emb}.")
    (cvqa_images_embed, cvqa_caption, cvqa_culture, cvqa_category) = pickle.load(open(fp_ds_emb, 'rb'))


####################
# Coyo Emb.
####################
def _invalid_images_as_none_(image_url):
    try:
        im = Image.open(requests.get(image_url, stream=True, timeout=8).raw).convert('RGB')
        return im if _filter_notna_nor_tiny_(im) else None
    except Exception:
        return None

def invalid_images_as_none(cur_batch):
    images = []
    for image_url in cur_batch["url"]:
        images.append(_invalid_images_as_none_(image_url))
    cur_batch["image"] = images
    return cur_batch


def invalid_images_as_none_single(cur_item):
    cur_item['image'] = _invalid_images_as_none_(cur_item['url'])
    return cur_item


def _filter_notna_nor_tiny_(img):
    return (img is not None) and (img.size[0] >= 50) and (img.size[1] >= 50)


def filter_notna_nor_tiny(d):
    return _filter_notna_nor_tiny_(d['image'])


def chunk_into(n, k, i):
    sz = n // k
    st = (i * sz) + (i if i < (n % k) else n % k)
    ed = st + sz + (1 if i < (n % k) else 0)
    return st, ed


def retrieve_idx(_indices):
    ret = []
    for rg in (list(map(int, rg.split('-'))) for rg in _indices.split(',')):
        ret.append(range(rg[0], rg[-1] + 1))
    return ret


bs = int(os.getenv('SEAVL_COYO_BATCH_SIZE', '512'))    # for GPU
num_proc = int(os.getenv('SEAVL_COYO_NUM_WORKERS', '0')) or None    # for CPU
print(f"batch size: {bs};    num_proc: {num_proc};")

coyo_dset_range_str = os.getenv('SEAVL_COYO_DSET_SELECT', '0-746972268')    # inclusive values. COYO trainset count is 746,972,269
coyo_dset_range = retrieve_idx(coyo_dset_range_str)[0]
print(f"COYO dataset range: {coyo_dset_range}")

batch_cnt = int(os.getenv('SEAVL_COYO_SPLIT_CNT', '1'))
print(f"split count: {batch_cnt:,}")

batch_indices_str = os.getenv('SEAVL_COYO_SPLIT_IDX', '0')
print(f"split indices (0-based): {batch_indices_str}.")
batch_indices = retrieve_idx(batch_indices_str)

dsets, dset_coyo = [], None
print(f"Loading COYO dataset (train)")
for batch_idx in chain(*batch_indices):
    current_dset_fn = f"coyo_rg{coyo_dset_range_str}_{batch_idx}of{batch_cnt}"
    print(f"dset: {current_dset_fn}")

    if not os.path.isdir(f"preload_dset/{current_dset_fn}"):
        print(f"Preload dset not found. Preparing dset ..")
        if dset_coyo is None:
            dset_coyo = load_dataset("kakaobrain/coyo-700m", split='train', num_proc=num_proc).select(coyo_dset_range)
        dset = dset_coyo
        print(f"coyo dataset loaded. dset count: {len(dset):,}")
        cur_st, cur_ed = chunk_into(len(dset), batch_cnt, batch_idx)
        print("slice the dataset according to chunk-split")
        dset = dset.select(range(cur_st, cur_ed))
        print(f"    > i.e. dset.select(range({cur_st:,}, {cur_ed:,}))")

        if os.getenv('SEAVL_COYO_PREDOWNLOAD_ONLY', None):
            dset = dset.map(
                # invalid_images_as_none, batched=True, batch_size=bs,
                invalid_images_as_none_single,
                num_proc=num_proc, desc=current_dset_fn,
                remove_columns=(set(dset.features.keys()) - {'id', 'url', 'text'}),
            )
            dset = dset.filter(filter_notna_nor_tiny, num_proc=num_proc, desc=current_dset_fn,)
            dset.save_to_disk(f"preload_dset/{current_dset_fn}")

    else:
        dset = load_from_disk(f"preload_dset/{current_dset_fn}")
        print(f"dataset loaded from disk")

    dsets.append(dset)

if os.getenv('SEAVL_COYO_PREDOWNLOAD_ONLY', None):
    exit(0)
else:
    dset = concatenate_datasets(dsets)
    # .with_transform(invalid_images_as_none)
    # print(f"    > with_transform(invalid_images_as_none)")

print(f"Preparing dataloader ..")
loader = DataLoader(dset, batch_size=bs, num_workers=num_proc or 0, prefetch_factor=2 if num_proc else None, collate_fn=lambda x: {k: [row[k] for row in x] for k in x[0]})
print(f"batch count (dataloader): {len(loader):,}")

print("Start image encoding ..")
indices, imgs, coyo_images_filt, coyo_caption = [], [], [], []
current_run_fn = f"coyo_rg{coyo_dset_range_str}"
with open(f"output/{current_run_fn}.pkl", "wb") as opt:
    def flush():
        global indices, imgs, coyo_images_filt, coyo_caption
        img_embeds = list(model.encode(imgs, batch_size=bs))
        pickle.dump((indices, coyo_images_filt, img_embeds, coyo_caption,), opt)
        indices, imgs, coyo_images_filt, coyo_caption = [], [], [], []

    with tqdm(total=len(dset), desc=current_run_fn) as pbar:
        for batch in loader:
            for j, img in enumerate(batch['image']):
                if (img is not None) and (img.size[0] >= 50) and (img.size[1] >= 50):
                    # indices.append(pbar.n)
                    indices.append(batch['id'][j])
                    imgs.append(img)
                    coyo_images_filt.append(batch['url'][j])
                    coyo_caption.append(batch['text'][j])
                    if len(imgs) >= bs:
                        flush()
                pbar.update(1)
        if imgs:
            flush()

    # for img_emb in img_embeds:
    #     coyo_images_embed.append(img_emb)
# print(len(coyo_images_embed), len(coyo_images_filt), len(coyo_caption), flush=True)
# pickle.dump((coyo_images_filt, coyo_images_embed, coyo_caption, []), open('./coyo_full.pkl', 'wb'))
