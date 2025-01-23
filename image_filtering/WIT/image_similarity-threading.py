import argparse
import asyncio
import logging
import multiprocessing
import os
import sys
import time

from functools import partial
from pathlib import Path
from queue import Queue
from threading import Thread

import datasets
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer

import requests
import pickle
from PIL import Image
from glob import glob
from pathlib import Path

from tqdm import tqdm


Image.MAX_IMAGE_PIXELS = None


def query_image(image=None, image_url=None, image_name=None, cache_dir=None, min_size=None):
    headers = {'User-Agent': 'CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org)'}
    assert image is not None or image_url is not None
    cache_path = os.path.join(cache_dir, image_name)
    image_data = {'name': image_name}
    try:
        if image is None:
            if os.path.isfile(cache_path):
                image = Image.open(cache_path)
            else:
                req = requests.get(image_url, headers=headers, stream=True, timeout=20)
                req.raise_for_status()
                image = Image.open(req.raw)
        image = image.convert('RGB')
        image_data['image'] = image
        if image is not None:
            if min_size is None or min(image.size[0], image.size[1]) >= min_size:
                if cache_dir is not None:
                    image.save(cache_path)
                    image_data['cache_path'] = cache_path
            else:
                logging.warning('[WARNING] image {} is too small ({} x {})\n'.format(
                    image_url, image.size[0], image.size[1]))
        else:
            logging.warning('[WARNING] failed to open {}\n'.format(image_url))
    except requests.exceptions.ConnectionError as e:
        logging.warning('[WARNING] Failed to download {}.\nConnection Error: {}\n'.format(image_url, e))
    except requests.exceptions.HTTPError as e:
        logging.warning('[WARNING] Failed to download {}.\nHTTP Error: {}\n'.format(image_url, e))
    except Exception as e:
        logging.warning('[WARNING] failed to download {}.\nError: {}\n'.format(image_url, e))

    return image_data


def save(data_list, image_dir=None, embed_dir=None):
    for image_data in data_list:
        try:
            print('Saving image under the path {}.'.format(image_data['name']), flush=True)
            image = image_data['image']
            embedding = image_data['embedding']
            
            if embed_dir:
                print('> Saving the embedding vector of {}.'.format(image_data['name']), flush=True)
                name, _ext = os.path.splitext(image_data['name'])
                embed_path = os.path.join(embed_dir, name + '.th')
                torch.save(embedding, embed_path)

            
            if image_dir:
                print('> Saving the image file of {}.'.format(image_data['name']), flush=True)
                outpath = os.path.join(image_dir, image_data['name'])
                image.save(outpath)
                print('> Removing the image file of {}.'.format(image_data['name']), flush=True)
                os.remove(image_data['cache_path'])
            print('> Saved {}.'.format(image_data['name']), flush=True)
        except Exception as e:
            print('> Failed to save {}: {}'.format(image_data['name'], e), flush=True)
        # for k, v in image_data.items():
        #     del image_data[k]
        # del image_data
        # print('> Freed all memory of {}.'.format(image_data['name']), flush=True)
    del data_list


def get_embedding(queue, model, save_fn=None, ref_embeddings=None, threshold=0):
    print(f'== Starting embedding worker', flush=True)
    while True:
        print(f'Waiting to embed. Queue size: {queue.qsize()}.', flush=True)
        items = queue.get()
        started_at = time.monotonic()

        # filter empty images
        items = [i for i in items if 'image' in i and i['image'] is not None]
        images = [i['image'] for i in items]
        embeddings = model.encode(images)
        assert len(embeddings) == len(images), "Embedding length is different!"
        for item, embedding in zip(items, embeddings):
            item['embedding'] = embedding
        items = [i for i in items if i['embedding'] is not None and i['embedding'].shape[-1] > 1]
        embeddings = [i['embedding'] for i in items]
        if len(embeddings) == 0:
            queue.task_done()
        else:
            embeddings = np.stack(embeddings)
            end_embed = time.monotonic()
            print(f'Finished embedding {len(images)} images in {end_embed - started_at} seconds', flush=True)
            
            # compare scraped images to CVQA and SEA-VQA
            if ref_embeddings is not None:
                try:
                    sim_scores = model.similarity(ref_embeddings, embeddings).mean(dim=0)
                except Exception as e:
                    print('ERROR:{}\nRef size:{}\nEmbedding size:{}'.format(e, ref_embeddings.shape, [e.shape for e in embeddings]), flush=True)
                    import sys; sys.exit(1)
                sim_scores = sim_scores.tolist()
                assert len(sim_scores) == len(embeddings)
                filtered_items = []
                for i_idx, item in enumerate(items):
                    sim_score = sim_scores[i_idx]
                    if sim_score > threshold:
                        filtered_item = item
                        filtered_item['name'] = 's{}_{}'.format(round(sim_score * 100, 2), item['name'])
                        assert isinstance(filtered_item, dict)
                        filtered_items.append(filtered_item)
                items = filtered_items
            print(f'Measuring similarity of {len(images)} images in {time.monotonic() - end_embed} seconds', flush=True)
            end_embed = time.monotonic()

            save_fn(items)
            print(f'Saving {len(images)} images took {time.monotonic() - end_embed} seconds', flush=True)

            queue.task_done()


def process_row(row, idx, queue, batch, query_fn, image_col=None, image_url_col=None,
                   image_name_col=None, image_dir=None, embed_dir=None, last=False):
    image_url = row[image_url_col]
    ori_image_name, ext = os.path.splitext(os.path.basename(image_url))
    if ext is None or len(ext) < 1:
        ext = '.jpg'
    elif ext == '.jpge':
        ext = '.jpg'
    if image_name_col is not None:
        image_basename = row[image_name_col].replace(" ", "_")
    else:
        image_basename = ori_image_name
    idx_basename = 'i{}_{}'.format(idx, image_basename)
    image_name = idx_basename + ext

    embed_path = os.path.join(embed_dir, idx_basename + '.th')
    outpath = os.path.join(image_dir, image_name)
    print(f'Querying {idx}. Queue size: {queue.qsize()}. Batch size: {len(batch)}', flush=True)
    if os.path.isfile(embed_path) and os.path.isfile(outpath):
        return
    elif image_col is not None:
        image_data = query_fn(image=row[image_col], image_name=image_name)
    else:
        image_data = query_fn(image_url=image_url, image_name=image_name)
    print(f'> Finished querying {idx}.', flush=True)
    
    batch.append(image_data)

    if queue.qsize() == 0 or last or len(batch) >= args.batch_size:
        print(f'Submitting to queue at {idx}. Queue size: {queue.qsize()}. Batch size: {len(batch)}', flush=True)
        queue.put([b for b in batch])
        batch.clear()


def producer(queue, query_fn, dataset, image_col=None, image_url_col=None,
                   image_name_col=None, image_dir=None, embed_dir=None, start_idx=1, data_len=None):
    batch = []
    assert image_col is not None or image_url_col is not None
    process_fn = partial(process_row, queue=queue, batch=batch, query_fn=query_fn,
                         image_col=image_col, image_url_col=image_url_col,
                         image_name_col=image_name_col, image_dir=image_dir, embed_dir=embed_dir)
    if isinstance(dataset, pd.DataFrame):
        iterator = dataset.iterrows()
    else:
        iterator = enumerate(dataset)
    
    if data_len is None:
        data_len = len(dataset)
    for idx, row in iterator:
        cur_idx = start_idx + idx
        print(f'[{cur_idx}/{data_len}]. Current queue size: {queue.qsize()}', flush=True)
        process_fn(row, cur_idx, last=(idx == len(dataset) - 1))
    
    queue.join()


def load_embeddings(paths):
    embeddings = []
    for path in paths:
        # load all files in dir
        embedding_files = [os.path.join(path, f) for f in os.listdir(path) \
                           if os.path.isfile(os.path.join(path, f))]
        for embedding_file in embedding_files:
            embeddings.append(torch.load(embedding_file))
    return np.stack(embeddings)


def main(args):
    queue = Queue(maxsize=32)
    model = SentenceTransformer("sentence-transformers/clip-ViT-B-32")
    # model = model.to('cuda')
    cache_dir = os.path.join(args.cache_dir, args.action)
    os.makedirs(cache_dir, exist_ok=True)
    image_dir = os.path.join(args.image_dir, args.action)
    os.makedirs(image_dir, exist_ok=True)
    embed_dir = os.path.join(args.embed_dir, args.action)
    os.makedirs(embed_dir, exist_ok=True)

    image_col = None
    if args.action.lower() == 'cvqa':
        dataset = load_dataset('afaji/cvqa')
        cvqa_sea_subsets = [
            "('Indonesian', 'Indonesia')",
            "('Javanese', 'Indonesia')",
            "('Minangkabau', 'Indonesia')",
            "('Sundanese', 'Indonesia')",
            "('Malay', 'Malaysia')",
            "('Chinese', 'Singapore')",
            "('Filipino', 'Philippines')",
        ]
        dataset = dataset['test'].filter(lambda x: x['Subset'] in cvqa_sea_subsets)
        image_url_col = 'Image Source'
        image_col = 'image'
        image_name_col = 'ID'
        ref_embeddings = None

    elif args.action.lower() == 'sea-vqa':
        datasets = []
        for split in ["cambodia", "indonesia", "laos", "malaysia", "philippines", "singapore", "thailand", "vietnam"]:
            datasets.append(load_dataset('wit543/sea-vqa', split=split))
        dataset = concatenate_datasets(datasets)
        image_url_col = 'image_path'
        image_name_col = 'question'
        ref_embeddings = None
    
    elif args.action.lower() in ['wit']:
        datasets = []
        for split in ["test", "validation", "train"]:
            datasets.append(load_dataset("SEACrowd/wit", name="wit_source", trust_remote_code=True, split=split))
        dataset = concatenate_datasets(datasets).sort('image_url')
        image_url_col = 'image_url'
        image_name_col = 'page_title'
        ref_embeddings = load_embeddings([os.path.join(args.embed_dir, f) for f in ['cvqa', 'sea-vqa']])
        
    else:
        raise ValueError("Unrecognized action {}. Please choose between 'cvqa', 'sea-vqa', and 'wit'.")
    
    query_fn = partial(query_image, cache_dir=cache_dir, min_size=50)
    save_fn = partial(save, image_dir=image_dir, embed_dir=embed_dir)
    
    started_at = time.monotonic()
    worker = Thread(target=get_embedding, args=(queue, model, save_fn, ref_embeddings), daemon=True)
    worker.start()
    threads = []
    data_len = len(dataset)
    thread_size = -(data_len // -args.num_thread) # ceil division
    for i in range(args.num_thread):
        start_idx = i * thread_size
        end_idx = min((i + 1) * thread_size, data_len)
        cur_dataset = dataset.select(range(start_idx, end_idx))
        t = Thread(target=producer, args=(queue, query_fn, cur_dataset, image_col, image_url_col,
                    image_name_col, image_dir, embed_dir, (start_idx + 1), data_len))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    exec_time = time.monotonic() - started_at
    print('Finished in {} seconds'.format(exec_time))


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help="")
    parser.add_argument('--embed_dir', type=str, required=True, help="")
    parser.add_argument('--action', type=str, required=True)
    parser.add_argument('--data_path', type=str, default=None, help="Only for WIT")
    parser.add_argument('--cache_dir', type=str, default='.cache', help="")
    parser.add_argument('--num_thread', type=int, default=1, help="")
    parser.add_argument('--min_size', type=int, default=50, help="")
    parser.add_argument('--batch_size', type=int, default=32, help="")
    parser.add_argument('--serialize', default=False, action='store_true', help='train the model')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    main(args)
