
## Filtering Culturally-Relevant COYO Dataset

---

### Extracting the Image Embeddings

The file to run is `Coyo_extrtact_embed.py`.
<br>It has several useful environment variables:

>
> - `SEAVL_COYO_DSET_SELECT`
>   - range of coyo trainset to be processed. default: '0-746972268' 
> 
> 
> - `SEAVL_COYO_PREDOWNLOAD_ONLY`
>   - when set to 1, will only download the images then save the huggingface dataset to disk.
>   - otherwise, will open existing dataset or just directly download, then proceed in computing the embeddings.
>
> 
> - `SEAVL_COYO_SPLIT_CNT`
>   - Number of split chunks.
>
> 
> - `SEAVL_COYO_SPLIT_IDX`
>   - 0-based indices based on `SEAVL_COYO_SPLIT_CNT`.
>   - Accepts either single value and ranges. Multiple values splited by a comma.
> 
> 
> - `SEAVL_COYO_BATCH_SIZE`
>   - as name suggests.
>
>
> - `SEAVL_COYO_NUM_WORKERS`
>   - as name suggests. (\# of CPU)
>

Example env file:

```dotenv
SEAVL_COYO_PREDOWNLOAD_ONLY=1
SEAVL_COYO_SPLIT_CNT=20
SEAVL_COYO_SPLIT_IDX="0,5-8,12,18-19"
SEAVL_COYO_BATCH_SIZE=512
SEAVL_COYO_NUM_WORKERS=2
```
