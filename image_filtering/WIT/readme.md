- Run Example
    ```shell
    SEAVL_WIT_IDX_ST=0 \
    SEAVL_WIT_IDX_ED=10000 \
    SEAVL_WIT_BS=8 \
    SEAVL_WIT_SPLITS="train" \
    python ./prepare_data.py
    ```

- Run Example (no slice)
    ```shell
    SEAVL_WIT_SPLITS="test validation" \
    SEAVL_WIT_DONT_USE_SEACROWD_PKG=1 \
    python ./prepare_data.py
    ```