# Dependencies

```
scrapy                    2.12.0          py312h7900ff3_1    conda-forge
selenium                  4.27.1             pyhd8ed1ab_0    conda-forge
```

# How to run
1. Go to `scrapy_unesco` directory
2. `scrapy crawl unesco_whc -O landmarks.jsonl`
3. Go to `selenium_unesco` directory
4. `python gallery_scraper landmarks.jsonl`