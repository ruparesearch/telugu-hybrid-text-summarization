---
dataset_info:
  features:
  - name: story_id
    dtype: int64
  - name: headline
    dtype: string
  - name: article
    dtype: string
  splits:
  - name: train
    num_bytes: 284670404
    num_examples: 83866
  - name: validation
    num_bytes: 13660238
    num_examples: 5962
  - name: test
    num_bytes: 44563431
    num_examples: 12504
  download_size: 52798742
  dataset_size: 342894073
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
tags:
- Telugu
- News
pretty_name: Telugu News
size_categories:
- 100K<n<1M
---

# Telugu News Dataset

This dataset contains Telugu news articles with their corresponding headlines. It is intended for use in natural language processing tasks such as text classification, summarization, and machine translation.


## Data Source

The data was collected from [Androjyothi paper](https://www.andhrajyothy.com/) and Eenadu paper.

## Data Format

The dataset is provided in `data/`. Each row represents story_id, news article and headline.

* **aticle:**  The news article in Telugu
* **headline** The new article's headline in Telugu

## Datset Partition

This dataset is separated according to the news paper's publication date. From September 25 to December 25, 2024, is the Train set, and from December 26 to January 9, 2025, is the Test set.
 


If you use this dataset please site it using following citation.

# Citation
```bibtex
@misc{telugu_news_dataset,
  author = {saidineshpola},
  title = {Telugu News Dataset},
  year = {2024},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/saidines12/telugu_news_dataset}},
}
```