# A Hybrid Extractive-Abstractive Framework for Telugu Text Summarization Using IndicBERT, IndicBART and Reinforcement Learning

This project explores a hybrid approach to Telugu news summarization. It combines Telugu text preprocessing, sentence ranking, abstractive generation, and a reward-based optimization experiment.

## Abstract

Telugu text summarization poses significant challenges due to the language's rich morphology and complex syntactic structure. To address these challenges, this study proposes a Morphology-Aware Hybrid Reinforcement-Based Telugu Text Summarization framework that integrates extractive and abstractive summarization techniques with a novel reinforcement learning optimization model. Initially, raw Telugu text is collected and pre-processed through Unicode normalization, sentence segmentation, tokenization, and stop-word removal. Contextual sentence representations are generated using IndicBERT, enriched with morphology-aware features to preserve semantic nuances. An extractive summarization module combining TextRank and a BiLSTM-Attention sentence ranking model identifies salient sentences. These sentences are subsequently refined using transformer-based abstractive models, namely IndicBART, to generate fluent and concise Telugu summaries. To further enhance summary quality, a Telugu-Adaptive Multi-Stage Optimization Reinforcement Learning (TAMSORL) model is introduced, which optimizes salience selection, semantic compression, and morphological integrity through language-aware reward mechanisms. 


## Repository contents

- `Main.py`
- `Graph.py`
- `Dataset/` 

## Requirements and usage

Use Python with the following packages installed: `pandas`, `numpy`, `torch`, `transformers`, `scikit-learn`, `networkx`, `rouge-score`, `telugu-stopwords`, `matplotlib`, and `pyarrow`.

The first run downloads the IndicBART model from Hugging Face and requires an internet connection. A CUDA-enabled GPU is optional; the script uses CUDA when available and otherwise runs on CPU.

From the repository root, run:

```bash
python Main.py
```


## Keywords

Telugu text summarization, hybrid model, reinforcement learning, abstractive summarization, BiLSTM-Attention, IndicBART, morphology-aware summarization.
