# A Hybrid Extractive-Abstractive Framework for Telugu Text Summarization Using IndicBERT, IndicBART and Reinforcement Learning

This project explores a hybrid approach to Telugu news summarization. It combines Telugu text preprocessing, sentence ranking, abstractive generation, and a reward-based optimization experiment.

## Abstract

Telugu text summarization poses significant challenges due to the language's rich morphology and complex syntactic structure. To address these challenges, this study proposes a Morphology-Aware Hybrid Reinforcement-Based Telugu Text Summarization framework that integrates extractive and abstractive summarization techniques with a novel reinforcement learning optimization model. Initially, raw Telugu text is collected and pre-processed through Unicode normalization, sentence segmentation, tokenization, and stop-word removal. Contextual sentence representations are generated using IndicBERT, enriched with morphology-aware features to preserve semantic nuances. An extractive summarization module combining TextRank and a BiLSTM-Attention sentence ranking model identifies salient sentences. These sentences are subsequently refined using transformer-based abstractive models, namely IndicBART, to generate fluent and concise Telugu summaries. To further enhance summary quality, a Telugu-Adaptive Multi-Stage Optimization Reinforcement Learning (TAMSORL) model is introduced, which optimizes salience selection, semantic compression, and morphological integrity through language-aware reward mechanisms. Experimental evaluation demonstrates superior performance with ROUGE-1: 55.7%, ROUGE-2: 31.4%, ROUGE-L: 50.9%, BLEU: 34.8%, and METEOR: 36.7%, confirming improved coherence, semantic retention, and linguistic accuracy for low-resource, morphologically rich languages like Telugu.

## Reported results

The following scores are those reported in the abstract; they are not automatically reproduced by running the current scripts.

| Metric | Reported score |
| --- | ---: |
| ROUGE-1 | 55.7% |
| ROUGE-2 | 31.4% |
| ROUGE-L | 50.9% |
| BLEU | 34.8% |
| METEOR | 36.7% |

## Method overview

The proposed framework describes these stages:

1. Normalize Telugu text, split it into sentences, tokenize it, and remove Telugu stop words.
2. Encode sentences and extract morphology-related features.
3. Rank sentences using TextRank and BiLSTM-Attention.
4. Generate a Telugu summary with IndicBART.
5. Experiment with reward-based optimization using a ROUGE-L reward.

## Repository contents

- `Main.py` — preprocessing, sentence representation, sentence selection, and summary-generation pipeline.
- `Graph.py` — plotting scripts for the comparison scores and reward curve.
- `Dataset/` — Telugu news dataset files and their accompanying dataset description.

## Requirements and usage

Use Python with the following packages installed: `pandas`, `numpy`, `torch`, `transformers`, `scikit-learn`, `networkx`, `rouge-score`, `telugu-stopwords`, `matplotlib`, and `pyarrow`.

The first run downloads the IndicBART model from Hugging Face and requires an internet connection. A CUDA-enabled GPU is optional; the script uses CUDA when available and otherwise runs on CPU.

From the repository root, run:

```bash
python Main.py
```

The script reads the train, validation, and test Parquet files from `Dataset/`. It currently limits each split to 20 documents (`MAX_DOCS = 20`) and prints an example generated summary.

## Implementation notes

The manuscript describes IndicBERT sentence representations, while the current `Main.py` loads `ai4bharat/IndicBART` and uses its encoder for sentence representations. The current BiLSTM-Attention module is instantiated without loading trained weights. Also, the final TAMSORL-named generation call in the script passes `use_tamsorl=False`. Consequently, this repository is an experimental implementation and does not, by itself, verify or reproduce the abstract's reported evaluation scores.

## Dataset attribution

The dataset description identifies the source as Telugu news from Andhra Jyothy and Eenadu and requests citation of the [Telugu News Dataset on Hugging Face](https://huggingface.co/datasets/saidines12/telugu_news_dataset). Please consult the dataset page and its terms before reusing or redistributing the data.

```bibtex
@misc{telugu_news_dataset,
  author = {saidineshpola},
  title = {Telugu News Dataset},
  year = {2024},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/saidines12/telugu_news_dataset}},
}
```

## Authors

**Rupa Devi T**

- Research Scholar, Department of Computer Science and Engineering, Koneru Lakshmaiah Education Foundation, Vaddeswaram, Guntur, Andhra Pradesh, India
- Assistant Professor, Keshav Memorial Institute of Technology
- Email: [rupa.devi179@gmail.com](mailto:rupa.devi179@gmail.com)
- ORCID: [0000-0003-0532-4621](https://orcid.org/0000-0003-0532-4621)

**Kolla Bhanu Prakash**

- Department of Computer Science and Engineering, Koneru Lakshmaiah Education Foundation, Vaddeswaram, Guntur, Andhra Pradesh, India
- Email: [drkbp@kluniversity.in](mailto:drkbp@kluniversity.in)
- ORCID: [0000-0002-7955-2777](https://orcid.org/0000-0002-7955-2777)

## Keywords

Telugu text summarization, hybrid model, reinforcement learning, abstractive summarization, BiLSTM-Attention, IndicBART, morphology-aware summarization.
