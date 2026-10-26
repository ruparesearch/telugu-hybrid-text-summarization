import pandas as pd
import re
import unicodedata
import numpy as np
import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import torch.nn as nn
from rouge_score import rouge_scorer
from telugu_stopwords import tsw

telugu_sw = tsw()
TELUGU_STOPWORDS = set(telugu_sw.get_stopwords(script="telugu"))

# ============================================================
# (2) Unicode Normalization
# ============================================================
def unicode_normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text)

# ============================================================
# (3) Sentence Segmentation
# ============================================================
def sentence_segmentation(text: str) -> List[str]:
    sentences = re.split(r"[.!?।]+", text)
    return [s.strip() for s in sentences if s.strip()]

# ============================================================
# (4) Tokenization + Stop-word Removal
# ============================================================
def tokenize_and_remove_stopwords(sentence: str) -> List[str]:
    tokens = re.findall(r"[\u0C00-\u0C7F]+", sentence)
    return [t for t in tokens if t not in TELUGU_STOPWORDS]

# ============================================================
# Load Parquet & Apply Preprocessing (Eq. 2 → 4)
# ============================================================
def preprocess_parquet(parquet_path: str, split_name: str) -> pd.DataFrame:
    print(f"\n================ {split_name.upper()} DATA =================")
    df = pd.read_parquet(parquet_path)

    # Detect Telugu text column
    for col in ["article", "text", "headline", "document"]:
        if col in df.columns:
            text_column = col
            break
    else:
        raise ValueError("No Telugu text column found")

    print(f"📄 Using text column: {text_column}")

    df["normalized_text"] = df[text_column].apply(unicode_normalize)
    df["segmented_sentences"] = df["normalized_text"].apply(sentence_segmentation)
    df["tokenized_sentences"] = df["segmented_sentences"].apply(
        lambda sents: [tokenize_and_remove_stopwords(s) for s in sents]
    )

    print("\n🔹 (2) Unicode Normalized Text:")
    print(df["normalized_text"].iloc[0][:300], "...")
    print("\n🔹 (3) Sentence Segmentation:")
    print(df["segmented_sentences"].iloc[0])
    print("\n🔹 (4) Tokenized Sentences:")
    print(df["tokenized_sentences"].iloc[0])

    return df

# ============================================================
# (5) Sentence Embeddings – IndicBART Encoder (Batched)
# ============================================================
MODEL_NAME = "ai4bharat/IndicBART"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, keep_accents=True)
bart_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
bart_model.eval()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bart_model.to(DEVICE)

def sentence_embedding_batch(sentences: List[List[str]]) -> List[np.ndarray]:

    flat_sents = [" ".join(s) for s in sentences]
    input_texts = [f"{s} </s> <2te>" for s in flat_sents]

    inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
        add_special_tokens=False
    ).to(DEVICE)

    with torch.no_grad():
        encoder_outputs = bart_model.model.encoder(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask
        )
    embeddings = encoder_outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return [embeddings[i] for i in range(len(sentences))]

def extract_sentence_embeddings(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    embeddings = []
    for doc in df["tokenized_sentences"]:
        embeddings.append(sentence_embedding_batch(doc))
    df["sentence_embeddings"] = embeddings

    print("\n🔹 (5) Sentence Embedding Shape (first sentence):")
    print(df["sentence_embeddings"].iloc[0][0].shape)
    return df

# ============================================================
# (6) Morphology-aware Feature Extraction
# ============================================================
COMMON_TELUGU_SUFFIXES = ["లు", "కు", "తో", "లో", "పై", "నుండి", "గా", "వల్ల", "ని", "లోని", "తోటి"]

def morphology_features(tokens: List[str]) -> dict:
    if not tokens:
        return {"avg_word_length":0.0,"max_word_length":0,"suffix_count":0,"token_count":0}
    lengths = [len(w) for w in tokens]
    suffix_count = sum(any(w.endswith(suf) for suf in COMMON_TELUGU_SUFFIXES) for w in tokens)
    return {"avg_word_length":float(np.mean(lengths)),
            "max_word_length":int(np.max(lengths)),
            "suffix_count":suffix_count,
            "token_count":len(tokens)}

def extract_morph_features(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    morph_feats = []
    for doc in df["tokenized_sentences"]:
        morph_feats.append([morphology_features(sent) for sent in doc])
    df["morph_features"] = morph_feats

    print("\n🔹 (6) Morphological Features (first sentence):")
    print(df["morph_features"].iloc[0][0])
    return df

# ============================================================
# TextRank
# ============================================================
def textrank_scores(sentence_embeddings: List[np.ndarray]) -> List[float]:
    if len(sentence_embeddings) < 2: return [1.0]
    sim_matrix = cosine_similarity(np.array(sentence_embeddings))
    np.fill_diagonal(sim_matrix, 0)
    graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(graph)
    return [scores[i] for i in range(len(scores))]

# ============================================================
# BiLSTM-Attention 
# ============================================================
class BiLSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim*2, 1)

    def forward(self, x):
        lstm_out, _ = self.bilstm(x.unsqueeze(0))
        attn_scores = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        return attn_scores.squeeze(0)

# ============================================================
# Hybrid Sentence Selection
# ============================================================
def hybrid_sentence_selection(sentences, embeddings, lambda_weight=0.6, top_k=3):
    tr_scores = textrank_scores(embeddings)
    embed_tensor = torch.tensor(np.array(embeddings), dtype=torch.float32)
    bilstm_model = BiLSTMAttention(embed_tensor.size(1))
    bilstm_model.eval()
    with torch.no_grad():
        attn_scores = bilstm_model(embed_tensor).numpy()
    hybrid_scores = [lambda_weight*tr + (1-lambda_weight)*attn for tr, attn in zip(tr_scores, attn_scores)]
    ranked = sorted(enumerate(hybrid_scores), key=lambda x:x[1], reverse=True)
    selected_indices = sorted([idx for idx,_ in ranked[:top_k]])
    return [sentences[i] for i in selected_indices]

# ============================================================
# Abstractive Rewrite using IndicBART
# ============================================================
def abstractive_rewrite(extracted_sentences: List[str]) -> str:
    text = " ".join(extracted_sentences)
    input_text = f"{text} </s> <2te>"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256, add_special_tokens=False).to(DEVICE)
    with torch.no_grad():
        summary_ids = bart_model.generate(
            inputs.input_ids,
            num_beams=4,
            max_length=120,
            early_stopping=True,
            repetition_penalty=2.0,
            decoder_start_token_id=tokenizer._convert_token_to_id_with_added_voc("<2te>")
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ============================================================
# TAMSORL: Reward + RL Optimization
# ============================================================
def compute_reward(summary: str, reference: str) -> float:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(reference, summary)['rougeL'].fmeasure

def tamsorl_optimize(extracted_sentences: List[str], reference: str,
                     alpha=1e-5, num_samples=1) -> str:
    bart_model.train()
    text = " ".join(extracted_sentences)
    input_text = f"{text} </s> <2te>"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=80, add_special_tokens=False).to(DEVICE)
    optimizer = torch.optim.Adam(bart_model.parameters(), lr=alpha)
    best_summary, best_reward = "", -1.0

    for _ in range(num_samples):

        summary_ids = bart_model.generate(
            inputs.input_ids,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_beams=1,
            max_length=120,
            repetition_penalty=2.0,
            decoder_start_token_id=tokenizer._convert_token_to_id_with_added_voc("<2te>")
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        reward = compute_reward(summary, reference)

        target_ids = tokenizer(summary, return_tensors="pt", truncation=True, max_length=256).input_ids.to(DEVICE)
        outputs = bart_model(input_ids=inputs.input_ids, labels=target_ids)
        loss = -outputs.loss * compute_reward(summary, reference)  # your RL weighting
         
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if reward > best_reward:
            best_reward = reward
            best_summary = summary

    bart_model.eval()
    return best_summary

def generate_summary(row, top_k=3, use_tamsorl=False):
    sentences = [" ".join(s) for s in row["tokenized_sentences"]]
    embeddings = row["sentence_embeddings"]

    extracted = hybrid_sentence_selection(
        sentences, embeddings, lambda_weight=0.6, top_k=top_k
    )

    if use_tamsorl and "article" in row:
        return tamsorl_optimize(extracted, row["article"])
    else:
        return abstractive_rewrite(extracted)

# ============================================================
# RUN PIPELINE
# ============================================================
if __name__ == "__main__":
    MAX_DOCS = 20

    # ---------------- TRAIN ----------------
    train_df = preprocess_parquet("Dataset/train-00000-of-00001.parquet", "train").head(MAX_DOCS)
    train_df = extract_sentence_embeddings(train_df, "train")
    train_df = extract_morph_features(train_df, "train")

    # ---------------- VAL ----------------
    val_df = preprocess_parquet("Dataset/val-00000-of-00001.parquet", "val").head(MAX_DOCS)
    val_df = extract_sentence_embeddings(val_df, "val")
    val_df = extract_morph_features(val_df, "val")

    # ---------------- TEST ----------------
    test_df = preprocess_parquet("Dataset/test-00000-of-00001.parquet", "test").head(MAX_DOCS)
    test_df = extract_sentence_embeddings(test_df, "test")
    test_df = extract_morph_features(test_df, "test")

    # ---------------- GENERATE SUMMARIES ----------------
    train_df["generated_summary"] = train_df.apply(lambda row: generate_summary(row, top_k=3), axis=1)
    print("\n📝 Generated Summary (first doc):")
    print(train_df["generated_summary"].iloc[0])

    # ---------------- GENERATE SUMMARIES USING TAMSORL ----------------
    train_df["generated_summary_tamsorl"] = train_df.apply(
        lambda row: generate_summary(row, top_k=3, use_tamsorl=False),
        axis=1
    )
import Graph