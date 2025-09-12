from __future__ import annotations
from typing import Iterable, Optional, Tuple, Dict, Any, Literal, Union
import numpy as np
import pandas as pd
import spacy
from collections import Counter
from nlpsych.utils import get_spacy_pipeline_base

# ===== Section: Core text stats =====
SplitOverall = Literal["combined", "per_column", "both"]

def spacy_descriptive_stats(
    *series_list: Iterable[pd.Series],
    use_lemmas_for_uniques: bool = True,
    split_overall: SplitOverall = "combined",
    nlp: Optional[spacy.Language] = None,
    n_process: int = 1,
    batch_size: int = 256,
    words_must_be_alpha: bool = True,
    drop_stopwords: bool = False,
    avg_sentence_mode: Literal["unweighted", "ratio"] = "unweighted"
) -> Tuple[pd.DataFrame, Union[Dict[str, Any], Dict[str, Dict[str, Any]]]]:
    """
    Compute per row text stats and aggregate overall.
    DataFrame index is MultiIndex of source_column and original index.
    avg_sentence_mode unweighted means mean of per sentence token counts.
    avg_sentence_mode ratio means total tokens divided by total sentences.
    """
    if nlp is None:
        nlp = get_spacy_pipeline_base()

    has_lemmatizer = "lemmatizer" in nlp.pipe_names
    if use_lemmas_for_uniques and not has_lemmatizer:
        use_lemmas_for_uniques = False

    rows = []
    for s in series_list:
        if not isinstance(s, pd.Series):
            raise ValueError("All inputs must be pandas Series")
        src = s.name if s.name is not None else "<unnamed>"
        for idx, text in s.dropna().items():
            if isinstance(text, str) and text.strip():
                rows.append({"text": text, "source_column": src, "index": idx})

    if not rows:
        return pd.DataFrame(), {}

    texts = [r["text"] for r in rows]

    def keep_token(t: spacy.tokens.Token) -> bool:
        if t.is_space or t.is_punct:
            return False
        if words_must_be_alpha and not t.is_alpha:
            return False
        if drop_stopwords and t.is_stop:
            return False
        return True

    stats_rows = []
    unique_sets = []
    pos_full_list = []

    # Process texts
    for doc, meta in zip(nlp.pipe(texts, batch_size=batch_size, n_process=n_process), rows):
        tokens = [t for t in doc if not t.is_space]
        word_tokens = [t for t in tokens if keep_token(t)]
        chars = len(meta["text"])
        word_count = len(word_tokens)
        sentences = list(doc.sents)
        sentence_count = len(sentences)

        # Average word length
        avg_word_length = float(np.mean([len(t.text) for t in word_tokens])) if word_count else 0.0

        # Average sentence length
        if sentence_count:
            if avg_sentence_mode == "unweighted":
                per_sent_counts = [len([t for t in sent if keep_token(t)]) for sent in sentences]
                avg_sentence_length = float(np.mean(per_sent_counts))
            else:
                total_in_sent = sum(len([t for t in sent if keep_token(t)]) for sent in sentences)
                avg_sentence_length = float(total_in_sent) / float(sentence_count) if sentence_count else 0.0
        else:
            avg_sentence_length = 0.0

        # Vocabulary set per row
        if use_lemmas_for_uniques:
            uniques = {t.lemma_.lower() for t in word_tokens if t.lemma_}
        else:
            uniques = {t.text.lower() for t in word_tokens}

        unique_words = len(uniques)
        lexical_diversity = float(unique_words / word_count) if word_count else 0.0
        pos_counts = Counter(t.pos_ for t in word_tokens)

        unique_sets.append((meta["source_column"], meta["index"], uniques))
        pos_full_list.append((meta["source_column"], meta["index"], pos_counts))

        stats_rows.append(
            {
                "source_column": meta["source_column"],
                "index": meta["index"],
                "char_count": int(chars),
                "word_count": int(word_count),
                "sentence_count": int(sentence_count),
                "avg_word_length": float(avg_word_length),
                "avg_sentence_length": float(avg_sentence_length),
                "unique_words": int(unique_words),
                "lexical_diversity": float(lexical_diversity),
                "noun_count": int(sum(1 for t in word_tokens if t.pos_ in {"NOUN", "PROPN"})),
                "verb_count": int(sum(1 for t in word_tokens if t.pos_ in {"VERB", "AUX"})),
                "adjective_count": int(sum(1 for t in word_tokens if t.pos_ == "ADJ")),
                "adverb_count": int(sum(1 for t in word_tokens if t.pos_ == "ADV")),
                "pos_full": dict(pos_counts),
            }
        )

    stats_df = pd.DataFrame(stats_rows)
    stats_df.set_index(["source_column", "index"], inplace=True)
    stats_df.sort_index(inplace=True)

    uniques_by_row = {(src, idx): u for src, idx, u in unique_sets}
    pos_by_row = {(src, idx): c for src, idx, c in pos_full_list}

    def compute_overall(index_like) -> Dict[str, Any]:
        sub = stats_df.loc[index_like]
        total_words = int(sub["word_count"].sum())
        vocab_union = set()
        for key in sub.index:
            vocab_union |= uniques_by_row.get(key, set())
        pos_total = Counter()
        for key in sub.index:
            pos_total += pos_by_row.get(key, Counter())

        # Average sentence length aggregation mirrors per row choice
        if len(sub):
            if avg_sentence_mode == "unweighted":
                avg_sentence_length_agg = float(sub["avg_sentence_length"].mean())
            else:
                total_sentences = int(sub["sentence_count"].sum())
                avg_sentence_length_agg = float(total_words) / float(total_sentences) if total_sentences else 0.0
        else:
            avg_sentence_length_agg = 0.0

        return {
            "total_chars": int(sub["char_count"].sum()),
            "total_words": total_words,
            "total_sentences": int(sub["sentence_count"].sum()),
            "avg_word_length": float(sub["avg_word_length"].mean()) if len(sub) else 0.0,
            "avg_sentence_length": float(avg_sentence_length_agg),
            "total_unique_words": int(len(vocab_union)),
            "lexical_diversity": float(len(vocab_union) / total_words) if total_words else 0.0,
            "pos_distribution_basic": {
                "nouns": int(sub["noun_count"].sum()),
                "verbs": int(sub["verb_count"].sum()),
                "adjectives": int(sub["adjective_count"].sum()),
                "adverbs": int(sub["adverb_count"].sum()),
            },
            "pos_distribution_full": dict(pos_total),
        }

    if stats_df.empty:
        return stats_df, {}

    if split_overall == "combined":
        return stats_df, compute_overall(stats_df.index)
    if split_overall == "per_column":
        out = {}
        for col, subdf in stats_df.groupby(level=0):
            out[col] = compute_overall(subdf.index)
        return stats_df, out
    if split_overall == "both":
        per_col = {}
        for col, subdf in stats_df.groupby(level=0):
            per_col[col] = compute_overall(subdf.index)
        return stats_df, {"combined": compute_overall(stats_df.index), "per_column": per_col}
    return stats_df, compute_overall(stats_df.index)

