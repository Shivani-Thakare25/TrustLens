import re
import numpy as np
import pandas as pd


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s!?.,']", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def text_features(text):
    text = clean_text(text)

    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]

    word_count = len(words)
    char_count = len(text)

    return {
        "clean_text": text,
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": len(sentences),
        "avg_word_length": (
            np.mean([len(w) for w in words])
            if words else 0
        ),
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "uppercase_ratio": (
            sum(c.isupper() for c in str(text))
            / max(len(str(text)), 1)
        )
    }


def add_text_features(df, text_column):
    features = df[text_column].apply(text_features).apply(pd.Series)

    result = pd.concat(
        [df.reset_index(drop=True),
         features.reset_index(drop=True)],
        axis=1
    )

    return result


def prepare_fake_review_features(df):
    return add_text_features(df, "text")


def prepare_sentiment_features(df):
    # Combine Review + Summary for NLP processing
    df = df.copy()

    df["combined_text"] = (
        df["Summary"].fillna("").astype(str)
        + " "
        + df["Review"].fillna("").astype(str)
    ).str.strip()

    return add_text_features(df, "combined_text")