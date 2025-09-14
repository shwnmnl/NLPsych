from __future__ import annotations
import sys, subprocess
import spacy
import warnings
from functools import lru_cache

@lru_cache(maxsize=1)
def get_spacy_pipeline_base(allow_download: bool = True) -> spacy.Language:
    """
    Load a light English spaCy pipeline that excludes NER and parser,
    and ensures a fast rule based sentencizer is present.
    On first run, if the model is missing, download it.
    If download fails, fall back to a blank English pipeline.
    """
    def _load():
        nlp = spacy.load("en_core_web_sm", exclude=["ner", "parser"])
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp

    try:
        return _load()
    except OSError:
        if allow_download:
            try:
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                return _load()
            except Exception as e:
                raise RuntimeError(
                    "Failed to download spaCy model en_core_web_sm. "
                ) from e
        raise RuntimeError(
            "spaCy model en_core_web_sm not installed"
        )

        
def get_st_model_base(model_name: str = "all-MiniLM-L6-v2", allow_download: bool = True) -> SentenceTransformer:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("sentence transformers is not installed") from e
    
    if allow_download:
        return SentenceTransformer(model_name)
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        raise RuntimeError(
            f"SentenceTransformer model {model_name} not found locally and downloads are disabled."
        ) from e