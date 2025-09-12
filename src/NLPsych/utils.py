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
            except Exception:
                pass
            warnings.warn(
                "Falling back to blank English pipeline (no pretrained vectors, NER, or parser).",
                UserWarning,
            )
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp
        
def get_st_model_base(model_name: str = "all-MiniLM-L6-v2", allow_download: bool = True) -> SentenceTransformer:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("sentence transformers is not installed") from e
    
    if allow_download:
        return SentenceTransformer(model_name)
    else:
        try:
            return SentenceTransformer(model_name)
        except Exception:
            raise RunTimeError(
                f"Model {model_name} not found locally and downloads are disabled."
            )