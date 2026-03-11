from __future__ import annotations
import sys, subprocess
import spacy
import warnings
from functools import lru_cache

@lru_cache(maxsize=1)
def get_spacy_pipeline_base(allow_download: bool = True) -> spacy.Language:
    """
    Load and cache the base spaCy English pipeline used across NLPsych.

    Parameters
    ----------
    allow_download : bool, default=True
        Whether to attempt downloading ``en_core_web_sm`` when it is not
        available locally.

    Returns
    -------
    spacy.Language
        Loaded spaCy pipeline with ``ner`` and ``parser`` excluded and a
        ``sentencizer`` component ensured.

    Raises
    ------
    RuntimeError
        If the model is unavailable and cannot be downloaded.
    """
    def _load():
        """
        Load ``en_core_web_sm`` with the lightweight component set.

        Returns
        -------
        spacy.Language
            Pipeline configured for fast tokenization/sentence segmentation.
        """
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
    """
    Load a SentenceTransformer model for text embedding.

    Parameters
    ----------
    model_name : str, default="all-MiniLM-L6-v2"
        Hugging Face model name or local model path.
    allow_download : bool, default=True
        Whether model download is allowed when the model is not cached locally.

    Returns
    -------
    SentenceTransformer
        Initialized sentence-transformer model ready for ``encode`` calls.

    Raises
    ------
    RuntimeError
        If the sentence-transformers package is missing, or if the model is
        unavailable locally while downloads are disabled.
    """
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
