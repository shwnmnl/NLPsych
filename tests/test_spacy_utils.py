import pytest

def test_spacy_helper_no_download():
    # Import here so the test doesn't require spaCy until this file runs
    from nlpsych.utils import get_spacy_pipeline_base
    nlp = get_spacy_pipeline_base(allow_download=False)
    # Should always return a Language object with a sentencizer
    from spacy.language import Language
    assert isinstance(nlp, Language)
    assert "sentencizer" in nlp.pipe_names, "sentencizer should always be present"
