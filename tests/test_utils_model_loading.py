import builtins
import sys
import types

import pytest
import spacy

import nlpsych.utils as utils


@pytest.fixture(autouse=True)
def _clear_spacy_cache():
    utils.get_spacy_pipeline_base.cache_clear()
    yield
    utils.get_spacy_pipeline_base.cache_clear()


def test_get_spacy_pipeline_base_downloads_missing_model_and_retries(monkeypatch):
    calls = {"load": 0, "run": None}

    def fake_load(name, exclude):
        calls["load"] += 1
        assert name == "en_core_web_sm"
        assert exclude == ["ner", "parser"]
        if calls["load"] == 1:
            raise OSError("missing model")
        return spacy.blank("en")

    def fake_run(cmd, check, stdout, stderr):
        calls["run"] = {
            "cmd": cmd,
            "check": check,
            "stdout": stdout,
            "stderr": stderr,
        }

    monkeypatch.setattr(utils.spacy, "load", fake_load)
    monkeypatch.setattr(utils.subprocess, "run", fake_run)

    nlp = utils.get_spacy_pipeline_base(allow_download=True)

    assert calls["load"] == 2
    assert calls["run"]["cmd"] == [
        sys.executable,
        "-m",
        "spacy",
        "download",
        "en_core_web_sm",
    ]
    assert calls["run"]["check"] is True
    assert "sentencizer" in nlp.pipe_names


def test_get_spacy_pipeline_base_warns_and_falls_back_when_download_fails(monkeypatch):
    monkeypatch.setattr(utils.spacy, "load", lambda name, exclude: (_ for _ in ()).throw(OSError("still missing")))
    monkeypatch.setattr(
        utils.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("download failed")),
    )

    with pytest.warns(RuntimeWarning, match="Failed to download spaCy model"):
        nlp = utils.get_spacy_pipeline_base(allow_download=True)

    assert isinstance(nlp, spacy.Language)
    assert nlp.lang == "en"
    assert "sentencizer" in nlp.pipe_names


def test_get_st_model_base_raises_clear_error_when_sentence_transformers_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sentence_transformers":
            raise ImportError("not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="sentence transformers is not installed"):
        utils.get_st_model_base("fake-model")


def test_get_st_model_base_respects_download_flag_for_local_cache_failures(monkeypatch):
    class FakeSentenceTransformer:
        calls = []

        def __init__(self, model_name):
            FakeSentenceTransformer.calls.append(model_name)
            raise OSError("missing local model")

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    with pytest.raises(RuntimeError, match="downloads are disabled"):
        utils.get_st_model_base("missing-model", allow_download=False)

    assert FakeSentenceTransformer.calls == ["missing-model"]


def test_get_st_model_base_returns_loaded_model_when_available(monkeypatch):
    class FakeSentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    model = utils.get_st_model_base("available-model", allow_download=True)

    assert isinstance(model, FakeSentenceTransformer)
    assert model.model_name == "available-model"
