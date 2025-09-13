def test_import_and_version():
    import nlpsych
    assert hasattr(nlpsych, "__version__"), "__version__ missing on package"
