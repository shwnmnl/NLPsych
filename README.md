<p align="center">
  <img src="https://raw.githubusercontent.com/shwnmnl/nlpsych/main/assets/NLPsych_logo.png" alt="NLPsych logo" width="150"/>
</p>

<h1 align="center">NLPsych</h1>

NLPsych (*Natural Language Psychometrics*) lets you upload a CSV file, pick your text columns, and get instant descriptive statistics, visualize semantic embeddings, and run predictive models all in one streamlined workflow.

> **Warning**
> `pip install` releases of NLPsych should be treated as unstable for now. If you want to use NLPsych today, prefer running it from source locally. The public demo can also be used for exploration, but it runs on third-party servers, so avoid uploading sensitive data there.

## ⚙️ Features
- 📊 Descriptive statistics: word counts, sentence lengths, lexical diversity
- 🔎 Embeddings: Sentence-Transformers + dimensionality reduction (PCA, UMAP, t-SNE)
- 🤖 Modeling: auto/forced classification-vs-regression, CV safeguards, and permutation tests
- 📑 Reports: HTML/Markdown outputs with per-target interpretations and example write-up sentences

💡 Tip: If you only need the functions, install via pip. If you want the interactive app, clone the repo and run it with Streamlit.

## 📦 Installation (library only)

To use NLPsych functions in your own Python code:

```bash
pip install nlpsych
```

Use lowercase in commands and imports (`nlpsych`, `nlpsych_app`).

## 🚀 Running the app

Installed package:

```bash
pip install "nlpsych[app]"
nlpsych-app
```

From this repo (source/dev or Streamlit Cloud-style entrypoint):

```bash
pip install -e ".[app]"
streamlit run streamlit_app.py
```

For Streamlit Community Cloud, set the main file path to `streamlit_app.py`.

## 🛠 Example Usage (library)
```python
import pandas as pd
from nlpsych.descriptive_stats import descriptive_stats
from nlpsych.embedding import embed_text_columns_simple_base, reduce_embeddings

df = pd.DataFrame({"text": [
    "Hello world. This is a tiny test.",
    "Patient denies chest pain. Vitals stable."
]})

stats_df, overall = descriptive_stats(df["text"])
print(overall["lexical_diversity"])

meta_df, emb, texts = embed_text_columns_simple_base([df["text"]])
Z = reduce_embeddings(emb, method="pca", n_components=2)
````

## 📂 Project structure

```
root/
├── .devcontainer/
├── .streamlit/   
│   └── config.toml
├── assets/    
│   └── NLPsych_logo.png          
├── streamlit_app.py
├── src/
│   ├── nlpsych     
│   │   ├── __init__.py
│   │   ├── descriptive_stats.py
│   │   ├── embedding.py
│   │   ├── modeling.py
│   │   ├── report.py
│   │   └── utils.py
│   ├── nlpsych_app/              
│   │   ├── assets/    
│   │   │   └── NLPsych_logo.png          
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── launch.py
│   │   └── pages/
│   │       └── Docs.py
└── tests/              
````

## 🖋 License

MIT License © 2025 Shawn Manuel

## 👇 Contributing

PRs are welcome! If you have feature ideas or bug fixes, feel free to open an issue or submit a pull request.
