# Movie Recommender — MIS 3080 Final Project

End-to-end ML + AI tool that takes a movie title, genre, or free-form theme as input and returns ranked movie recommendations.

## Architecture (candidate generation → ranking)

1. **AI component (candidate generation)** — every movie's plot/title/tagline/genres/keywords are encoded with `sentence-transformers/all-MiniLM-L6-v2`. A user query is embedded the same way and the top 50 movies by cosine similarity are retrieved.
2. **ML component (ranking)** — an XGBoost regressor trained on TMDB tabular features **plus 36 collaborative-filtering signals derived from the Netflix Prize dataset** (4 per-movie aggregates + 32 SVD latent factors) predicts each movie's `vote_average`. The 50 candidates are re-ranked by a blend of similarity and predicted rating.
3. **Decision/output** — top 5–15 movies returned with TMDB rating, predicted rating, similarity, blended score, plot overview, and (optionally) a one-line LLM-generated explanation per movie via the Google Gemini API.

## Repo layout

```
movie-recommender/
  data/                                  # raw TMDB CSVs
  ../Netflix Prize Data/                 # raw Netflix Prize files (sibling folder)
  notebooks/
    01_eda.ipynb                         # EDA + feature engineering    -> movies_clean.parquet
    01_eda_colab.ipynb
    02_train_model.ipynb                 # XGBoost rating regressor      -> model.joblib + meta
    02_train_model_colab.ipynb
    03_build_embeddings.ipynb            # sentence-transformers         -> embeddings.npy
    03_build_embeddings_colab.ipynb
    04_netflix_features.ipynb            # Netflix aggregates + SVD      -> movies_with_netflix.parquet
    04_netflix_features_colab.ipynb
  artifacts/
    movies_with_netflix.parquet          # final enriched dataset
    movies_clean.parquet                 # intermediate (TMDB-only)
    model.joblib                         # trained on TMDB + Netflix
    model_meta.json
    embeddings.npy
  app.py                                 # Streamlit UI
  requirements.txt
  writeup.docx
  README.md
```

The notebooks run **once** to produce artifacts. The Streamlit app loads them at startup and never re-trains or re-embeds at request time.

## Notebook execution order

Run notebooks in this order, each consuming the previous one's output:

1. **01** — produces `movies_clean.parquet`
2. **04** — adds Netflix Prize features → `movies_with_netflix.parquet`
3. **02** — trains the XGBoost regressor on the enriched dataset → `model.joblib`
4. **03** — encodes movies into embeddings → `embeddings.npy` (independent of 02 + 04)

Notebooks 01 and 03 only need the TMDB CSVs in `data/`. Notebook 04 also needs the Netflix Prize files; for the local notebook those should sit in `../Netflix Prize Data/` (sibling folder), and for the Colab notebook they're loaded via Drive mount.

## Running locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# (run notebooks 01, 04, 02, 03 once to build artifacts)
streamlit run app.py
```

The app opens at `http://localhost:8501`. First query takes ~5 sec (model load); subsequent queries return in **30–150 ms**.

## Deploying to Streamlit Community Cloud

1. Push this directory to a GitHub repo (artifacts included — total <14 MB).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → point at the repo, branch, and `app.py`.
3. Streamlit auto-installs from `requirements.txt` and gives you a public URL.

Note: the free tier sleeps after ~30 min of inactivity. Wake-up takes 30-60 sec on first hit; pre-warm before demoing.

## Held-out model performance

XGBoost rating regressor on a 20% held-out test split (731 movies, see `02_train_model.ipynb`):

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Mean baseline | 0.666 | 0.850 | 0.000 |
| Linear regression (TMDB) | 0.476 | 0.611 | 0.483 |
| XGBoost (TMDB only) | 0.418 | 0.550 | 0.580 |
| **XGBoost (TMDB + Netflix)** ← saved | **0.379** | **0.500** | **0.653** |

The Netflix Prize features lift R² by 7.3 points (+12.6% relative) over the TMDB-only model.

## Status

- [x] EDA + data prep (`01_eda.ipynb` / `01_eda_colab.ipynb`)
- [x] Embedding index (`03_build_embeddings.ipynb` / `03_build_embeddings_colab.ipynb`)
- [x] Netflix Prize features (`04_netflix_features.ipynb` / `04_netflix_features_colab.ipynb`)
- [x] Rating regressor training (`02_train_model.ipynb` / `02_train_model_colab.ipynb`)
- [x] Streamlit app (`app.py`)
- [ ] Deployment to Streamlit Community Cloud
- [x] Writeup (`writeup.docx`)

## Data sources

- [TMDB 5000 Movies & Credits](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) — public Kaggle dataset, ~4,803 movies released 1916-2017.
- [Netflix Prize Data](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) — ~100M ratings on ~17,770 movies, collected through 2005.
