# Medical Transcript Classification (NLP)

An NLP pipeline that classifies raw medical transcriptions into their **medical specialty** using TF-IDF features and a linear SVM, with spaCy-based linguistic analysis (noun extraction, dependency parsing, rule-based negation detection) and exploratory visualisations.

## Pipeline

```
Raw transcript → negation tagging → cleaning / abbreviation expansion / lemmatisation
              → TF-IDF (1–2 grams) → LinearSVC → evaluation
```

1. **Data prep** – keep `medical_specialty` and `transcription`, drop nulls (4,999 → 4,966 rows), then restrict to the 4 most frequent specialties: Surgery, Consult – History and Phy., Cardiovascular / Pulmonary, Orthopedic.
2. **Preprocessing** – replace negation cues (`no`, `denies`, `without`) with a `NEG` token, expand clinical abbreviations (`pt`, `hx`, `dx`, `tx`, `sx`), strip non-alphabetic characters, remove stopwords (negation words kept), lemmatise with NLTK WordNet.
3. **Linguistic analysis (spaCy `en_core_web_sm`)** – most frequent nouns, word clouds per specialty, dependency parse visualisation, and a dependency-based negation detector (`Present` / `Negated`).
4. **Features** – `TfidfVectorizer` (max 7,000 features, unigrams + bigrams, `min_df=2`); t-SNE plot of the feature space.
5. **Model** – `LinearSVC(C=0.5)` on a stratified 80/20 split.
6. **Evaluation** – accuracy and confusion matrix.

## Results

| Model | Features | Test accuracy |
|---|---|---|
| LinearSVC (C=0.5) | TF-IDF, 1–2 grams | **~60.3%** |

The classes overlap heavily in clinical vocabulary (e.g. orthopedic surgery vs. general surgery), which limits a bag-of-words model. Natural next steps: class weighting, per-class precision/recall, and transformer embeddings (ClinicalBERT / BioBERT).

## Dataset

A CSV of medical transcriptions with (at least) the columns `medical_specialty` and `transcription` (~5k rows; the MTSamples-style medical transcription dataset). The data file is **not** included in this repo. Place it in `data/` and update the loading cell in the notebook.

## Setup

```bash
git clone https://github.com/TheVariant0105/Medical-Transcipt-Classification.git
cd Medical-Transcipt-Classification
pip install -r requirements.txt
python -m spacy download en_core_web_sm
jupyter notebook notebooks/Medical_Transcript_Classification.ipynb
```

The notebook was written in Google Colab, so the upload cell uses `google.colab.files.upload()`. Outside Colab, replace it with `pd.read_csv("data/<your_file>.csv")`.

## Tech Stack

Python · pandas · scikit-learn · spaCy · NLTK · Matplotlib · Seaborn · WordCloud

## Author

Somesh Thumatti ([@TheVariant0105](https://github.com/TheVariant0105)) – part of my [profile projects](https://github.com/TheVariant0105).
