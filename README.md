# Telegram NLP Bot (WSEI NLP)

Jeden bot Telegram, kazde laboratorium dostaje swoj folder (`lab01/`, `lab02/`, ...) z wlasnymi
plikami: tresc zadania (`LabXX.md`), kod specyficzny dla danego laba, wygenerowane dane i wykresy
(`plots/`). Kod wspoldzielony przez wszystkie laby (tokenizacja, czyszczenie tekstu, klasyfikator
z Lab01) jest w `common/`.

```text
common/            wspolne moduly NLP (nlp_task.py, classifier.py)
lab01/             Lab01.md, sentences.json, plots/
lab02/             Lab02.md, lab2_*.py, lab2results.csv, lab2_similar_words.txt, plots/
lab03/             lab03.md, sentiment_dataset.csv, config/datasets/preprocessing/
                   sentiment_methods/training/compare/commands/model_loader/visualizations.py,
                   models/, lab3plots/, lab3results.csv
bot.py             jeden punkt wejscia - importuje z common/ i lab0N/
```

Projekt realizuje wymagania z `lab01/Lab01.md`, `lab02/Lab02.md` i `lab03/lab03.md`.

## Funkcjonalnosci

Bot obsluguje komendy:

- `/task <task_name> "text" "class"`
- `/full_pipeline "text" "class"`
- `/classifier "text"`
- `/stats`

Obslugiwane klasy:

- `pozytywny`
- `neutralny`
- `negatywny`

Przy `/task` oraz `/full_pipeline` dane sa dopisywane do `sentences.json` w formacie:

```json
{ "text": "...", "class": "..." }
```

## Wymagania

- Python 3.10+
- Telegram Bot Token (z BotFather)

## Instalacja

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Konfiguracja

Ustaw token bota:

```bash
export TELEGRAM_BOT_TOKEN="<YOUR_TOKEN>"
```

Opcjonalne zmienne:

- `SENTENCES_PATH` (domyslnie: `lab01/sentences.json`)
- `PLOTS_DIR` (domyslnie: `lab01/plots`)
- `LAB2_PLOTS_DIR` (domyslnie: `lab02/plots`)
- `LAB2_RESULTS_PATH` (domyslnie: `lab02/lab2results.csv`)
- `POLISH_STOPWORDS_URL` (domyslnie: `https://raw.githubusercontent.com/stopwords-iso/stopwords-pl/master/raw/gh-stopwords-json-pl.txt`)

## Uruchomienie

```bash
python3 bot.py
```

## Przyklady uzycia

```text
/task tokenize "To byl bardzo interesujacy wyklad." "neutralny"
/task plot_wordcloud "To jest test test slowo." "neutralny"
/full_pipeline "System dziala szybko, ale interfejs wymaga poprawy." "neutralny"
/classifier "To byl fantastyczny film"
/stats
```

## Lab02 - eksperymenty klasyfikacji (`/classify`)

Komenda `/classify` (patrz `Lab02.md`) trenuje wybrane modele na calym zbiorze danych,
przy uzyciu wielu reprezentacji tekstu (`bow`, `tfidf`, `word2vec`, `glove`).

```text
/classify dataset=20news_group method=all gridsearch=false run=1
/classify dataset=20news_group method=logreg gridsearch=true run=2
/classify dataset=20news_group method=rf,nb gridsearch=false run=3
```

Parametry:

- `dataset` - wbudowane: `20news_group` (sklearn `fetch_20newsgroups`, 4 kategorie). Inne datasety (`ag_news` itp.)
  wymagaja lokalnego pliku CSV z kolumnami `text,label` w katalogu `datasets/<nazwa>.csv`.
- `method` - `nb`, `rf`, `mlp`, `logreg` lub `all` (lista po przecinku, np. `rf,nb`).
- `gridsearch` - `true`/`false`, strojenie hiperparametrow z `cv=3`.
- `run` - liczba powtorzen z innym seedem (`42`, `1337`, `2024`, ...), wyniki sa usredniane w odpowiedzi bota.

Wygenerowane artefakty (w `lab02/`):

- `lab02/plots/wordcloud_corpus.png`, `lab02/plots/wordcloud_class_<klasa>.png`
- `lab02/plots/{dataset}_{embedding}_{pca|tsne|svd}_embedding.png` - wizualizacja przestrzeni embeddingu
- `lab02/plots/confusion_<embedding>_<model>.png` - macierze pomylek
- `lab02/plots/feature_importance_<dataset>_<embedding>_<model>.txt` - top cechy (dla `bow`/`tfidf` + `rf`/`logreg`)
- `lab02/lab2_similar_words.txt` - podobne slowa dla `word2vec`/`glove` (space, computer, science, music, car), osobna sekcja na embedding
- `lab02/plots/word_embedding_<embedding>_pca.png`, `lab02/plots/word_embedding_<embedding>_tsne.png` (per `word2vec`/`glove`)
- `lab02/lab2results.csv` - wszystkie wyniki (`dataset,embedding,model,accuracy,macro_f1,seed`)

Domyslnie kazda klasa datasetu jest ograniczona do `LAB2_MAX_SAMPLES_PER_CLASS` (120) przykladow, zeby
eksperyment liczyl sie w rozsadnym czasie - mozna to zmienic zmienna srodowiskowa.

## Lab03 - analiza sentymentu i sieci sekwencyjne

Komendy (patrz `lab03/lab03.md`):

```text
/sentiment method=<rule|nb|rf|transformer|textblob|stanza|simplernn|lstm|gru> text="tekst"
/train model=<simplernn|lstm|gru> dataset=<amazon|imdb|custom>
/compare dataset=<amazon|imdb|custom> methods=<lista_metod>
/add_sentiment "tekst" "etykieta"
/models
```

Przyklady:

```text
/sentiment method=rule text="To był naprawdę świetny film"
/sentiment method=transformer text="Produkt przyszedł uszkodzony"
/train model=lstm dataset=custom
/compare dataset=custom methods=rule,nb,rf,textblob
/add_sentiment "Obsługa była poprawna, ale niczym mnie nie zachwyciła" "neutralny"
/models
```

Datasety:

- `custom` - `lab03/sentiment_dataset.csv` (kolumny `text,label`), rozszerzany przez `/add_sentiment`
  (kazdy tekst zapisywany jest jako jeden rekord, bez dzielenia na zdania).
- `imdb` - `tensorflow.keras.datasets.imdb` (dekodowany do tekstu, klasy: pozytywny/negatywny - 2 klasy).
- `amazon` - wymaga lokalnego pliku `datasets/amazon_sentiment.csv` (`text,label`) - Kaggle wymaga
  uwierzytelnienia, wiec nie da sie pobrac automatycznie.

Metody `/sentiment`:

- `rule` - prosty lexicon PL (slowa pozytywne/negatywne, bez wymagan treningowych).
- `nb` / `rf` - trenowane "na zywo" na `lab03/sentiment_dataset.csv` (TF-IDF + MultinomialNB / RandomForest).
- `transformer` - `transformers.pipeline("sentiment-analysis")`, model wielojezyczny
  `nlptown/bert-base-multilingual-uncased-sentiment` (1-5 gwiazdek zmapowane na 3 klasy);
  uzywa backendu TensorFlow (`framework="tf"`), zeby nie wymagac PyTorch.
- `textblob` - `TextBlob.sentiment.polarity`; biblioteka jest trenowana po angielsku, dla polskiego
  tekstu wyniki sa orientacyjne.
- `stanza` - pipeline `stanza` z modelem sentymentu dla `en` (Stanza nie ma wbudowanego modelu
  sentymentu dla polskiego); jak wyzej - ograniczona jakosc dla tekstu PL.
- `simplernn` / `lstm` / `gru` - wymagaja wczesniejszego `/train`. `/sentiment` szuka wytrenowanego
  modelu w kolejnosci datasetow `custom -> imdb -> amazon` (komenda `/sentiment` nie przyjmuje
  parametru dataset).

`/train`:

- buduje `Embedding -> (SimpleRNN|LSTM|GRU) -> Dense -> Dense(softmax)`,
  domyslnie `embedding_dim=100`, `max_len=200`, `batch_size=32`, `epochs=10`,
  `EarlyStopping(patience=10% epochs)` na `val_loss`.
- zapisuje `lab03/models/<model>_<dataset>.h5`, `..._tokenizer.h5`, `..._label_encoder.h5`
  (tokenizer/label encoder sa zapisane jako pickle pod rozszerzeniem `.h5`, zgodnie z nazewnictwem
  z `lab03.md` - nie sa to natywne pliki HDF5).
- generuje `lab03/lab3plots/train_history_<model>_<dataset>.png` (accuracy/loss).
- `max_len` mozna eksperymentalnie zmieniac przez `LAB3_MAX_LEN` (np. przetestuj 50/100/200/300).

`/compare`:

- uruchamia kazda metode na tym samym (ograniczonym `sample_limit`) zbiorze, liczy accuracy/precision/recall/macro_f1,
  dopisuje wiersze do `lab03/lab3results.csv`, zapisuje `lab03/lab3plots/confusion_<method>_<dataset>.png`
  i `lab03/lab3plots/compare_methods_<dataset>.png`, a takze word cloud per klasa
  (`lab03/lab3plots/wordcloud_<klasa>.png`) i rozklad klas (`lab03/lab3plots/class_distribution_<dataset>.png`).

## Struktura plikow

- `bot.py` - Telegram handlers i uruchomienie bota
- `common/nlp_task.py` - zadania NLP, pipeline, statystyki, wykresy (uzywane przez Lab01 i Lab02)
- `common/classifier.py` - zapis/odczyt JSON, trening modelu, predykcja (Lab01)
- `lab01/sentences.json` - dane treningowe Lab01
- `lab01/plots/` - wygenerowane wykresy PNG z Lab01
- `lab02/lab2_*.py` - dataset/embeddingi/modele/wizualizacje/orkiestracja eksperymentu z Lab02
- `lab02/plots/` - wygenerowane wykresy PNG z Lab02

## Uwagi

- Nazwy wykresow maja format `Sentence_{YYYY-MM-DD}_{HH-MM-SS}.png`.
- `/full_pipeline` dzieli tekst na zdania i zapisuje kazde zdanie z ta sama klasa.
- Klasyfikator wymaga co najmniej 2 klas w danych treningowych.
- Stop words dla jezyka polskiego sa pobierane z repo `stopwords-iso/stopwords-pl` (z fallbackiem do listy NLTK i listy awaryjnej).
