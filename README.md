# Telegram NLP Bot (Lab01)

Projekt realizuje wymagania z `Lab01.md`.

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

- `SENTENCES_PATH` (domyslnie: `sentences.json`)
- `PLOTS_DIR` (domyslnie: `plots`)
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

Wygenerowane artefakty:

- `lab2plots/wordcloud_corpus.png`, `lab2plots/wordcloud_class_<klasa>.png`
- `lab2plots/{dataset}_{embedding}_{pca|tsne|svd}_embedding.png` - wizualizacja przestrzeni embeddingu
- `lab2plots/confusion_<embedding>_<model>.png` - macierze pomylek
- `lab2plots/feature_importance_<dataset>_<embedding>_<model>.txt` - top cechy (dla `bow`/`tfidf` + `rf`/`logreg`)
- `lab2_similar_words.txt` - podobne slowa dla `word2vec`/`glove` (space, computer, science, music, car), osobna sekcja na embedding
- `lab2plots/word_embedding_<embedding>_pca.png`, `lab2plots/word_embedding_<embedding>_tsne.png` (per `word2vec`/`glove`)
- `lab2results.csv` - wszystkie wyniki (`dataset,embedding,model,accuracy,macro_f1,seed`)

Domyslnie kazda klasa datasetu jest ograniczona do `LAB2_MAX_SAMPLES_PER_CLASS` (120) przykladow, zeby
eksperyment liczyl sie w rozsadnym czasie - mozna to zmienic zmienna srodowiskowa.

## Struktura plikow

- `bot.py` - Telegram handlers i uruchomienie bota
- `nlp_task.py` - zadania NLP, pipeline, statystyki, wykresy
- `classifier.py` - zapis/odczyt JSON, trening modelu, predykcja
- `sentences.json` - dane treningowe
- `plots/` - wygenerowane wykresy PNG

## Uwagi

- Nazwy wykresow maja format `Sentence_{YYYY-MM-DD}_{HH-MM-SS}.png`.
- `/full_pipeline` dzieli tekst na zdania i zapisuje kazde zdanie z ta sama klasa.
- Klasyfikator wymaga co najmniej 2 klas w danych treningowych.
- Stop words dla jezyka polskiego sa pobierane z repo `stopwords-iso/stopwords-pl` (z fallbackiem do listy NLTK i listy awaryjnej).
