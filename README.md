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
lab04/             Lab04.md, config/ner/nel/translation/summarization/knowledge_graph/
                   language/commands/utils.py, lab4plots/, summaries/
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
  domyslnie `embedding_dim=100`, `max_len=200` (gorny limit), `batch_size=32` (gorny limit),
  `epochs=30`, `EarlyStopping(patience=max(3, 10% epochs))` na `val_loss`.
- **automatyczne dopasowanie do rozmiaru datasetu**: dla malych zbiorow (np. `custom` z dziesiatkami
  przykladow) `batch_size=32` oznaczalby jedna aktualizacje wag na epoke, a `max_len=200` rozmywaloby
  krotkie zdania samym paddingiem - zarowno `max_len`, jak i `batch_size`, sa wiec automatycznie
  zmniejszane: `max_len` do 95. percentyla dlugosci zdan treningowych (+2), `batch_size` do
  `len(train)//4` (min. 2). Faktyczny `max_len` jest zapisywany razem z `label_encoder` i odczytywany
  przy predykcji, zeby dlugosc sekwencji sie zgadzala.
- zapisuje `lab03/models/<model>_<dataset>.h5`, `..._tokenizer.h5`, `..._label_encoder.h5`
  (tokenizer/label encoder sa zapisane jako pickle pod rozszerzeniem `.h5`, zgodnie z nazewnictwem
  z `lab03.md` - nie sa to natywne pliki HDF5).
- generuje `lab03/lab3plots/train_history_<model>_<dataset>.png` (accuracy/loss).
- `max_len`/`batch_size`/`epochs` (gorne limity) mozna eksperymentalnie zmieniac przez
  `LAB3_MAX_LEN`, `LAB3_BATCH_SIZE`, `LAB3_EPOCHS`.
- na bardzo malych zbiorach (kilkadziesiat przykladow) wyniki sieci sekwencyjnych maja duza wariancje
  miedzy architekturami/seedami - to oczekiwane, nie blad; im wiecej danych w `sentiment_dataset.csv`
  (przez `/add_sentiment`), tym stabilniejsze wyniki.

`/compare`:

- uruchamia kazda metode na tym samym (ograniczonym `sample_limit`) zbiorze, liczy accuracy/precision/recall/macro_f1,
  dopisuje wiersze do `lab03/lab3results.csv`, zapisuje `lab03/lab3plots/confusion_<method>_<dataset>.png`
  i `lab03/lab3plots/compare_methods_<dataset>.png`, a takze word cloud per klasa
  (`lab03/lab3plots/wordcloud_<klasa>.png`) i rozklad klas (`lab03/lab3plots/class_distribution_<dataset>.png`).

## Lab04 - NER/NEL/NED, tlumaczenie i podsumowania LLM

Komendy (patrz `lab04/Lab04.md`):

```text
/ner method=<spacy|stanza> text="tekst"
/nel text="tekst" language=<en|pl>
/ned entity="tekst" context="tekst"
/translate text="tekst" target_lang=<en|pl|de|fr|es>
/summarize text="tekst" summary_type=<extractive|abstractive|bullets> length=<short|medium|long>
/analyze_entities text="tekst" link=<true|false>
/knowledge_graph text="tekst"
/language_detect text="tekst"
```

Przyklady:

```text
/ner method=spacy text="Steve Jobs, założyciel Apple, urodził się w San Francisco."
/nel text="Steve Jobs" language=en
/ned entity="Steve Jobs" context="Steve Jobs founded Apple, technology company"
/translate text="The quick brown fox jumps over the lazy dog" target_lang=pl
/summarize text="Dlugi tekst do podsumowania..." summary_type=abstractive length=medium
/analyze_entities text="Elon Musk posiada firme Tesla w Austin." link=true
/knowledge_graph text="Elon Musk posiada firme Tesla w Austin."
/language_detect text="To jest dluzszy tekst po polsku do wykrycia jezyka."
```

**NER** - `spacy` (`en_core_web_sm`/`pl_core_news_sm`, jezyk wykrywany automatycznie przez `langdetect`
jesli nie podano) i `stanza` (model NER dla wykrytego/wskazanego jezyka, pobierany automatycznie).
Male modele `*_sm` maja ograniczona dokladnosc dla nietypowych nazw firm/produktow (np. "Tesla", "xAI"
w polskim tekscie moga nie zostac wykryte) - to ograniczenie modelu, nie bledu integracji.

**NEL/NED** - wyszukiwanie kandydatow przez Wikidata API (`wbsearchentities`) + podglad z Wikipedia REST
API (wymaga naglowka `User-Agent`, inaczej Wikimedia zwraca 403). `/nel` zwraca liste kandydatow
z confidence (ranking wedlug pozycji w wynikach wyszukiwania). `/ned` dodatkowo re-rankuje kandydatow
wedlug nakladania sie opisu kandydata z podanym kontekstem i odfiltrowuje te o niskiej pewnosci
(`LAB4_NEL_CONFIDENCE_THRESHOLD`, domyslnie 0.3).

**Tlumaczenie** - modele `Helsinki-NLP/opus-mt-<src>-<tgt>` ladowane bezposrednio przez
`AutoTokenizer`/`AutoModelForSeq2SeqLM` (transformers 5.x nie ma juz generycznego pipeline'u
`translation`, wiec generowanie tlumaczenia robione jest recznie przez `model.generate()`).
Dla `en->pl` nie istnieje bezposredni model Helsinki-NLP - uzywany jest wielojezyczny
`opus-mt-en-sla` z prefiksem `>>pol<<` (`LAB4_TRANSLATION_MODEL_OVERRIDES` w `config.py`).
Wymaga pakietow `sentencepiece` i `sacremoses` (tokenizery Marian).

**Podsumowania (`/summarize`)** - komunikacja z lokalnym Ollama (`http://localhost:11434/api/generate`,
`LAB4_OLLAMA_MODEL` domyslnie `llama3.2`). Wymaga zainstalowanego i odpalonego Ollama
(`brew install ollama`, `ollama serve`, `ollama pull <model>`) - bez tego komenda zwraca czytelny blad
z instrukcja. Streszczenia zapisywane do `lab04/summaries/summary_<typ>_<dlugosc>_<timestamp>.txt`.

**`/analyze_entities`** - laczy NER (spacy) z NEL (Wikidata) dla kazdej wykrytej encji.

**`/knowledge_graph`** (opcjonalne) - heurystyczne relacje miedzy encjami wspolwystepujacymi w tym samym
zdaniu (na podstawie slow kluczowych w oknie tekstu wokol par encji - `founder`/`worked_for`/
`located_in`/`belongs_to`, domyslnie `related_to`), wizualizacja `networkx`+`matplotlib`,
zapis do `lab04/lab4plots/knowledge_graph_<timestamp>.png`.

**`/language_detect`** - `langdetect`; na bardzo krotkich tekstach (kilka slow) detekcja jezyka
moze byc niepewna (znana slabość statystycznych metod tego typu) - dluzsze teksty dzialaja stabilnie.

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
