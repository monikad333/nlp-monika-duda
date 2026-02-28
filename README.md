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
