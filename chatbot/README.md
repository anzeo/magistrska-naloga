# Implementacija pogovornega robota

Ta direktorij vsebuje glavno logiko pogovornega robota, specializiranega za vprašanja glede **Evropskega akta o umetni inteligenci (AI Act)**.  
Robota je mogoče uporabljati tako prek terminala kot tudi prek API vmesnika za povezavo s frontendom ali drugimi sistemi.

---

## Zahteve za zagon

- **Python**: priporočena različica `3.11.9`
- **OpenAI API ključ** (za dostop do modelov prek OpenAI)

### Namestitev odvisnosti

Pred zagonom namesti vse potrebne knjižnice:

```bash
pip install -r requirements.txt
```

Nastavi **OpenAI API ključ** v svojem okolju:

```bash
export OPENAI_API_KEY={your_key}
```

---

## Načini zagona

### 1. Poganjanje robota v terminalu

Za interaktivno testiranje robota v terminalu zaženi:

```bash
make serve-terminal
```
Pri tem načinu uporabe se zgodovina pogovora ohranja le v času trajanja trenutne seje in se ob ponovnem zagonu skripte izgubi.

### 2. Zagon API vmesnika (FastAPI)

Če želimo dostopati do robota prek API-ja (npr. za uporabo v spletni aplikaciji), zaženi:

```bash
make serve-api
```

V tem načinu se zgodovina pogovora persistira v SQLite bazo in se ohrani tudi med ponovnimi zagoni aplikacije.

---

## Shranjevanje besedil v vektorskem prostoru (Embeddings)

Če želimo na novo ustvariti vektorsko predstavitev dokumentov s pomočjo **TF-IDF**, zaženi:

```bash
make store
```

Vektorji bodo shranjeni v direktorij: `src/retriever/tfidf_embeddings/`

---

## Struktura projekta

Glavne komponente projekta so organizirane v naslednje module:

- `data/` - vsebina akta, razdeljena na manjše enote
- `src/core/` – logika pogovornega robota
- `src/retriever/` – retrieval sistem (iskanje relevantnih vsebin)
- `src/api/` – FastAPI vmesnik

---

## Makefile ukazi

| Ukaz                  | Namen                                      |
|------------------------|--------------------------------------------|
| `make serve-terminal`  | Zagon robota v terminalu                   |
| `make serve-api`       | Zagon FastAPI API vmesnika                 |
| `make store`           | Ponovno izračunavanje TF-IDF vektorjev     |

---

## Opombe

- Če se lokacije datotek spremenijo, preveri poti v `src/config.py`.
- Projekt uporablja **trajen spomin** za shranjevanje zgodovine klepeta preko **SQLite baze**. 
Ob zagonu katere koli izmed zgornjih skript se v korenskem direktoriju samodejno ustvari mapa `db/`, ki vsebuje vse potrebne datoteke za delovanje baze.
- Zagon API vmesnika uporablja **FastAPI**.
