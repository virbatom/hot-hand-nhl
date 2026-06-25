# The Hot Hand in the NHL

Code for the master's thesis *The Hot Hand in the NHL: An Analysis of Players'
Shooting Behaviour* (Prague University of Economics and Business, FIS).

The project tests whether scoring a goal changes how much an NHL player shoots
over the rest of the same game, and whether any change reflects the player's own
momentum or his team's strategic response to the score. It uses fifteen regular
seasons (2010–11 to 2024–25) of five-on-five play-by-play data and estimates
Poisson, negative binomial, and hurdle regressions on a player–game panel.

## Repository structure

```
.
├── src/                     # all Python scripts (numbered pipeline + scraper)
│   ├── 01_convert_to_parquet.py
│   ├── 02_sanity_check.py
│   ├── 03_player_game_appearances.py
│   ├── 04_player_history.py
│   ├── 05_build_panel.py
│   ├── 06_add_toi.py
│   ├── 07a_models.py … 17_advanced_models_figures.py
│   ├── scrape_nhl_eventdata.py    # NHL play-by-play / shift scraper
│   └── scrape_few_matches.py      # wrapper to scrape a few games for validation
├── data/
│   ├── raw/                 # input CSVs            (not in repo — see "Data")
│   ├── interim/             # intermediate parquet  (generated)
│   └── processed/           # final analysis panel  (generated)
├── output/
│   ├── figures/             # figures used in the thesis
│   └── tables/              # result tables
├── requirements.txt
└── README.md
```

> **Note on paths.** Every script locates the project root with
> `Path(__file__).resolve().parents[1]`, so the scripts must stay **one level
> below the repository root** (i.e. inside `src/`). Moving them elsewhere breaks
> the relative paths to `data/` and `output/`.

## Data

The data files are **not included in this repository** because they are too
large. Only the code and the (small) generated outputs are tracked. The empty
`data/` sub-folders are kept via `.gitkeep` files so the structure exists after
cloning.

To reproduce the dataset, place the following raw files in `data/raw/`:

| File                | Source                                                                                       |
|---------------------|----------------------------------------------------------------------------------------------|
| `NHL_EventData.csv` | NHL public API, scraped with `src/scrape_nhl_eventdata.py` or hockey-statistics.com          |
| `NHL_Shifts.csv`    | hockey-statistics.com                                                                        |
| `NHL_Schedule.csv`  | hockey-statistics.com                                                                        |
| `NHL_Players.csv`   | hockey-statistics.com                                                                        |
| `results-and-odds.csv` | hockey-statistics.com                                                                     |

Running the scraper:

```bash
python src/scrape_nhl_eventdata.py            # full event data -> NHL_EventData.csv
python src/scrape_few_matches.py              # a few games, for validation
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.10 or newer.

## Running the pipeline

After the raw files are in `data/raw/`, run the numbered scripts in order. The
pipeline moves data from `raw` → `interim` → `processed`, then produces the
figures and tables in `output/`:

1. **`01`–`02`** — convert the raw CSVs to parquet and run sanity checks.
2. **`03`–`06`** — build the player–game panel: appearances, player history,
   the analysis panel, and time-on-ice columns.
3. **`07`–`08`** — baseline Poisson / negative-binomial models, with and
   without score-state controls.
4. **`09`–`10`** — descriptive statistics and the descriptive figures.
5. **`11`–`15`** — score-state decomposition and the zero-truncated Poisson
   models, on both the game-clock and ice-time exposures.
6. **`16`** — the hurdle model (the thesis's final specification).
7. **`17`** — advanced-model figures.

Example:

```bash
python src/01_convert_to_parquet.py
python src/02_sanity_check.py
# ... continue in numerical order through 17
```

## Notes

- The scraper caches every API response under `nhl_cache/`, so interrupted runs
  resume without re-downloading.
- The full dataset can be regenerated from scratch with the scraper; if you need
  the exact files used in the thesis, they can be shared on request.
