"""
Lightweight wrapper around scrape_nhl_data.py that produces the FULL 52-column
output but only for a handful of specific games (no full-season discovery).

Usage:
    python scrape_few_matches.py                       # default sample game IDs
    python scrape_few_matches.py --game-ids 2024020001 2024020002
    python scrape_few_matches.py --count 5             # first 5 reg-season games of 2024-25
    python scrape_few_matches.py --season 2023 --count 3
"""
from __future__ import annotations
import argparse
from pathlib import Path

import scrape_nhl_eventdata as nhl

# Default sample: first 3 games of the 2024-25 regular season
DEFAULT_GAME_IDS = [2024020001, 2024020002, 2024020003]
OUT_CSV = Path("NHL_FewMatches.csv")


def first_n_regular_season_ids(season_start: int, count: int) -> list[int]:
    """Probe sequential reg-season game IDs and return the first `count` that exist."""
    found: list[int] = []
    misses = 0
    for n in range(1, 2000):
        gid = int(f"{season_start}02{n:04d}")
        if nhl._pbp_exists(gid):
            found.append(gid)
            misses = 0
            if len(found) >= count:
                break
        else:
            misses += 1
            if misses >= 20:
                break
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game-ids", nargs="+", type=int, default=None,
                    help="explicit NHL game IDs to scrape")
    ap.add_argument("--season", type=int, default=2024,
                    help="season start year when using --count (default: 2024)")
    ap.add_argument("--count", type=int, default=None,
                    help="grab the first N regular-season games of --season")
    ap.add_argument("--out", type=Path, default=OUT_CSV,
                    help=f"output CSV path (default: {OUT_CSV})")
    ap.add_argument("--no-shifts", action="store_true",
                    help="skip /shiftcharts calls (leaves on-ice cols blank, runs faster)")
    args = ap.parse_args()

    nhl.CACHE_DIR.mkdir(exist_ok=True)

    if args.game_ids:
        game_ids = args.game_ids
    elif args.count:
        print(f"Looking up first {args.count} reg-season games of "
              f"{args.season}-{args.season + 1}...")
        game_ids = first_n_regular_season_ids(args.season, args.count)
    else:
        game_ids = DEFAULT_GAME_IDS

    print(f"Scraping {len(game_ids)} game(s): {game_ids}")

    all_rows: list[dict] = []
    for gid in game_ids:
        rows = nhl.parse_play_by_play(gid)
        if not rows:
            print(f"  game {gid}: no plays returned, skipping")
            continue
        if not args.no_shifts:
            nhl.add_on_ice(gid, rows)
        else:
            # still need the empty on-ice columns to exist for the final write
            for r in rows:
                for side in ("Home", "Away"):
                    for grp in ("Forwards", "Defenders"):
                        r[f"{side}_{grp}_ID"] = None
                        r[f"{side}_{grp}"] = None
                    r[f"{side}_Goalie_ID"] = None
                    r[f"{side}_Goalie"] = None
                r["ShiftIndex"] = None
        all_rows.extend(rows)
        print(f"  game {gid}: {len(rows)} events")

    if not all_rows:
        print("No events collected.")
        return

    # finalize_and_write handles BoxID/BoxID_rev/BoxSize, xG_F/xG_S, column ordering
    nhl.finalize_and_write(all_rows, args.out)


if __name__ == "__main__":
    main()
