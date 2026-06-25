#!/usr/bin/env python3
"""
scrape_nhl_eventdata.py

Scrape NHL play-by-play from the public NHL API and write it in the exact
54-column layout of hockey-statistics.com's `NHL_EventData` file, saved as
`NHL_EventData.csv`.

Data sources used

* Play-by-play : https://api-web.nhle.com/v1/gamecenter/{gameId}/play-by-play
* Shift charts : https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={gameId}
* Player info  : https://api-web.nhle.com/v1/player/{playerId}/landing   (position, handedness)

Which columns are exact vs. approximate

EXACT (straight from the play-by-play, or simple math on it):
    GameID, Season, SeasonState, Venue, Period, GameTime, StrengthState, TypeCode,
    Event, x, y, Zone, Reason, ShotType, SecondaryReason, TypeCode2, PEN_Duration,
    EventTeam, Goalie_ID, Goalie, Player1..3(_ID), Corsi, Fenwick, Shot, Goal,
    EventIndex, ScoreState, ShotDistance, ShotAngle, Position
FROM player landing lookup:
    Shoots
FROM shift charts (reconstructed on-ice rosters):
    Home/Away_Forwards/Defenders/Goalie (+_ID), ShiftIndex

Coverage note

The API only carries real play-by-play (coordinates, situations, shifts) from about the
2010-11 season onward. Earlier seasons return little or nothing and are skipped quickly.
You asked to sweep the full 1917-2025 range, so START_SEASON defaults to 1917; bump it to
2010 to avoid hours of empty requests if you only need the modern era.

Requirements

    pip install requests pandas
    pip install scikit-learn        # only needed for the xG columns; without it xG is left blank

Usage

    python scrape_nhl_eventdata.py                 # full run -> NHL_EventData.csv
    python scrape_nhl_eventdata.py --selftest      # offline checks of the math/parsing helpers
    python scrape_nhl_eventdata.py --seasons 2023 2024   # just 2023-24 and 2024-25

Re-runs are cheap: every API response is cached under ./nhl_cache/, so interrupting and
restarting resumes without re-downloading.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import requests
import pandas as pd

# Configuration

START_SEASON = 2023          # season start year of the FIRST season to scrape (1917 => 1917-18)
END_SEASON   = 2024          # season start year of the LAST season (2024 => 2024-25)
GAME_TYPES   = (2, 3)        # 2 = regular season, 3 = playoffs

OUT_CSV   = Path("NHL_EventData.csv")
CACHE_DIR = Path("nhl_cache")            # on-disk JSON cache (resume-friendly)
THROTTLE  = 0.35                          # polite delay (seconds) between live requests
MAX_REG_GAME = 1500                       # upper bound on regular-season game number to probe
MISS_STREAK_STOP = 40                     # stop probing a season after this many consecutive 404s
CHECKPOINT_EVERY = 250                    # write partial CSV every N games

API_WEB   = "https://api-web.nhle.com/v1"
API_STATS = "https://api.nhle.com/stats/rest/en"

# Exact output column order (must match the target file byte-for-byte in header)
COLUMNS = [
    "GameID", "Season", "SeasonState", "Venue", "Period", "GameTime", "StrengthState",
    "TypeCode", "Event", "x", "y", "Zone", "Reason", "ShotType", "SecondaryReason",
    "TypeCode2", "PEN_Duration", "EventTeam", "Goalie_ID", "Goalie", "Player1_ID",
    "Player1", "Player2_ID", "Player2", "Player3_ID", "Player3", "Corsi", "Fenwick",
    "Shot", "Goal", "EventIndex", "ShiftIndex", "ScoreState", "Home_Forwards_ID",
    "Home_Forwards", "Home_Defenders_ID", "Home_Defenders", "Home_Goalie_ID",
    "Home_Goalie", "Away_Forwards_ID", "Away_Forwards", "Away_Defenders_ID",
    "Away_Defenders", "Away_Goalie_ID", "Away_Goalie", "BoxID", "BoxID_rev", "BoxSize",
    "ShotDistance", "ShotAngle", "Position", "Shoots", "xG_F", "xG_S",
]

# Shot-attempt flag lookup keyed by the API's typeDescKey
#                 Corsi Fenwick Shot Goal
SHOT_FLAGS = {
    "goal":         (1, 1, 1, 1),
    "shot-on-goal": (1, 1, 1, 0),
    "missed-shot":  (1, 1, 0, 0),
    "blocked-shot": (1, 0, 0, 0),
}

POS_GROUP = {"C": "F", "L": "F", "R": "F", "C/L": "F", "L/R": "F", "D": "D", "G": "G"}

GOAL_X = 89.0   # x-coordinate of the goal line; coordinates are normalised so the attack is +x


# HTTP with retries + on-disk cache

_session = requests.Session()
_session.headers.update({"User-Agent": "nhl-eventdata-scraper/1.0 (academic use)"})


def _get_json(url: str, cache_key: str | None = None, allow_404: bool = True):
    """GET a URL as JSON, with disk caching and retry/backoff. Returns dict or None."""
    if cache_key is not None:
        cache_path = CACHE_DIR / cache_key
        if cache_path.exists():
            txt = cache_path.read_text(encoding="utf-8")
            return None if txt == "" else json.loads(txt)

    data = None
    for attempt in range(5):
        try:
            r = _session.get(url, timeout=30)
            if r.status_code == 404:
                data = None
                break
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            break
        except (requests.RequestException, ValueError):
            time.sleep(1.0 * (attempt + 1))
    else:
        data = None

    time.sleep(THROTTLE)
    if cache_key is not None:
        (CACHE_DIR / cache_key).write_text("" if data is None else json.dumps(data),
                                           encoding="utf-8")
    return data


# Small pure helpers (validated by --selftest)

def mmss_to_sec(t: str) -> int:
    """'MM:SS' -> integer seconds. Returns 0 on bad input."""
    try:
        m, s = t.split(":")
        return int(m) * 60 + int(s)
    except (ValueError, AttributeError):
        return 0


def game_time_seconds(period: int, time_in_period: str) -> int:
    """Elapsed seconds since the opening faceoff (regulation periods are 1200 s each)."""
    return (period - 1) * 1200 + mmss_to_sec(time_in_period)


def shot_distance(x: float, y: float) -> float:
    """Distance to the attacking goal (at +89). x is the normalised coordinate."""
    return math.sqrt((GOAL_X - x) ** 2 + y ** 2)


def shot_angle(x: float, y: float) -> float:
    """Signed angle (degrees) to the attacking goal; sign convention matches the target file."""
    return -math.degrees(math.atan2(y, GOAL_X - x))


def strength_state(situation_code: str | None, event_is_home: bool):
    """
    Decode the 4-digit situationCode into the event team's strength state.

    situationCode = 'A B C D' with
        A = away goalie in net (1) / pulled (0)
        B = away skater count
        C = home skater count
        D = home goalie in net (1) / pulled (0)
    Returns a string like '5v5', '5v4', or 'ENF'/'ENA' when a net is empty, or None.
    """
    if not situation_code or len(situation_code) != 4 or not situation_code.isdigit():
        return None
    away_g, away_s, home_s, home_g = (int(c) for c in situation_code)
    own_goalie = home_g if event_is_home else away_g
    opp_goalie = away_g if event_is_home else home_g
    if own_goalie == 0:
        return "ENF"          # event team has pulled its own goalie (empty net for)
    if opp_goalie == 0:
        return "ENA"          # opponent net empty (empty net against)
    return f"{home_s}v{away_s}" if event_is_home else f"{away_s}v{home_s}"


def box_label(x: float, y: float, zone: str | None):
    """
    Approximate hockey-statistics.com's rink-grid label from NORMALISED coordinates
    (event team attacks +x). Returns (BoxID, BoxID_rev, BoxSize).

    NOTE: This reproduces the *form* of their scheme (zone letter O/D/N + a bin number,
    with `_rev` being the same cell from the other team's perspective). The exact integer
    index and area are an approximation, not their proprietary values.
    """
    if x is None or y is None or (isinstance(x, float) and math.isnan(x)):
        return "N02", "N05", 875          # the default neutral cell used for coordinate-less events

    letter = zone if zone in ("O", "D", "N") else ("O" if x > 25 else "D" if x < -25 else "N")

    # x distance from the attacking goal, binned; y binned into lateral bands.
    dxg = GOAL_X - x
    xb = min(max(int(dxg // 15), 0), 12)              # 0..12 distance bands (~15 ft each)
    yb = min(max(int((y + 42.5) // 17), 0), 4)        # 0..4 lateral bands (~17 ft each)
    num = xb * 5 + yb + 1                              # 1..65 within a zone (clamped below)
    num = min(num, 26)                                 # observed numbers top out around 26

    rev_letter = {"O": "D", "D": "O", "N": "N"}[letter]
    rev_yb = 4 - yb
    rev_num = min(xb * 5 + rev_yb + 1, 26)
    box_size = int(15 * 17)                            # ~cell area in ft^2 (approx, 255)
    return f"{letter}{num:02d}", f"{rev_letter}{rev_num:02d}", box_size

# Player position / handedness cache (one landing lookup per unique player)

_player_cache: dict[int, dict] = {}


def player_info(player_id: int) -> dict:
    """Return {'pos': 'F'/'D'/'G', 'shoots': 'L'/'R'/None} for a player id (cached)."""
    if player_id in _player_cache:
        return _player_cache[player_id]
    info = {"pos": None, "shoots": None}
    if player_id and player_id > 0:
        data = _get_json(f"{API_WEB}/player/{player_id}/landing",
                         cache_key=f"player_{player_id}.json")
        if data:
            info["pos"] = POS_GROUP.get(data.get("position"), None)
            info["shoots"] = data.get("shootsCatches")
    _player_cache[player_id] = info
    return info

# Game discovery (probe constructed game IDs; works across all eras, needs no team list)

def discover_game_ids(season_start: int):
    """Yield candidate gameIds for a season by probing the play-by-play endpoint."""
    yr = season_start
    if 2 in GAME_TYPES:                                   # regular season: {yr}02{0001..}
        miss = 0
        for n in range(1, MAX_REG_GAME + 1):
            gid = int(f"{yr}02{n:04d}")
            if _pbp_exists(gid):
                miss = 0
                yield gid
            else:
                miss += 1
                if miss >= MISS_STREAK_STOP:
                    break
    if 3 in GAME_TYPES:                                   # playoffs: {yr}030{round}{series}{game}
        for rnd in range(1, 5):
            for series in range(1, 9):
                got_any = False
                for game in range(1, 8):
                    gid = int(f"{yr}030{rnd}{series}{game}")
                    if _pbp_exists(gid):
                        got_any = True
                        yield gid
                if not got_any and series > 1:
                    break


def _pbp_exists(game_id: int) -> bool:
    """Cheap existence check (cached)."""
    data = _get_json(f"{API_WEB}/gamecenter/{game_id}/play-by-play",
                     cache_key=f"pbp_{game_id}.json")
    return bool(data) and bool(data.get("plays"))



# Play-by-play parsing

def _player_team(roster: dict, pid) -> int | None:
    return roster.get(pid, {}).get("teamId") if pid else None


def parse_play_by_play(game_id: int):
    """Fetch and flatten one game's play-by-play into a list of partial event dicts."""
    data = _get_json(f"{API_WEB}/gamecenter/{game_id}/play-by-play",
                     cache_key=f"pbp_{game_id}.json")
    if not data or not data.get("plays"):
        return []

    season = data.get("season")
    gtype = data.get("gameType")
    season_state = "playoffs" if gtype == 3 else "regular"
    home_id = data.get("homeTeam", {}).get("id")
    away_id = data.get("awayTeam", {}).get("id")
    team_abbr = {
        home_id: data.get("homeTeam", {}).get("abbrev"),
        away_id: data.get("awayTeam", {}).get("abbrev"),
    }

    # rosterSpots: playerId -> {name, posGroup, teamId}
    roster: dict[int, dict] = {}
    for sp in data.get("rosterSpots", []):
        pid = sp.get("playerId")
        nm = f"{sp.get('firstName', {}).get('default', '')} {sp.get('lastName', {}).get('default', '')}".strip()
        roster[pid] = {
            "name": nm or None,
            "pos": POS_GROUP.get(sp.get("positionCode"), None),
            "teamId": sp.get("teamId"),
        }

    def name_of(pid):
        return roster.get(pid, {}).get("name") if pid else None

    rows = []
    home_goals = away_goals = 0          # running score for ScoreState

    for pl in data["plays"]:
        d = pl.get("details", {}) or {}
        type_key = pl.get("typeDescKey")
        type_code = pl.get("typeCode")
        period = pl.get("periodDescriptor", {}).get("number")
        tip = pl.get("timeInPeriod", "00:00")
        sort_order = pl.get("sortOrder", 0)

        # event team (eventOwnerTeamId), with blocked-shot fixed to the SHOOTING team
        ev_team_id = d.get("eventOwnerTeamId")
        goalie_id = d.get("goalieInNetId")
        p1 = p2 = p3 = None
        reason = d.get("reason")
        secondary = d.get("secondaryReason")
        shot_type = d.get("shotType")
        pen_dur = None
        type_code2 = None

        if type_key == "goal":
            p1, p2, p3 = d.get("scoringPlayerId"), d.get("assist1PlayerId"), d.get("assist2PlayerId")
        elif type_key in ("shot-on-goal", "missed-shot"):
            p1 = d.get("shootingPlayerId") or d.get("playerId")
        elif type_key == "blocked-shot":
            p1 = d.get("shootingPlayerId")
            p2 = d.get("blockingPlayerId")
            # api-web attributes the event to the blocking team; Corsi is for the shooter,
            # so override EventTeam to the shooter's team when we can resolve it.
            st = _player_team(roster, p1)
            if st is not None:
                ev_team_id = st
        elif type_key == "faceoff":
            p1, p2 = d.get("winningPlayerId"), d.get("losingPlayerId")
        elif type_key == "hit":
            p1, p2 = d.get("hittingPlayerId"), d.get("hitteePlayerId")
        elif type_key in ("giveaway", "takeaway"):
            p1 = d.get("playerId")
        elif type_key == "penalty":
            p1 = d.get("committedByPlayerId") or d.get("servedByPlayerId")
            p2 = d.get("drawnByPlayerId")
            pen_dur = d.get("duration")
            type_code2 = d.get("descKey") or d.get("typeCode")

        ev_home = (ev_team_id == home_id)
        corsi, fenwick, shot, goal = SHOT_FLAGS.get(type_key, (0, 0, 0, 0))

        # ScoreState from the event team's perspective, BEFORE this event's own goal
        score_state = (home_goals - away_goals) if ev_home else (away_goals - home_goals)
        if ev_team_id is None:
            score_state = home_goals - away_goals

        rows.append({
            "GameID": game_id,
            "Season": season,
            "SeasonState": season_state,
            "Venue": ("Home" if ev_home else "Away") if ev_team_id is not None else None,
            "Period": period,
            "GameTime": game_time_seconds(period, tip) if period else None,
            "_situation": pl.get("situationCode"),
            "_ev_home": ev_home,
            "TypeCode": type_code,
            "Event": type_key,
            "_x_raw": d.get("xCoord"),
            "_y_raw": d.get("yCoord"),
            "Zone": d.get("zoneCode"),
            "Reason": reason,
            "ShotType": shot_type,
            "SecondaryReason": secondary,
            "TypeCode2": type_code2,
            "PEN_Duration": pen_dur,
            "EventTeam": team_abbr.get(ev_team_id),
            "_ev_team_id": ev_team_id,
            "Goalie_ID": goalie_id,
            "Goalie": name_of(goalie_id),
            "Player1_ID": p1 or 0,
            "Player1": name_of(p1),
            "Player2_ID": p2 or 0,
            "Player2": name_of(p2),
            "Player3_ID": p3 or 0,
            "Player3": name_of(p3),
            "Corsi": corsi, "Fenwick": fenwick, "Shot": shot, "Goal": goal,
            "EventIndex": int(f"{game_id}{sort_order:04d}"),
            "ScoreState": score_state,
            "_home_id": home_id, "_away_id": away_id,
            "Position": roster.get(p1, {}).get("pos"),
            "_p1": p1,
        })

        if type_key == "goal":
            if ev_home:
                home_goals += 1
            else:
                away_goals += 1

    _normalise_coordinates(rows)
    return rows


def _normalise_coordinates(rows: list[dict]) -> None:
    """
    Flip raw (x, y) per (period, eventTeam) so the event team always attacks +x, matching
    the target file. Direction is inferred from the sign of the team's offensive-zone shot x.
    Writes final 'x','y','ShotDistance','ShotAngle' onto each row.
    """
    # decide flip per (period, team) from offensive-zone shot attempts
    acc: dict[tuple, list] = {}
    for r in rows:
        if r["Corsi"] == 1 and r["Zone"] == "O" and r["_x_raw"] is not None:
            acc.setdefault((r["Period"], r["_ev_team_id"]), []).append(r["_x_raw"])
    flip = {}
    for key, xs in acc.items():
        xs_sorted = sorted(xs)
        med = xs_sorted[len(xs_sorted) // 2]
        flip[key] = med < 0
    for r in rows:
        x, y = r["_x_raw"], r["_y_raw"]
        if x is None or y is None:
            r["x"] = r["y"] = None
            r["ShotDistance"] = r["ShotAngle"] = None
        else:
            if flip.get((r["Period"], r["_ev_team_id"]), False):
                x, y = -x, -y
            r["x"], r["y"] = float(x), float(y)
            if r["Corsi"] == 1:
                r["ShotDistance"] = shot_distance(x, y)
                r["ShotAngle"] = shot_angle(x, y)
            else:
                r["ShotDistance"] = r["ShotAngle"] = None
        # StrengthState now that we know home/away
        r["StrengthState"] = strength_state(r["_situation"], r["_ev_home"])



# Shift charts -> on-ice rosters + ShiftIndex

def add_on_ice(game_id: int, rows: list[dict]) -> None:
    """Reconstruct on-ice skaters/goalies per event from the shift-chart API."""
    # default empties
    for r in rows:
        for side in ("Home", "Away"):
            for grp in ("Forwards", "Defenders"):
                r[f"{side}_{grp}_ID"] = None
                r[f"{side}_{grp}"] = None
            r[f"{side}_Goalie_ID"] = None
            r[f"{side}_Goalie"] = None
        r["ShiftIndex"] = None

    if not rows:
        return
    home_id, away_id = rows[0]["_home_id"], rows[0]["_away_id"]
    data = _get_json(f"{API_STATS}/shiftcharts?cayenneExp=gameId={game_id}",
                     cache_key=f"shifts_{game_id}.json")
    shifts = (data or {}).get("data") or []
    if not shifts:
        return

    # build per-period interval boundaries (for the approximate ShiftIndex)
    boundaries: dict[int, set] = {}
    parsed = []
    for s in shifts:
        per = s.get("period")
        start = mmss_to_sec(s.get("startTime", "00:00"))
        end = mmss_to_sec(s.get("endTime", "00:00"))
        if end <= start:
            continue
        pid = s.get("playerId")
        nm = f"{s.get('firstName', '')} {s.get('lastName', '')}".strip()
        parsed.append((per, start, end, pid, s.get("teamId"), nm))
        boundaries.setdefault(per, set()).update([start, end])
    bnd_sorted = {p: sorted(b) for p, b in boundaries.items()}

    def on_ice(period, sec):
        out = []
        for per, start, end, pid, team_id, nm in parsed:
            if per == period and start <= sec < end:
                out.append((pid, team_id, nm))
        return out

    for r in rows:
        per, gt = r["Period"], r["GameTime"]
        if per is None or gt is None:
            continue
        sec = gt - (per - 1) * 1200
        players = on_ice(per, sec)
        buckets = {(home_id, "F"): [], (home_id, "D"): [], (home_id, "G"): [],
                   (away_id, "F"): [], (away_id, "D"): [], (away_id, "G"): []}
        for pid, team_id, nm in players:
            grp = player_info(pid)["pos"]
            if (team_id, grp) in buckets:
                buckets[(team_id, grp)].append((pid, nm))
        for side, tid in (("Home", home_id), ("Away", away_id)):
            for grp, col in (("F", "Forwards"), ("D", "Defenders")):
                lst = sorted(buckets[(tid, grp)], key=lambda t: t[0])
                if lst:
                    r[f"{side}_{col}_ID"] = " ".join(str(p) for p, _ in lst)
                    r[f"{side}_{col}"] = " - ".join(n for _, n in lst if n)
            g = sorted(buckets[(tid, "G")], key=lambda t: t[0])
            if g:
                r[f"{side}_Goalie_ID"] = float(g[0][0])
                r[f"{side}_Goalie"] = g[0][1]
        # approximate ShiftIndex: gameId + index of the time interval this event falls in
        bl = bnd_sorted.get(per, [])
        idx = sum(1 for b in bl if b <= sec)
        r["ShiftIndex"] = float(f"{game_id}{idx:04d}")



# Expected-goals model (approximation of xG_F / xG_S)

def add_expected_goals(df: pd.DataFrame) -> None:
    """Fit a logistic-regression xG model on the scraped shots and fill xG_F / xG_S.

    xG_F is fit on Fenwick (unblocked) attempts; xG_S on shots on goal. This is a public
    approximation and will NOT equal hockey-statistics.com's proprietary values.
    """
    df["xG_F"] = pd.NA
    df["xG_S"] = pd.NA
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("  [xG] scikit-learn not installed -> leaving xG_F / xG_S blank.")
        return

    def feature_frame(mask):
        sub = df[mask].copy()
        if sub.empty:
            return None, None
        st = sub["ShotType"].fillna("unknown").astype(str)
        feats = pd.DataFrame({
            "dist": sub["ShotDistance"].astype(float),
            "angle": sub["ShotAngle"].abs().astype(float),
            "dist_angle": sub["ShotDistance"].astype(float) * sub["ShotAngle"].abs().astype(float),
        })
        feats = pd.concat([feats, pd.get_dummies(st, prefix="st")], axis=1)
        return sub.index, feats

    for col, mask in (("xG_F", (df["Fenwick"] == 1) & df["ShotDistance"].notna()),
                      ("xG_S", (df["Shot"] == 1) & df["ShotDistance"].notna())):
        idx, X = feature_frame(mask)
        if X is None or df.loc[idx, "Goal"].nunique() < 2:
            continue
        y = df.loc[idx, "Goal"].astype(int).values
        Xv = X.fillna(0.0).values
        model = LogisticRegression(max_iter=1000)
        model.fit(Xv, y)
        df.loc[idx, col] = model.predict_proba(Xv)[:, 1]



# Box columns

def add_box_columns(df: pd.DataFrame) -> None:
    bids, brev, bsize = [], [], []
    for x, y, z in zip(df["x"], df["y"], df["Zone"]):
        a, b, c = box_label(x, y, z)
        bids.append(a); brev.append(b); bsize.append(c)
    df["BoxID"], df["BoxID_rev"], df["BoxSize"] = bids, brev, bsize

# Main driver

def finalize_and_write(rows: list[dict], path: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        print("No events collected; nothing written.")
        return
    add_box_columns(df)
    add_expected_goals(df)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[COLUMNS]
    df.to_csv(path, index=False)
    print(f"Wrote {len(df):,} events x {len(COLUMNS)} cols -> {path}")


def run(seasons):
    CACHE_DIR.mkdir(exist_ok=True)
    all_rows: list[dict] = []
    games_done = 0
    for season_start in seasons:
        print(f"\n=== Season {season_start}-{season_start+1} ===")
        season_games = 0
        for gid in discover_game_ids(season_start):
            rows = parse_play_by_play(gid)
            if not rows:
                continue
            add_on_ice(gid, rows)
            all_rows.extend(rows)
            games_done += 1
            season_games += 1
            if games_done % 25 == 0:
                print(f"  ...{games_done} games, {len(all_rows):,} events "
                      f"(latest {gid})")
            if games_done % CHECKPOINT_EVERY == 0:
                finalize_and_write(all_rows, OUT_CSV)
        print(f"  season complete: {season_games} games with play-by-play")
    finalize_and_write(all_rows, OUT_CSV)
    print("\nDone.")


# Offline self-test of the pure helpers

def selftest():
    ok = True

    def check(name, got, exp, tol=1e-6):
        nonlocal ok
        good = (abs(got - exp) <= tol) if isinstance(exp, (int, float)) else (got == exp)
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {name}: got={got!r} exp={exp!r}")

    # geometry (values verified against the sneak peek)
    check("dist(62,22)", shot_distance(62, 22), 34.828149534536, tol=1e-6)
    check("angle(62,22)", shot_angle(62, 22), -39.1736579614997, tol=1e-4)
    check("dist(-54,-15) far goal", shot_distance(-54, -15), 143.784561, tol=1e-3)
    check("angle(63,-11)", shot_angle(63, -11), 22.932100, tol=1e-3)

    # time
    check("game_time P3 04:17", game_time_seconds(3, "04:17"), 2657)
    check("mmss 12:30", mmss_to_sec("12:30"), 750)

    # strength state
    check("5v5 home", strength_state("1551", True), "5v5")
    check("4v5 away (CAR)", strength_state("1451", False), "4v5")
    check("5v4 home (1451)", strength_state("1451", True), "5v4")
    check("4v5 home (1541)", strength_state("1541", True), "4v5")
    check("ENF own goalie pulled (home)", strength_state("1550", True), "ENF")
    check("ENA opp goalie pulled (home)", strength_state("0551", True), "ENA")
    check("bad code", strength_state("", True), None)

    # shot flags
    check("goal flags", SHOT_FLAGS["goal"], (1, 1, 1, 1))
    check("blocked flags", SHOT_FLAGS["blocked-shot"], (1, 0, 0, 0))

    # box label shape + null default
    bid, brev, bsz = box_label(62, 22, "O")
    check("box letter O", bid[0], "O")
    check("box rev letter D", brev[0], "D")
    check("box null default", box_label(float("nan"), float("nan"), None)[0], "N02")

    print("\nSELF-TEST:", "ALL PASSED" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Scrape NHL play-by-play into NHL_EventData.csv")
    ap.add_argument("--selftest", action="store_true", help="run offline checks and exit")
    ap.add_argument("--seasons", nargs="+", type=int, default=None,
                    help="season start years to scrape (default: START_SEASON..END_SEASON)")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    season_list = args.seasons if args.seasons else list(range(START_SEASON, END_SEASON + 1))
    run(season_list)
