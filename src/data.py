"""Build a clean daily dataset from the public Snam "Dati Operativi" workbook.

The workbook contains one row per hourly publication. A gas day runs from
06:00 to 06:00, so each row is identified by the gas day it belongs to plus
the publication hour (06, 07, ..., 23, 00, ..., 05).

Sheets used
-----------
Storico_G-1 : provisional final values of the previous gas day (actuals)
Storico_G   : intraday estimate of the current gas day
Storico_G+1 : Snam's published forecast for the next gas day (benchmark),
              including the forecast heating degree-days (HDD)

Every forecast in this project is issued on gas day D at a fixed hour
(ISSUE_HOUR) for gas day D+1, using only what was published by then.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "DatiOperativi_2023-IT.xlsx"
PROCESSED_FILE = ROOT / "data" / "processed" / "daily_2023.csv"

ISSUE_HOUR = 12  # publication hour used as the forecast issue time

DEMAND = "RICONSEGNATO (1)"
HDD = "Gradi Giorno Medi Ponderali Italia 18°C"

# Italian national holidays in the period covered by the data
HOLIDAYS_IT = pd.to_datetime([
    "2023-01-01", "2023-01-06", "2023-04-09", "2023-04-10", "2023-04-25",
    "2023-05-01", "2023-06-02", "2023-08-15", "2023-11-01", "2023-12-08",
    "2023-12-25", "2023-12-26", "2024-01-01",
])


def _read_sheet(sheet: str) -> pd.DataFrame:
    df = pd.read_excel(RAW_FILE, sheet_name=sheet)
    parts = df["pubblicazione"].str.split("_", expand=True)
    df["gas_day"] = pd.to_datetime(parts[0] + "-" + parts[1] + "-" + parts[2])
    df["pub_hour"] = parts[3].astype(int)
    # position of the publication inside the gas day (06:00 -> 0, 05:00 -> 23)
    df["pub_order"] = (df["pub_hour"] - 6) % 24
    # a few hours were re-published (suffix "_1"): keep the latest version
    df["updated_at"] = pd.to_datetime(df["aggiornato il"], dayfirst=True)
    df = (df.sort_values("updated_at")
            .drop_duplicates(["gas_day", "pub_hour"], keep="last"))
    return df


def build_daily() -> pd.DataFrame:
    """Return one row per target gas day with actuals and issue-time info."""
    g_minus_1 = _read_sheet("Storico_G-1")
    g_today = _read_sheet("Storico_G")
    g_plus_1 = _read_sheet("Storico_G+1")

    # Actuals: last publication of day D+1 holds the settled value of day D
    last = (g_minus_1.sort_values("pub_order")
            .groupby("gas_day").tail(1).set_index("gas_day").sort_index())
    actual = pd.DataFrame({
        "demand": last[DEMAND],
        "hdd": last[HDD],
    })
    actual.index = actual.index - pd.Timedelta(days=1)

    # Value of day D-1 as it was known at issue time on day D
    known = g_minus_1[g_minus_1["pub_hour"] == ISSUE_HOUR].set_index("gas_day")
    known_prev = pd.DataFrame({
        "demand_prev_at_issue": known[DEMAND],
        "hdd_prev_at_issue": known[HDD],
    })
    # issue day D -> target day D+1 (so D-1 is two days before the target)
    known_prev.index = known_prev.index + pd.Timedelta(days=1)

    # Snam's intraday estimate of today's (day D) total, at issue time
    today = g_today[g_today["pub_hour"] == ISSUE_HOUR].set_index("gas_day")
    today_est = pd.DataFrame({
        "demand_today_at_issue": today[DEMAND],
        "hdd_today_at_issue": today[HDD],
    })
    today_est.index = today_est.index + pd.Timedelta(days=1)

    # Snam forecast for D+1 published on day D at issue time
    fc = g_plus_1[g_plus_1["pub_hour"] == ISSUE_HOUR].set_index("gas_day")
    snam = pd.DataFrame({
        "snam_forecast": fc[DEMAND],
        "hdd_forecast": fc[HDD],
    })
    snam.index = snam.index + pd.Timedelta(days=1)

    daily = (actual.join(known_prev, how="left")
                   .join(today_est, how="left")
                   .join(snam, how="left"))
    daily.index.name = "date"
    daily["is_holiday"] = daily.index.isin(HOLIDAYS_IT).astype(int)
    return daily


if __name__ == "__main__":
    daily = build_daily()
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(PROCESSED_FILE)
    print(f"Saved {len(daily)} days to {PROCESSED_FILE}")
    print(daily.describe().round(1))
