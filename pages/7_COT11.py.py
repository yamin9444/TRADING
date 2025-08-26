#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# app_streamlit.py
# COT Dashboard — FX & Commodities (Legacy Futures Only, Socrata)
# Patches: since 2005, date-aggregation, colored net line, robust fetch, NO CACHE.

import os, time, requests
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="COT Dashboard — FX & Commodities", layout="wide")

# --- assure qu'aucune donnée précédente ne reste en cache Streamlit ---
try:
    st.cache_data.clear()
except Exception:
    pass

# ------------- Config -------------
DOMAIN = "https://publicreporting.cftc.gov"
DATASET_ID = "6dca-aqww"               # Legacy - Futures Only
SINCE_DATE = "2005-01-01"              # point de départ des séries

FX_TARGETS = {
    "U.S. DOLLAR INDEX": "USD",
    "EURO FX": "EUR",
    "BRITISH POUND": "GBP",
    "JAPANESE YEN": "JPY",
    "CANADIAN DOLLAR": "CAD",
    "AUSTRALIAN DOLLAR": "AUD",
    "SWISS FRANC": "CHF",
    "NEW ZEALAND DOLLAR": "NZD",
}
COMMO_TARGETS = {
    "GOLD": "Gold",
    "SILVER": "Silver",
    "COPPER": "Copper",
    "CRUDE OIL WTI": "Crude Oil",
    "CRUDE OIL, LIGHT SWEET": "Crude Oil",  # alias WTI
    "NATURAL GAS": "Nat Gas",
    "CORN": "Corn",
    "WHEAT": "Wheat",
}

TARGET_COLS = [
    "report_date_as_yyyy_mm_dd",
    "report_date",
    "market_and_exchange_names",
    "open_interest_all",
    "noncomm_positions_long_all", "noncomm_positions_short_all",
    "comm_positions_long_all", "comm_positions_short_all",
]

# ------------- Socrata utils -------------
def _sleep_backoff(i, base=1.5):
    time.sleep(base ** (i + 1))

def socrata_get_json(dataset_id: str, params: dict, max_tries: int = 6):
    url = f"{DOMAIN}/resource/{dataset_id}.json"
    q = dict(params)
    headers = {}
    tok = os.environ.get("SOCRATA_APP_TOKEN")  # optionnel
    if tok:
        headers["X-App-Token"] = tok
    # forcer aucun cache côté proxy éventuel
    headers["Cache-Control"] = "no-cache"

    for i in range(max_tries):
        r = requests.get(url, params=q, headers=headers, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504, 400):
            _sleep_backoff(i)
            if i >= 2 and q.get("$limit", 50000) > 20000:
                q["$limit"] = int(q["$limit"] // 2)
            if i >= 1 and "$order" in q:
                q.pop("$order", None)
            continue
        r.raise_for_status()
    return []

def probe_columns(dataset_id: str) -> tuple[list, str]:
    rows = socrata_get_json(dataset_id, {"$limit": 1})
    if not rows:
        return TARGET_COLS, "report_date_as_yyyy_mm_dd"
    available = rows[0].keys()
    select_cols = [c for c in TARGET_COLS if c in available]
    date_col = (
        "report_date_as_yyyy_mm_dd" if "report_date_as_yyyy_mm_dd" in select_cols
        else ("report_date" if "report_date" in select_cols else "report_date_as_yyyy_mm_dd")
    )
    return select_cols, date_col

SELECT_COLS_SAFE, DATE_COL = probe_columns(DATASET_ID)

def _where_contains(key: str) -> str:
    return f"upper(market_and_exchange_names) like '%{key.upper()}%'"

def _typify(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    dcol = DATE_COL if DATE_COL in df.columns else (
        "report_date_as_yyyy_mm_dd" if "report_date_as_yyyy_mm_dd" in df.columns else "report_date"
    )
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    for c in df.columns:
        if c not in (":id", "market_and_exchange_names"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=[dcol]).sort_values(dcol)

# --- MARCHÉS PRÉFÉRÉS (USD ICE, NZD CME) ---
PREFERRED_MARKETS = {
    "U.S. DOLLAR INDEX": r"USD INDEX.*ICE FUTURES U\.S\.",
    "NEW ZEALAND DOLLAR": r"NZ DOLLAR.*CHICAGO MERCANTILE EXCHANGE",
}

def fetch_single_target(key: str) -> pd.DataFrame:
    def _dl(params):
        rows, offset = [], 0
        while True:
            params["$offset"] = offset
            chunk = socrata_get_json(DATASET_ID, params)
            if not chunk:
                break
            rows.extend(chunk)
            if len(chunk) < params.get("$limit", 50000):
                break
            offset += params.get("$limit", 50000)
        return _typify(pd.DataFrame.from_records(rows))

    # 1) contains + since date
    params = {
        "$select": ",".join(SELECT_COLS_SAFE),
        "$where": f"{DATE_COL} >= '{SINCE_DATE}' AND " + _where_contains(key),
        "$order": DATE_COL,
        "$limit": 50000,
    }
    df = _dl(params)

    # 2) $q + since date (fallback)
    if df.empty:
        params = {
            "$select": ",".join(SELECT_COLS_SAFE),
            "$q": key,
            "$where": f"{DATE_COL} >= '{SINCE_DATE}'",
            "$order": DATE_COL,
            "$limit": 50000,
        }
        df = _dl(params)
        if not df.empty:
            mask = df["market_and_exchange_names"].str.upper().str.contains(key.upper(), na=False)
            df = df[mask].copy()

    # 3) fallback massif récent
    if df.empty:
        params = {
            "$select": ",".join(SELECT_COLS_SAFE),
            "$where": f"{DATE_COL} >= '{SINCE_DATE}'",
            "$order": DATE_COL,
            "$limit": 200000,
        }
        df = _dl(params)
        if df.empty:
            return df
        mask = df["market_and_exchange_names"].str.upper().str.contains(key.upper(), na=False)
        df = df[mask].copy()

    # --- Sélection du marché correct ---
    if df.empty or "market_and_exchange_names" not in df.columns:
        return df

    # a) marché préféré s'il existe
    pref = PREFERRED_MARKETS.get(key)
    if pref:
        pref_df = df[df["market_and_exchange_names"].str.contains(pref, case=False, regex=True, na=False)]
        if not pref_df.empty:
            df = pref_df

    # b) sinon : marché avec la date max la plus récente
    dcol = DATE_COL if DATE_COL in df.columns else (
        "report_date_as_yyyy_mm_dd" if "report_date_as_yyyy_mm_dd" in df.columns else "report_date"
    )
    if dcol in df.columns and not df.empty:
        grp = df.groupby("market_and_exchange_names")[dcol].max()
        if not grp.empty:
            best_market = grp.sort_values(ascending=False).index[0]
            df = df[df["market_and_exchange_names"] == best_market].copy()

    return df

def fetch_many(keys: list[str]) -> pd.DataFrame:
    dfs = []
    for k in keys:
        with st.spinner(f"Téléchargement : {k}"):
            d = fetch_single_target(k)
            if not d.empty:
                d["__key"] = k
                dfs.append(d)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ------------- Série + résumé -------------
def compute_series(df: pd.DataFrame, series_kind: str, ma_weeks: int) -> pd.DataFrame:
    """Agréger par date, calculer MA & COT index, filtrer depuis SINCE_DATE."""
    dcol = DATE_COL if DATE_COL in df.columns else (
        "report_date_as_yyyy_mm_dd" if "report_date_as_yyyy_mm_dd" in df.columns else "report_date"
    )
    if dcol not in df.columns or df.empty:
        return pd.DataFrame(columns=["date", "net", "ma", "cot_index", "oi"])

    if series_kind == "Non-Commercial":
        net = df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]
    else:
        net = df["comm_positions_long_all"] - df["comm_positions_short_all"]

    tmp = pd.DataFrame({
        "date": pd.to_datetime(df[dcol], errors="coerce"),
        "net":  pd.to_numeric(net, errors="coerce"),
        "oi":   pd.to_numeric(df.get("open_interest_all"), errors="coerce"),
    }).dropna(subset=["date"])

    # Agrégation par date (supprime doublons/échéances)
    tmp = tmp.groupby("date", as_index=False).sum(numeric_only=True)

    # Filtre visuel depuis 2005
    tmp = tmp[tmp["date"] >= pd.Timestamp(SINCE_DATE)].sort_values("date")

    # MA + COT Index
    tmp["ma"] = tmp["net"].rolling(ma_weeks, min_periods=1).mean()
    roll = tmp["net"].rolling(ma_weeks, min_periods=1)
    rmin, rmax = roll.min(), roll.max()
    denom = (rmax - rmin).replace(0, np.nan)
    tmp["cot_index"] = (100 * (tmp["net"] - rmin) / denom).fillna(50.0)
    return tmp

def classify_trend(ma: pd.Series) -> str:
    if ma is None or len(ma) < 2:
        return "flat"
    delta = float(ma.iloc[-1] - ma.iloc[-2])
    rng = float(ma.max() - ma.min())
    eps = max(0.01 * (rng if rng > 0 else 1.0), 500.0)
    return "up" if delta > eps else ("down" if delta < -eps else "flat")

def summarize(series: pd.DataFrame, ma_weeks: int) -> dict:
    if series is None or series.empty:
        return {"last_date": None, "last_net": np.nan, f"MA_{ma_weeks}w": np.nan, "COT_Index": np.nan, "trend": "flat"}
    s = series.dropna(subset=["date"]).sort_values("date")
    last = s.iloc[-1]
    last_net = float(last.get("net", np.nan))
    last_ma  = float(last.get("ma", np.nan))
    last_idx = float(last.get("cot_index", np.nan))
    return {
        "last_date": pd.to_datetime(last["date"]).date(),
        "last_net": round(last_net) if pd.notna(last_net) else np.nan,
        f"MA_{ma_weeks}w": round(last_ma) if pd.notna(last_ma) else np.nan,
        "COT_Index": round(last_idx, 1) if pd.notna(last_idx) else np.nan,
        "trend": classify_trend(s["ma"]),
    }

# ------------- Chart rouge/vert + MA -------------
def plot_colored(series: pd.DataFrame):
    data = series.copy()
    # coupe la ligne quand le signe change
    data["net_pos"] = np.where(data["net"] >= 0, data["net"], np.nan)
    data["net_neg"] = np.where(data["net"] < 0,  data["net"], np.nan)

    base = alt.Chart(data).encode(x=alt.X('date:T', title=''))
    net_pos = base.mark_line(size=1.6, color='#16a34a').encode(y=alt.Y('net_pos:Q', title='Net positions'))
    net_neg = base.mark_line(size=1.6, color='#dc2626').encode(y='net_neg:Q')
    ma_line = base.mark_line(size=1, color='#0ea5e9', opacity=0.9).encode(y='ma:Q')
    zero    = base.mark_rule(color='#9ca3af').encode(y=alt.datum(0))

    st.altair_chart((zero + net_pos + net_neg + ma_line).properties(height=260),
                    use_container_width=True)

# ------------- UI -------------
st.title("COT Dashboard — FX & Commodities")
series_kind = st.radio("Série :", ["Non-Commercial", "Commercial"], horizontal=True)
ma_weeks = st.selectbox("Moyenne mobile (semaines)", [8, 12, 26, 52], index=0)
st.caption(f"Données CFTC (Legacy Futures Only) — filtrées depuis **{SINCE_DATE}**")

tab_fx, tab_cm = st.tabs(["💱 Devises monétaires", "🛢️ Matières premières"])

with tab_fx:
    df_all = fetch_many(list(FX_TARGETS.keys()))
    if df_all.empty:
        st.warning("Aucune donnée récupérée (FX). Réessaie plus tard.")
    else:
        rows = []
        for key, code in FX_TARGETS.items():
            sub = df_all[df_all["market_and_exchange_names"].str.upper().str.contains(key.upper(), na=False)]
            if sub.empty:
                continue
            series = compute_series(sub, series_kind, ma_weeks)
            if series.empty:
                continue
            info = summarize(series, ma_weeks)
            rows.append({"FX": code, "label": key, **info})

            c1, c2 = st.columns([2.7, 1])
            with c1:
                st.markdown(f"### {key}")
                plot_colored(series)
            with c2:
                st.metric("Dernière position nette", f"{info['last_net']:,.0f}".replace(",", " "))
                st.write(f"**Dernière mise à jour :** {info['last_date']}")
                st.write(f"**Tendance (MA {ma_weeks}) :** {info['trend']}")
                st.write(f"**COT Index :** {info['COT_Index']}")

        if rows:
            fx_table = pd.DataFrame(rows).sort_values("FX")
            st.subheader("Tableau récap — FX")
            st.dataframe(fx_table, use_container_width=True)
            st.download_button(
                "⬇️ Télécharger CSV (FX)",
                data=fx_table.to_csv(index=False).encode("utf-8"),
                file_name="fx_summary.csv",
                mime="text/csv",
            )

with tab_cm:
    df_all = fetch_many(list(COMMO_TARGETS.keys()))
    if df_all.empty:
        st.warning("Aucune donnée récupérée (Matières). Réessaie plus tard.")
    else:
        done, rows = set(), []
        for key, label in COMMO_TARGETS.items():
            if label in done:
                continue           # évite doublons WTI
            sub = df_all[df_all["market_and_exchange_names"].str.upper().str.contains(key.upper(), na=False)]
            if sub.empty:
                continue
            series = compute_series(sub, series_kind, ma_weeks)
            if series.empty:
                continue
            info = summarize(series, ma_weeks)
            rows.append({"Commodity": label, "key": key, **info})
            done.add(label)

            c1, c2 = st.columns([2.7, 1])
            with c1:
                st.markdown(f"### {label}")
                plot_colored(series)
            with c2:
                st.metric("Dernière position nette", f"{info['last_net']:,.0f}".replace(",", " "))
                st.write(f"**Dernière mise à jour :** {info['last_date']}")
                st.write(f"**Tendance (MA {ma_weeks}) :** {info['trend']}")
                st.write(f"**COT Index :** {info['COT_Index']}")

        if rows:
            cm_table = pd.DataFrame(rows).drop_duplicates(subset=["Commodity"]).sort_values("Commodity")
            st.subheader("Tableau récap — Matières premières")
            st.dataframe(cm_table, use_container_width=True)
            st.download_button(
                "⬇️ Télécharger CSV (Commodities)",
                data=cm_table.to_csv(index=False).encode("utf-8"),
                file_name="commodities_summary.csv",
                mime="text/csv",
            )

