#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# app.py — Big Picture Leading (layout Excel + robust FRED)
# Lancer : streamlit run app.py
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# ============ CONFIG ============
FRED_API_KEY = "b81975b746a1ad458cb81ca5843153e6"

SERIES_GROUPS = {
    "Consommation": {
        "REVOLSL": "Crédit Consommateur (Revolving)",
        "LES1252881600Q": "Salaire médian réel hebdo (1982-84$)",
        "RSXFS": "Ventes au détail (Advance Retail Sales)",
        "A229RX0": "Revenu dispo réel par habitant",
    },
    "Immobilier": {
        "REALLN": "Real Estate Loans (banques commerciales)",
        "PERMIT": "Permis de construire",
        "HSN1F": "Maisons unifamiliales vendues (US)",
        "PRRESCON": "Construction résidentielle privée (dépenses)",
    },
}

MA = {"CT": 4, "MT": 8, "LT": 12}
CUT = {"CT": "2023-01-01", "MT": "2010-01-01", "LT": None}

SECTEURS = {
    "Expansion": {"UP": "Consumer Discretionary, Financials, Industrials",
                  "DOWN": "Utilities, Health Care"},
    "Tension": {"UP": "Utilities, Consumer Staples, Health Care",
                "DOWN": "Consumer Discretionary, Tech, Industrials"},
    "Cycle incertain": {"UP": "Health Care, Utilities, Telecom",
                        "DOWN": "Cyclicals, Discretionary, Industrials"},
}

# Matplotlib sombre
plt.rcParams.update({
    "figure.facecolor": "#2b2b2b",
    "axes.facecolor": "#2b2b2b",
    "savefig.facecolor": "#2b2b2b",
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "#666",
    "axes.titleweight": "bold",
})

st.set_page_config(page_title="Big Picture Leading", layout="wide", page_icon="📊")

# ============ FRED (robuste) ============
@st.cache_data(show_spinner=False)
def get_fred(series_id: str) -> pd.DataFrame:
    """Retourne DataFrame(date,value) — jamais d’exception, peut être vide."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": FRED_API_KEY, "file_type": "json"}
    try:
        r = requests.get(url, params=params, timeout=30)
        if not r.ok:
            return pd.DataFrame(columns=["date", "value"])
        js = r.json()
        obs = js.get("observations", [])
        df = pd.DataFrame(obs)
        if df.empty:
            return pd.DataFrame(columns=["date", "value"])
        df = df[df["value"] != "."]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date", "value"]).sort_values("date")
        return df[["date", "value"]]
    except Exception:
        return pd.DataFrame(columns=["date", "value"])

def add_ma_phase_score(df: pd.DataFrame, window: int) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return pd.DataFrame(columns=["date","value","MA","Phase","Score"])
    out["MA"] = out["value"].rolling(window).mean()
    def rule(v, m):
        if pd.isna(m): return "Neutre", 0
        if v > m * 1.0025: return "Expansion", 1
        if v < m * 0.99:   return "Tension", -1
        return "Neutre", 0
    phases_scores = out.apply(lambda r: rule(r["value"], r["MA"]), axis=1)
    out["Phase"] = [ps[0] for ps in phases_scores]
    out["Score"] = [ps[1] for ps in phases_scores]
    return out

def slice_from(df: pd.DataFrame, start: str|None) -> pd.DataFrame:
    if df.empty: return df
    return df[df["date"] >= pd.to_datetime(start)] if start else df

def last_phase_score(df: pd.DataFrame) -> tuple[str,int]:
    if df.empty: return "Neutre", 0
    return str(df["Phase"].iloc[-1]), int(df["Score"].iloc[-1])

def plot_panel(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    if df.empty:
        ax.set_title(title, fontsize=14, color="white")
        ax.text(0.5, 0.5, "Données indisponibles", ha="center", va="center", color="white")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
        st.pyplot(fig, use_container_width=True)
        return
    ax.plot(df["date"], df["value"], color="#ff8c00", lw=2, label="Valeur")
    ax.plot(df["date"], df["MA"], color="#ffffff", lw=2, label="Moyenne mobile")
    ax.grid(True, alpha=0.25)
    for s in ax.spines.values(): s.set_edgecolor("#888")
    ax.set_title(title, fontsize=14, color="white")
    ax.legend(facecolor="#2b2b2b", edgecolor="#888")
    st.pyplot(fig, use_container_width=True)

def cycle_global(score_total: int) -> str:
    if score_total >= 2: return "Expansion"
    if score_total <= -2: return "Tension"
    return "Cycle incertain"

# ============ UI ============
st.markdown("## 📚 Visualisation des indicateurs Leading Consommation & Immobilier (CT=4, MT=8, LT=12)")

# Boucle : par bloc (Conso/Immo), puis 4 séries
for bloc, items in SERIES_GROUPS.items():
    st.markdown(f"### {bloc}")
    for sid, name in items.items():
        df_all = get_fred(sid)

        # Préparer les 3 fenêtres
        df_ct = add_ma_phase_score(slice_from(df_all, CUT["CT"]), MA["CT"])
        df_mt = add_ma_phase_score(slice_from(df_all, CUT["MT"]), MA["MT"])
        df_lt = add_ma_phase_score(slice_from(df_all, CUT["LT"]), MA["LT"])

        # Affichage : 3 graphes côte à côte + commentaires
        c1, c2, c3, c4 = st.columns([3,3,3,2], gap="small")

        with c1:
            plot_panel(df_ct, f"{sid} · Court Terme (MA {MA['CT']})")
        with c2:
            plot_panel(df_mt, f"{sid} · Moyen Terme (MA {MA['MT']})")
        with c3:
            plot_panel(df_lt, f"{sid} · Long Terme (MA {MA['LT']})")
        with c4:
            st.markdown("#### Commentaires")
            p, s = last_phase_score(df_ct)
            st.write(f"**Cycle CT** : {p}")
            st.write(f"**Score CT** : {s}")
            st.write("---")
            p, s = last_phase_score(df_mt)
            st.write(f"**Cycle MT** : {p}")
            st.write(f"**Score MT** : {s}")
            st.write("---")
            p, s = last_phase_score(df_lt)
            st.write(f"**Cycle LT** : {p}")
            st.write(f"**Score LT** : {s}")

        st.markdown("""<hr style="border:0;height:1px;background:#444;margin:10px 0 20px 0;">""",
                    unsafe_allow_html=True)

# ============ SYNTHÈSES EN BAS ============
st.markdown("---")
st.markdown("## 📌 Synthèse Globale (QoQ)")

def score_total_ct(series_ids: dict) -> int:
    total = 0
    for sid in series_ids.keys():
        df = get_fred(sid)
        df_ct = add_ma_phase_score(slice_from(df, CUT["CT"]), MA["CT"])
        if not df_ct.empty:
            total += int(df_ct["Score"].iloc[-1])
    return total

# Consommation
score_conso = score_total_ct(SERIES_GROUPS["Consommation"])
cycle_conso = cycle_global(score_conso)

# Immobilier
score_immo = score_total_ct(SERIES_GROUPS["Immobilier"])
cycle_immo = cycle_global(score_immo)

cc1, cc2 = st.columns(2)

with cc1:
    st.markdown("### Cycle consommateur possible (QoQ)")
    st.markdown(f"**{cycle_conso}** — **Score total CT = {score_conso}**")
    st.success(f"Secteur UP : {SECTEURS[cycle_conso]['UP']}")
    st.error(f"Secteur DOWN : {SECTEURS[cycle_conso]['DOWN']}")

with cc2:
    st.markdown("### Cycle Immobilier possible (QoQ)")
    st.markdown(f"**{cycle_immo}** — **Score total CT = {score_immo}**")
    st.success(f"Secteur UP : {SECTEURS[cycle_immo]['UP']}")
    st.error(f"Secteur DOWN : {SECTEURS[cycle_immo]['DOWN']}")

