#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import yfinance as yf

# ----------- OUTILS DE BASE -----------
def get_last_price(ticker):
    try:
        data = yf.Ticker(ticker).history(period="1d")
        if data.empty: return None
        return round(data['Close'][-1], 2)
    except: return None

def get_price_variation(ticker, period="3mo"):
    try:
        data = yf.Ticker(ticker).history(period=period)
        if data.empty: return None, None
        price_start = data['Close'][0]
        price_end = data['Close'][-1]
        variation = (price_end - price_start) / price_start * 100
        return round(variation, 2), price_end > price_start
    except: return None, None

def mini_analyse(note):
    if note >= 17:
        return "🟢 Excellente entreprise, très solide selon les ratios sectoriels."
    elif note >= 14:
        return "🟡 Entreprise correcte, mais surveille certains critères."
    elif note >= 11:
        return "🟠 Score moyen, attention à la volatilité ou à la dette."
    else:
        return "🔴 Score faible, à éviter pour un investissement long terme."

def color_note(val):
    if isinstance(val, str): return ""
    if val >= 16: return "background-color: #2ecc40; color: white;"
    elif val >= 14: return "background-color: #ffdc00; color: black;"
    elif val >= 11: return "background-color: #ff851b; color: white;"
    else: return "background-color: #ff4136; color: white;"

# =================== BARÈMES SECTORIELS (EXTRAIT, A COMPLETER SI BESOIN) ===================
SECTOR_BENCHMARKS = {
    # --------------- AGRO / COMMODITIES ---------------
    "Agriculture, Forestry, Fishing & Hunting": {
        "roe": [5, 8, 12],
        "roa": [2, 5, 8],
        "de_ratio": [1.2, 0.7, 0.3],
        "ev_ebitda": [14, 11, 8],
        "margin": [3, 6, 10],
        "current": [1.0, 1.3, 1.7],
        "quick": [0.7, 1.1, 1.5],
    },
    "Mining": {
        "roe": [7, 12, 18],
        "roa": [3, 7, 10],
        "de_ratio": [1, 0.6, 0.3],
        "ev_ebitda": [13, 10, 7],
        "margin": [6, 10, 16],
        "current": [1.0, 1.3, 1.7],
        "quick": [0.7, 1.2, 1.5],
    },
    "Utilities": {
        "roe": [5, 8, 12],
        "roa": [3, 5, 8],
        "de_ratio": [1.5, 1.0, 0.6],
        "ev_ebitda": [12, 10, 8],
        "margin": [6, 8, 10],
        "current": [1.0, 1.2, 1.5],
        "quick": [0.7, 1.0, 1.3],
    },
    "Construction": {
        "roe": [5, 8, 12],
        "roa": [2, 5, 8],
        "de_ratio": [1, 0.6, 0.3],
        "ev_ebitda": [14, 11, 8],
        "margin": [4, 8, 14],
        "current": [1.0, 1.3, 1.7],
        "quick": [0.7, 1.2, 1.5],
    },

    # --------------- TRADE / DISTRIBUTION ---------------
    "Wholesale Trade": {
        "roe": [4, 8, 12],
        "roa": [2, 5, 8],
        "de_ratio": [1.1, 0.7, 0.4],
        "ev_ebitda": [12, 10, 7],
        "margin": [3, 6, 10],
        "current": [1.0, 1.3, 1.7],
        "quick": [0.7, 1.1, 1.5],
    },
    "Retail Trade": {
        "roe": [3, 5, 8],
        "roa": [2, 4, 6],
        "de_ratio": [1, 0.7, 0.4],
        "ev_ebitda": [16, 14, 8],
        "margin": [2, 5, 8],
        "current": [1.0, 1.3, 1.7],
        "quick": [0.7, 1.1, 1.5],
    },

    # --------------- TRANSPORT ---------------
    "Transportation & Warehousing": {
        "roe": [5, 8, 12],
        "roa": [2, 5, 8],
        "de_ratio": [2, 1.2, 0.6],
        "ev_ebitda": [10, 8, 6],
        "margin": [2, 5, 8],
        "current": [0.8, 1.1, 1.4],
        "quick": [0.5, 0.8, 1.2],
    },

    # --------------- SERVICES ---------------
    "Information": {
        "roe": [8, 14, 20],
        "roa": [4, 8, 12],
        "de_ratio": [1, 0.5, 0.2],
        "ev_ebitda": [16, 13, 10],
        "margin": [7, 12, 18],
        "current": [1.0, 1.3, 1.7],
        "quick": [0.7, 1.2, 1.5],
    },
    "Finance & Insurance": {
        "roe": [5, 8, 12],
        "roa": [1, 2, 4],
        "de_ratio": [3, 2, 0.8],  # Finance: ratio élevé
        "ev_ebitda": [22, 18, 14],
        "margin": [5, 10, 15],
        "current": [1.0, 1.3, 1.7],
        "quick": [0.7, 1.1, 1.5],
    },
    "Real Estate, Rental & Leasing": {
        "roe": [5, 8, 12],
        "roa": [2, 4, 6],
        "de_ratio": [1.5, 1.0, 0.7],
        "ev_ebitda": [14, 12, 10],
        "margin": [4, 8, 12],
        "current": [0.8, 1.1, 1.4],
        "quick": [0.5, 0.8, 1.2],
    },
    "Professional, Scientific & Technical Services": {
        "roe": [8, 12, 18],
        "roa": [4, 8, 12],
        "de_ratio": [1, 0.5, 0.2],
        "ev_ebitda": [16, 13, 10],
        "margin": [6, 12, 18],
        "current": [1.0, 1.3, 1.7],
        "quick": [0.7, 1.2, 1.5],
    },
    "Management of Companies & Support Services": {
        "roe": [7, 12, 18],
        "roa": [3, 7, 10],
        "de_ratio": [1, 0.6, 0.3],
        "ev_ebitda": [15, 12, 9],
        "margin": [5, 10, 15],
        "current": [1.0, 1.3, 1.7],
        "quick": [0.7, 1.2, 1.5],
    },
    "Educational Services": {
        "roe": [5, 10, 16],
        "roa": [2, 5, 8],
        "de_ratio": [1.2, 0.7, 0.3],
        "ev_ebitda": [14, 11, 8],
        "margin": [4, 9, 14],
        "current": [1.0, 1.3, 1.7],
        "quick": [0.7, 1.1, 1.5],
    },
    "Health Care & Social Assistance": {
        "roe": [5, 10, 20],
        "roa": [3, 6, 10],
        "de_ratio": [1, 0.7, 0.4],
        "ev_ebitda": [18, 14, 10],
        "margin": [5, 10, 20],
        "current": [1.0, 1.4, 2.0],
        "quick": [0.8, 1.2, 1.8],
    },
    "Arts, Entertainment & Recreation": {
        "roe": [2, 7, 14],
        "roa": [1, 4, 8],
        "de_ratio": [1.5, 1.0, 0.7],
        "ev_ebitda": [20, 15, 12],
        "margin": [2, 6, 12],
        "current": [0.8, 1.1, 1.5],
        "quick": [0.5, 0.8, 1.2],
    },
    "Accommodation & Food Services": {
        "roe": [5, 10, 16],
        "roa": [2, 5, 8],
        "de_ratio": [1.2, 0.7, 0.3],
        "ev_ebitda": [14, 11, 8],
        "margin": [4, 9, 14],
        "current": [1.0, 1.3, 1.7],
        "quick": [0.7, 1.1, 1.5],
    },
    "Public Administration": {
        "roe": [2, 5, 8],
        "roa": [1, 3, 5],
        "de_ratio": [1.5, 1.0, 0.7],
        "ev_ebitda": [20, 15, 12],
        "margin": [2, 6, 12],
        "current": [0.8, 1.1, 1.5],
        "quick": [0.5, 0.8, 1.2],
    },

    # --------------- INDUSTRIE / MANUFACTURING ---------------
    "Food, Beverage & Tobacco Products": {
        "roe": [10, 15, 21],
        "roa": [6, 12, 18],
        "de_ratio": [1, 0.7, 0.4],
        "ev_ebitda": [13, 10, 8],
        "margin": [10, 17, 24],
        "current": [1.0, 1.3, 1.7],
        "quick": [0.7, 1.2, 1.5],
    },
    "Textile Mills": {
        "roe": [5, 9, 15],
        "roa": [2, 6, 10],
        "de_ratio": [1.2, 0.7, 0.3],
        "ev_ebitda": [14, 11, 8],
        "margin": [4, 9, 14],
        "current": [1.0, 1.3, 1.7],
        "quick": [0.7, 1.2, 1.5],
    },
    "Apparel, Leather & Allied Products": {
        "roe": [8, 13, 18],
        "roa": [4, 9, 14],
        "de_ratio": [1, 0.7, 0.3],
        "ev_ebitda": [14, 11, 8],
        "margin": [6, 10, 15],
        "current": [1.0, 1.3, 1.7],
        "quick": [0.7, 1.2, 1.5],
    },
    "Wood Products": {
        "roe": [5, 10, 18],
        "roa": [2, 7, 13],
        "de_ratio": [1.2, 0.7, 0.3],
        "ev_ebitda": [13, 10, 7],
        "margin": [5, 10, 16],
        "current": [1.0, 1.3, 1.7],
        "quick": [0.7, 1.2, 1.5],
    },
    "Paper Products": {
        "roe": [6, 12, 18],
        "roa": [2, 7, 13],
        "de_ratio": [1.1, 0.6, 0.3],
        "ev_ebitda": [13, 10, 7],
        "margin": [6, 11, 17],
        "current": [1.0, 1.3, 1.7],
        "quick": [0.7, 1.2, 1.5],
    },
    "Printing & Related Support Activities": {
        "roe": [5, 10, 16],
        "roa": [2, 7, 13],
        "de_ratio": [1.2, 0.7, 0.3],
        "ev_ebitda": [13, 10, 7],
        "margin": [5, 10, 15],
        "current": [1.0, 1.3, 1.7],
        "quick": [0.7, 1.2, 1.5],
    },
    "Petroleum & Coal Products": {
        "roe": [8, 15, 22],
        "roa": [3, 10, 18],
        "de_ratio": [1.1, 0.7, 0.3],
        "ev_ebitda": [12, 9, 6],
        "margin": [6, 12, 18],
        "current": [1.0, 1.2, 1.5],
        "quick": [0.7, 1.0, 1.3],
    },
    "Chemical Products": {
        "roe": [7, 12, 18],
        "roa": [3, 7, 10],
        "de_ratio": [1, 0.5, 0.2],
        "ev_ebitda": [15, 12, 8],
        "margin": [5, 10, 15],
        "current": [1.0, 1.3, 1.7],
        "quick": [0.7, 1.2, 1.5],
    },
    "Plastics & Rubber Products": {
        "roe": [5, 10, 16],
        "roa": [2, 7, 12],
        "de_ratio": [1.2, 0.7, 0.3],
        "ev_ebitda": [14, 11, 8],
        "margin": [5, 10, 15],
        "current": [1.0, 1.3, 1.7],
        "quick": [0.7, 1.2, 1.5],
    },
    "Nonmetallic Mineral Products": {
        "roe": [6, 11, 17],
        "roa": [2, 7, 12],
        "de_ratio": [1, 0.6, 0.2],
        "ev_ebitda": [14, 11, 8],
        "margin": [4, 8, 14],
        "current": [1.0, 1.3, 1.7],
        "quick": [0.7, 1.2, 1.5],
    },
    "Primary Metals": {
        "roe": [7, 12, 18],
        "roa": [3, 8, 13],
        "de_ratio": [1, 0.7, 0.3],
        "ev_ebitda": [13, 10, 7],
        "margin": [5, 10, 15],
        "current": [1.0, 1.3, 1.7],
        "quick": [0.7, 1.2, 1.5],
    },
    "Fabricated Metal Products": {
        "roe": [5, 10, 15],
        "roa": [2, 7, 13],
        "de_ratio": [1.2, 0.7, 0.3],
        "ev_ebitda": [14, 11, 8],
        "margin": [4, 9, 14],
        "current": [1.0, 1.3, 1.7],
        "quick": [0.7, 1.2, 1.5],
    },
    "Machinery": {
        "roe": [7, 12, 18],
        "roa": [3, 7, 10],
        "de_ratio": [1, 0.6, 0.3],
        "ev_ebitda": [14, 11, 8],
        "margin": [6, 10, 15],
        "current": [1.0, 1.3, 1.7],
        "quick": [0.7, 1.2, 1.5],
    },
    "Computer & Electronic Products": {
        "roe": [10, 15, 25],
        "roa": [5, 8, 12],
        "de_ratio": [1, 0.5, 0.25],
        "ev_ebitda": [20, 15, 10],
        "margin": [8, 15, 25],
        "current": [1.2, 1.5, 3],
        "quick": [1, 1.5, 3],
    },
    "Electrical Equipment, Appliances & Components": {
        "roe": [8, 13, 20],
        "roa": [4, 9, 15],
        "de_ratio": [1, 0.7, 0.3],
        "ev_ebitda": [15, 12, 9],
        "margin": [6, 11, 17],
        "current": [1.0, 1.3, 1.7],
        "quick": [0.7, 1.2, 1.5],
    },
    "Transportation Equipment": {
        "roe": [6, 10, 16],
        "roa": [2, 6, 10],
        "de_ratio": [2, 1.2, 0.6],
        "ev_ebitda": [10, 8, 6],
        "margin": [3, 7, 12],
        "current": [1.0, 1.2, 1.6],
        "quick": [0.7, 1.0, 1.4],
    },
    "Furniture & Related Products": {
        "roe": [5, 10, 18],
        "roa": [2, 7, 13],
        "de_ratio": [1.2, 0.7, 0.3],
        "ev_ebitda": [13, 10, 7],
        "margin": [5, 10, 16],
        "current": [1.0, 1.3, 1.7],
        "quick": [0.7, 1.2, 1.5],
    },
    "Miscellaneous Manufacturing": {
        "roe": [5, 10, 18],
        "roa": [2, 7, 13],
        "de_ratio": [1.2, 0.7, 0.3],
        "ev_ebitda": [13, 10, 7],
        "margin": [5, 10, 16],
        "current": [1.0, 1.3, 1.7],
        "quick": [0.7, 1.2, 1.5],
    }
}
# Aliases
SECTOR_BENCHMARKS.update({
    # High-Tech & Industrie
    "Technology": SECTOR_BENCHMARKS["Computer & Electronic Products"],
    "Information Technology": SECTOR_BENCHMARKS["Computer & Electronic Products"],
    "Consumer Electronics": SECTOR_BENCHMARKS["Computer & Electronic Products"],
    "Industrials": SECTOR_BENCHMARKS["Machinery"],
    "Manufacturing": SECTOR_BENCHMARKS["Machinery"],
    "Industrial Goods": SECTOR_BENCHMARKS["Machinery"],

    # Consommation
    "Consumer Goods": SECTOR_BENCHMARKS["Food, Beverage & Tobacco Products"],
    "Consumer Defensive": SECTOR_BENCHMARKS["Food, Beverage & Tobacco Products"],
    "Consumer Staples": SECTOR_BENCHMARKS["Food, Beverage & Tobacco Products"],
    "Consumer Cyclical": SECTOR_BENCHMARKS["Apparel, Leather & Allied Products"],

    # Santé
    "Healthcare": SECTOR_BENCHMARKS["Health Care & Social Assistance"],
    "Health Care": SECTOR_BENCHMARKS["Health Care & Social Assistance"],

    # Énergie
    "Energy": SECTOR_BENCHMARKS["Petroleum & Coal Products"],
    "Oil, Gas & Consumable Fuels": SECTOR_BENCHMARKS["Petroleum & Coal Products"],

    # Finance
    "Financial Services": SECTOR_BENCHMARKS["Finance & Insurance"],
    "Finance": SECTOR_BENCHMARKS["Finance & Insurance"],

    # Matériaux
    "Materials": SECTOR_BENCHMARKS["Chemical Products"],

    # Immobilier
    "Real Estate": SECTOR_BENCHMARKS["Real Estate, Rental & Leasing"],

    # Services
    "Services": SECTOR_BENCHMARKS["Professional, Scientific & Technical Services"],

    # Utilities
    "Utilities": SECTOR_BENCHMARKS["Utilities"],

    # Agro/commodities
    "Agriculture": SECTOR_BENCHMARKS["Agriculture, Forestry, Fishing & Hunting"],

    # Transport
    "Transportation": SECTOR_BENCHMARKS["Transportation & Warehousing"],

    # Distribution
    "Wholesale": SECTOR_BENCHMARKS["Wholesale Trade"],
    "Retail": SECTOR_BENCHMARKS["Retail Trade"],

    # Education & Public
    "Education": SECTOR_BENCHMARKS["Educational Services"],
    "Public Administration": SECTOR_BENCHMARKS["Public Administration"],
})
DEFAULT_SECTOR = "Technology"
POINTS = {0: 0.6, 1: 1.3, 2: 1.9, 3: 2.5}
CATEGORIES = {0: "Chocolate", 1: "Bronze", 2: "Silver", 3: "Gold"}

# ----------- SCORING -----------
def score_ratio(value, thresholds, reverse=False):
    if value is None: return 0
    if reverse:
        if value > thresholds[0]: return 0
        elif value > thresholds[1]: return 1
        elif value > thresholds[2]: return 2
        else: return 3
    else:
        if value < thresholds[0]: return 0
        elif value < thresholds[1]: return 1
        elif value < thresholds[2]: return 2
        else: return 3

def score_analyst(x):
    if x > 3.5: return 0
    elif x > 2.5: return 1
    elif x > 1.5: return 2
    else: return 3

# ----------- ANALYSE PAR TICKER -----------
def analyze_ticker(ticker, sector_override=None, manual_inputs=None):
    info = yf.Ticker(ticker).info
    sector_detected = info.get("sector") or DEFAULT_SECTOR
    sector = sector_override if sector_override else sector_detected
    if sector not in SECTOR_BENCHMARKS:
        sector = DEFAULT_SECTOR
    benchmarks = SECTOR_BENCHMARKS.get(sector, SECTOR_BENCHMARKS[DEFAULT_SECTOR])

    ratios_list = [
        ("EBITDA", "ebitdaMargins", 0, 100, "ex: 23.0"),
        ("Marge nette", "profitMargins", 0, 100, "ex: 15.3"),
        ("D/E ratio", "debtToEquity", 0, 0.01, "ex: 1.3"),
        ("Current ratio", "currentRatio", 0, 1, "ex: 1.2"),
        ("Quick ratio", "quickRatio", 0, 1, "ex: 0.9"),
        ("ROA", "returnOnAssets", 0, 100, "ex: 8.2"),
        ("ROE", "returnOnEquity", 0, 100, "ex: 21.5"),
        ("Analystes", "recommendationMean", 3, 1, "ex: 2.1 (1=achat fort, 5=vente)")
    ]
    d = {}
    for i, (lab, k, defval, mult, exemple) in enumerate(ratios_list):
        if manual_inputs and lab in manual_inputs:
            v = manual_inputs[lab]
        else:
            v = info.get(k, None)
            if (v is None or v == 0 or v == '' or v is False):
                v = defval
        d[lab] = round(v*mult, 3) if v is not None else None

    scores = [
        score_ratio(d["EBITDA"], benchmarks["ev_ebitda"], reverse=True),
        score_ratio(d["Marge nette"], benchmarks["margin"]),
        score_ratio(d["D/E ratio"], benchmarks["de_ratio"], reverse=True),
        score_ratio(d["Current ratio"], benchmarks["current"]),
        score_ratio(d["Quick ratio"], benchmarks["quick"]),
        score_ratio(d["ROA"], benchmarks["roa"]),
        score_ratio(d["ROE"], benchmarks["roe"]),
        score_analyst(d["Analystes"])
    ]

    labels = [x[0] for x in ratios_list]
    points = [POINTS[s] for s in scores]
    cats = [CATEGORIES[s] for s in scores]
    note = sum(points)
    df_detail = pd.DataFrame({
        "Critère": labels,
        "Valeur": [d[k] for k in labels],
        "Score": cats,
        "Points": points,
    })
    missing = []
    for i, (lab, k, defval, mult, exemple) in enumerate(ratios_list):
        if manual_inputs and lab in manual_inputs:
            continue
        v = info.get(k, None)
        if (v is None or v == 0 or v == '' or v is False):
            missing.append((lab, defval, exemple))
    return df_detail, note, sector, missing

# ----------- APP STREAMLIT -----------
st.title("📊 Classement de tickers US (scoring auto sectorisé)")
st.caption("Ajoutez vos tickers, comparez-les et classez-les automatiquement selon leur note sectorisée.")

if "tickers" not in st.session_state:
    st.session_state["tickers"] = []
if "selected_ticker" not in st.session_state:
    st.session_state["selected_ticker"] = None
if "manual_inputs" not in st.session_state:
    st.session_state["manual_inputs"] = {}
if "sector_override" not in st.session_state:
    st.session_state["sector_override"] = {}

with st.form(key="add_ticker_form"):
    ticker = st.text_input("Entrez un ticker US (ex: AAPL)").upper()
    col1, col2 = st.columns([1,1])
    with col1:
        add_btn = st.form_submit_button("Ajouter")
    with col2:
        clear_btn = st.form_submit_button("Clear")

if add_btn and ticker and ticker not in st.session_state["tickers"]:
    st.session_state["tickers"].append(ticker)
    st.session_state["selected_ticker"] = ticker
    st.session_state["manual_inputs"][ticker] = {}
    st.session_state["sector_override"][ticker] = None
if clear_btn:
    st.session_state["tickers"] = []
    st.session_state["manual_inputs"] = {}
    st.session_state["selected_ticker"] = None
    st.session_state["sector_override"] = {}

# ----------- Affichage tableau classement -----------
notes_data = []
for t in st.session_state["tickers"]:
    df_detail, note, secteur, _ = analyze_ticker(
        t, 
        st.session_state["sector_override"].get(t), 
        st.session_state["manual_inputs"].get(t)
    )
    notes_data.append({
        "Entreprise": t, 
        "Secteur": secteur,
        "Note sur 20": round(note, 2)
    })
df_notes = pd.DataFrame(notes_data)
if "Note sur 20" in df_notes.columns:
    df_notes["Note sur 20"] = pd.to_numeric(df_notes["Note sur 20"], errors='coerce')
    df_notes = df_notes.sort_values(by="Note sur 20", ascending=False)

st.markdown("### Classement des entreprises")
if not df_notes.empty and df_notes["Note sur 20"].apply(lambda x: isinstance(x, (float, int))).any():
    styled_df = df_notes.style.applymap(color_note, subset=["Note sur 20"])
    st.dataframe(styled_df, use_container_width=True, height=280)
else:
    st.dataframe(df_notes, use_container_width=True)
st.caption("Clique sur une entreprise ci-dessous pour afficher ses détails.")

# ----------- SÉLECTION DE L'ENTREPRISE POUR VOIR LE DÉTAIL -----------
for i, row in df_notes.iterrows():
    if st.button(f"Voir détails {row['Entreprise']}", key=f"show_{row['Entreprise']}"):
        st.session_state["selected_ticker"] = row["Entreprise"]

# ----------- AFFICHAGE DES DÉTAILS POUR LE TICKER SÉLECTIONNÉ -----------
if st.session_state["selected_ticker"]:
    ticker = st.session_state["selected_ticker"]
    manual_inputs = st.session_state["manual_inputs"].get(ticker, {})
    sector_override = st.session_state["sector_override"].get(ticker, None)
    df_detail, note, secteur, missing = analyze_ticker(ticker, sector_override, manual_inputs)
    info = yf.Ticker(ticker).info
    nom_complet = info.get('shortName', ticker)
    prix_actuel = get_last_price(ticker)
    variation_3mois, hausse = get_price_variation(ticker, "3mo")
    analyse = mini_analyse(note)

    st.markdown(f"#### {nom_complet} ({ticker})")
    # --- SÉLECTEUR SECTEUR S'IL MANQUE ---
    if info.get("sector", None) not in SECTOR_BENCHMARKS:
        secteur_selected = st.selectbox(
            f"Secteur non reconnu pour {ticker}. Choisis un secteur :",
            options=list(SECTOR_BENCHMARKS.keys()),
            key=f"sector_select_{ticker}",
            index=0 if sector_override is None else list(SECTOR_BENCHMARKS.keys()).index(sector_override) if sector_override in SECTOR_BENCHMARKS else 0
        )
        st.session_state["sector_override"][ticker] = secteur_selected
        secteur = secteur_selected

    st.write(f"**Secteur utilisé :** {secteur}")
    st.write(f"**Prix actuel :** {prix_actuel if prix_actuel is not None else 'Non dispo'} $")
    if variation_3mois is not None:
        fleche = "🔺" if hausse else "🔻"
        st.write(f"**Variation 3 mois :** {fleche} {abs(variation_3mois)} %")
    else:
        st.write("**Variation 3 mois :** Non dispo")
    st.write(f"**Mini-analyse :** {analyse}")

    # --- INPUTS MANQUANTS SI BESOIN ---
    if len(missing) > 0:
        with st.form(key=f"manquants_{ticker}"):
            inputs = {}
            for (lab, defval, exemple) in missing:
                value = st.number_input(
                    f"Valeur manquante ou douteuse pour {lab} ({ticker}), saisis-la ({exemple}) :",
                    key=f"{ticker}_{lab}_input",
                    value=float(defval)
                )
                inputs[lab] = value
            valider = st.form_submit_button("Valider ces valeurs")
            if valider:
                st.session_state["manual_inputs"][ticker] = {**manual_inputs, **inputs}
                st.experimental_rerun()
    st.dataframe(df_detail, use_container_width=True)
    st.markdown(f"[Voir sur Yahoo Finance](https://finance.yahoo.com/quote/{ticker})")

