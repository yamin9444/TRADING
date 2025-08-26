#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import io
import re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Screener ISM — Achat/Vente", layout="wide")

# --------------- Helpers ---------------

def normalize_name(s: str) -> str:
    """Col name -> snake_case simplifié, sans accents, espaces -> '_'"""
    s = str(s)
    repl = {'é':'e','è':'e','ê':'e','ë':'e','à':'a','â':'a','ä':'a','î':'i','ï':'i',
            'ô':'o','ö':'o','ù':'u','û':'u','ü':'u','ç':'c'}
    s = s.strip().lower()
    for k,v in repl.items(): s = s.replace(k,v)
    s = s.replace('\n',' ')
    s = re.sub(r'\s+',' ', s).replace(' ', '_')
    s = re.sub(r'[^a-z0-9_()]','', s)
    return s

def to_num_series(series):
    """Convertit texte -> float : retire %, espaces, NBSP, , → ., gère #DIV/0! / N/A."""
    cleaned = (pd.Series(series)
               .astype(str)
               .str.replace('%','', regex=False)
               .str.replace('\u00a0','', regex=False)
               .str.replace(' ','', regex=False)
               .str.replace(',','.', regex=False)
               .replace({'nan': None, 'None': None, '': None,
                         '#DIV/0!': None, 'NaN': None, 'N/A': None}))
    return pd.to_numeric(cleaned, errors='coerce')

def best_colmap(cols):
    """Mappe colonnes réelles -> cibles normalisées"""
    norm = {c: normalize_name(c) for c in cols}
    inv = {v:k for k,v in norm.items()}  # norm -> original
    wanted = {
        'market_cap_mil': ['market_cap_(mil)', 'market_cap_mil', 'market_cap', 'marketcap_mil', 'mkt_cap_(mil)'],
        'sector': ['sector'],
        'industry': ['industry'],
        'eg1': ['eg1'], 'eg2': ['eg2'],
        'pe1': ['pe1'], 'pe2': ['pe2'],
        'peg1': ['peg1'], 'peg2': ['peg2'],
    }
    colmap = {}
    for tgt, variants in wanted.items():
        found = None
        for v in variants:
            if v in inv:
                found = inv[v]; break
        if not found and tgt=='market_cap_mil':
            for n, orig in inv.items():
                if 'market' in n and 'cap' in n and '(mil)' in n:
                    found = orig; break
        if found: colmap[tgt] = found
    return colmap

@st.cache_data(show_spinner=False)
def try_read_file(file_bytes: bytes, filename: str):
    name = (filename or '').lower()
    if name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(io.BytesIO(file_bytes))
        return df, 'excel', None
    # CSV: brute-force encodage/séparateur
    for enc in ('utf-8', 'utf-8-sig', 'latin-1'):
        for sep in (',',';','\t','|'):
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc, sep=sep)
                if df.shape[1] >= 2:
                    return df, enc, sep
            except Exception:
                pass
    raise ValueError("Impossible de lire le fichier (CSV/Excel).")

def prepare_df(df: pd.DataFrame):
    """Renomme colonnes -> cibles et convertit en numérique."""
    colmap = best_colmap(df.columns)
    required = {'market_cap_mil','eg1','eg2','pe1','pe2','peg1','peg2'}
    if not required <= set(colmap.keys()):
        missing = required - set(colmap.keys())
        raise ValueError(f"Colonnes manquantes: {missing}\nColonnes trouvées: {list(df.columns)}")
    work = df.rename(columns={orig: tgt for tgt, orig in colmap.items()}).copy()
    # conversions
    for c in ['market_cap_mil','eg1','eg2','pe1','pe2','peg1','peg2']:
        work[c] = to_num_series(work[c])
    # sector/industry si présents
    work['sector']   = df[colmap['sector']].astype(str)   if 'sector'   in colmap else '(inconnu)'
    work['industry'] = df[colmap['industry']].astype(str) if 'industry' in colmap else '(inconnu)'
    return work

def filtre_achat(x, mc_min, mc_max):
    return (
        (x['market_cap_mil'].between(mc_min, mc_max, inclusive='both')) &
        (x['eg1'] < x['eg2']) &
        (x['pe1'] > x['pe2']) &
        (x['peg1'] > x['peg2'])
    )

def filtre_vente(x, mc_min):
    return (
        (x['market_cap_mil'] >= mc_min) &
        (x['eg1'] > x['eg2']) &
        (x['pe1'] < x['pe2']) &
        (x['peg1'] < x['peg2'])
    )

def df_to_csv_bytes(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode('utf-8')

def df_to_xlsx_bytes(df):
    import xlsxwriter
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine='xlsxwriter') as w:
        df.to_excel(w, index=False, sheet_name='resultats')
    bio.seek(0)
    return bio.getvalue()

# --------------- UI ---------------

st.title("📈 Screener ISM — Achat / Vente (Secteur + Industry)")

uploaded = st.file_uploader(
    "Charge ton fichier (CSV / Excel)", type=["csv", "xlsx", "xls"], accept_multiple_files=False
)

left, mid, right = st.columns(3)
with left:
    mode = st.selectbox("Mode", ["achat", "vente"])
with mid:
    mc_buy_min = st.number_input("MC min (achat, en millions)", value=3000.0, step=100.0)
with right:
    mc_buy_max = st.number_input("MC max (achat, en millions)", value=10000.0, step=100.0)

mc_sell_min = st.number_input("MC min (vente, en millions)", value=20000.0, step=500.0)

if uploaded:
    try:
        raw_df, enc_used, sep_used = try_read_file(uploaded.getvalue(), uploaded.name)
        st.success(f"Fichier chargé : {raw_df.shape[0]} lignes, {raw_df.shape[1]} colonnes "
                   f"({ 'excel' if enc_used=='excel' else f'enc={enc_used}, sep={repr(sep_used)}' })")
    except Exception as e:
        st.error(f"Erreur de lecture : {e}")
        st.stop()

    # Choix secteur/industry basés sur les colonnes d’origine (plus large)
    cmap = best_colmap(raw_df.columns)
    if 'sector' in cmap:
        sectors = ['Tous'] + sorted(raw_df[cmap['sector']].dropna().astype(str).unique())
    else:
        sectors = ['Tous']
    sector = st.selectbox("Secteur", sectors, index=0)

    # Industry dépend du secteur choisi
    if 'industry' in cmap:
        if sector != 'Tous' and 'sector' in cmap:
            mask = raw_df[cmap['sector']].astype(str).str.lower() == sector.lower()
            industries = ['Tous'] + sorted(raw_df.loc[mask, cmap['industry']].dropna().astype(str).unique())
        else:
            industries = ['Tous'] + sorted(raw_df[cmap['industry']].dropna().astype(str).unique())
    else:
        industries = ['Tous']
    industry = st.selectbox("Industry (optionnel)", industries, index=0)

    # Préparation / nettoyage
    try:
        work = prepare_df(raw_df)
    except Exception as e:
        st.error(f"Erreur préparation colonnes : {e}")
        st.stop()

    # Filtres secteur + industry
    if sector != 'Tous':
        work = work[work['sector'].astype(str).str.lower() == sector.lower()]
    if industry != 'Tous':
        work = work[work['industry'].astype(str).str.lower() == industry.lower()]

    # Application règles
    if mode == "achat":
        res = work[filtre_achat(work, mc_buy_min, mc_buy_max)].copy()
    else:
        res = work[filtre_vente(work, mc_sell_min)].copy()

    if 'market_cap_mil' in res.columns:
        res = res.sort_values('market_cap_mil', ascending=False)

    st.subheader(f"Résultats : {len(res)} lignes")
    st.dataframe(res.head(200), use_container_width=True)

    # Downloads
    colA, colB = st.columns(2)
    with colA:
        st.download_button(
            "⬇️ Télécharger CSV",
            data=df_to_csv_bytes(res),
            file_name=f"screen_{mode}_{sector}_{industry}.csv",
            mime="text/csv"
        )
    with colB:
        try:
            st.download_button(
                "⬇️ Télécharger Excel",
                data=df_to_xlsx_bytes(res),
                file_name=f"screen_{mode}_{sector}_{industry}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.info("Install `xlsxwriter` si le bouton Excel ne fonctionne pas : `pip install xlsxwriter`")

else:
    st.info("Charge un fichier pour commencer. Astuce : CSV/Excel exporté depuis ton outil habituel.")

