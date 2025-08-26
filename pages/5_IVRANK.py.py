#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# streamlit run zz.py
# -*- coding: utf-8 -*-
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ========= PARAMÈTRES PAR DÉFAUT =========
DEFAULT_TICKER   = "AAPL"
DEFAULT_PERIOD   = "6mo"   # "1mo","3mo","6mo","1y","2y","5y","max"
DEFAULT_INTERVAL = "1d"    # "1d","1h","30m","15m","5m"

IV_WINDOW   = 30
Z_HIGH      = 1.5
Z_LOW       = -1.5
MA_FAST     = 20
MA_SLOW     = 50
NOISE_IV_SIGMA = 0.10

# --- util: force 1D Series propre (ta fonction inchangée) ---
def to_series_1d(obj, index=None, name=None):
    """
    Convertit obj (Series/DataFrame/ndarray/liste) en Series 1D.
    Si DataFrame à 1 colonne -> prend la 1ère. Si ndarray 2D -> ravel().
    Réindexe si un index est fourni.
    """
    if isinstance(obj, pd.Series):
        s = obj.copy()
    elif isinstance(obj, pd.DataFrame):
        s = obj.squeeze("columns")
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s = s.copy()
    else:
        arr = np.asarray(obj)
        s = pd.Series(arr.ravel())
    if index is not None:
        s.index = index
    if name is not None:
        s.name = name
    return s

# ========= UI =========
st.set_page_config(page_title="Signal Options du Jour", layout="wide")
st.title("📈 Signal Options du Jour")

c1, c2, c3 = st.columns([2,1,1])
with c1:
    TICKER = st.text_input("Ticker", value=DEFAULT_TICKER).upper().strip()
with c2:
    PERIOD = st.selectbox("Période", ["1mo","3mo","6mo","1y","2y","5y","max"],
                          index=["1mo","3mo","6mo","1y","2y","5y","max"].index(DEFAULT_PERIOD))
with c3:
    INTERVAL = st.selectbox("Intervalle", ["1d","1h","30m","15m","5m"], index=0)

if not TICKER:
    st.stop()

# ========= DONNÉES PRIX =========
with st.spinner("Téléchargement des données…"):
    data = yf.download(TICKER, period=PERIOD, interval=INTERVAL, auto_adjust=True)
if data.empty:
    st.error("Aucune donnée téléchargée. Vérifie le ticker/période/intervalle.")
    st.stop()

# Close -> Series 1D
close_raw = data["Close"]
close = to_series_1d(close_raw, index=data.index, name="Close").astype(float)
returns = close.pct_change()

# ========= VOL RÉALISÉE & PROXY IV =========
realized_vol = returns.rolling(window=IV_WINDOW).std() * np.sqrt(252)
realized_vol = realized_vol.dropna()

np.random.seed(0)
noise = pd.Series(np.random.normal(0, NOISE_IV_SIGMA, size=len(realized_vol)),
                  index=realized_vol.index)

iv_proxy = (realized_vol * (1 + noise)).clip(lower=1e-8)
iv_mean  = iv_proxy.rolling(window=IV_WINDOW).mean()
iv_std   = iv_proxy.rolling(window=IV_WINDOW).std()
iv_z     = (iv_proxy - iv_mean) / iv_std

# ========= IV RANK (ajout) =========
# IV Rank = (IV actuelle - IV min) / (IV max - IV min) × 100
iv_min = iv_proxy.min()
iv_max = iv_proxy.max()
denom = (iv_max - iv_min)
iv_rank = ((iv_proxy - iv_min) / denom * 100) if denom != 0 else pd.Series(np.nan, index=iv_proxy.index)
iv_rank.name = "IV_RANK"

# ========= TENDANCE PRIX =========
ma_fast = close.rolling(MA_FAST).mean()
ma_slow = close.rolling(MA_SLOW).mean()

# ========= ALIGNEMENT & TABLE =========
close    = to_series_1d(close,    index=close.index,          name="Close")
ma_fast  = to_series_1d(ma_fast,  index=close.index,          name="MA_FAST")
ma_slow  = to_series_1d(ma_slow,  index=close.index,          name="MA_SLOW")
iv_proxy = to_series_1d(iv_proxy, index=realized_vol.index,   name="IV")
iv_z     = to_series_1d(iv_z,     index=realized_vol.index,   name="IV_Z")
iv_rank  = to_series_1d(iv_rank,  index=realized_vol.index,   name="IV_RANK")

df = pd.concat([close, ma_fast, ma_slow, iv_proxy, iv_z, iv_rank], axis=1, join="inner").dropna()

def trend_label(fast, slow, tol=0.005):
    if np.isnan(fast) or np.isnan(slow):
        return np.nan
    if fast > slow * (1 + tol):
        return "Haussier"
    elif fast < slow * (1 - tol):
        return "Baissier"
    else:
        return "Neutre/Range"

df["Trend"] = [trend_label(f, s) for f, s in zip(df["MA_FAST"], df["MA_SLOW"])]

def decide(z, trend, z_low=Z_LOW, z_high=Z_HIGH):
    if z <= z_low:
        if trend == "Haussier":
            return "ACHAT CALL (débit)", "Bull call spread"
        elif trend == "Baissier":
            return "ACHAT PUT (débit)", "Bear put spread"
        else:
            return "ACHAT STRADDLE/STRANGLE", "Calendar spread (long vega)"
    if z >= z_high:
        if trend == "Haussier":
            return "VENTE PUT cash-secured", "Bull put CREDIT spread"
        elif trend == "Baissier":
            return "VENTE CALL couvert", "Bear call CREDIT spread"
        else:
            return "IRON CONDOR / SHORT STRANGLE", "Butterfly crédit"
    if trend == "Haussier":
        return "CALL DEBIT spread (si signal fort)", "Achat CALL si catalyseur"
    elif trend == "Baissier":
        return "PUT DEBIT spread (si signal fort)", "Achat PUT si catalyseur"
    else:
        return "Pas de bord clair (range)", "Vente gamma intraday (avancé)"

df[["Action_Principale", "Alternatives"]] = [decide(z, t) for z, t in zip(df["IV_Z"], df["Trend"])]

# ========= TABLEAU & DERNIER SIGNAL =========
out_cols = ["Close","MA_FAST","MA_SLOW","Trend","IV","IV_Z","IV_RANK","Action_Principale","Alternatives"]
table = df[out_cols].round({"Close":2,"MA_FAST":2,"MA_SLOW":2,"IV":3,"IV_Z":2,"IV_RANK":1})
last = table.iloc[-1]

# ========= ONGLETs =========
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Tableau & Dernier signal",
    "📈 IV (proxy) + moyenne",
    "📉 IVMR (z-score)",
    "💹 Prix + MAs + résumé",
    "🟣 IVMR + signaux"
])

with tab1:
    st.subheader("Tableau de décision (10 dernières lignes)")
    st.dataframe(table.tail(10), use_container_width=True)

    st.subheader("Dernier signal")
    last_signal = {
        "Date": df.index[-1].strftime("%Y-%m-%d"),
        "Trend": last["Trend"],
        "IV": round(float(last["IV"]), 3),
        "IV_Z": round(float(last["IV_Z"]), 2),
        "IV_RANK (%)": round(float(last["IV_RANK"]), 1),
        "Action_Principale": last["Action_Principale"],
        "Alternatives": last["Alternatives"],
    }
    st.table(pd.DataFrame([last_signal]))

with tab2:
    st.subheader("Volatilité implicite (proxy) et moyenne mobile")
    iv_mean_plot = df["IV"].rolling(IV_WINDOW).mean()
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df.index, df["IV"], label="IV (proxy)")
    ax.plot(df.index, iv_mean_plot, linestyle="--", label=f"Moyenne IV {IV_WINDOW}j")
    ax.set_title("Volatilité implicite (proxy) & moyenne")
    ax.legend(); ax.grid(True); fig.tight_layout()
    st.pyplot(fig)

with tab3:
    st.subheader("IVMR (z-score) — zones d'achat/vente de volatilité")
    fig2, ax2 = plt.subplots(figsize=(10,3.8))
    ax2.plot(df.index, df["IV_Z"], label="IV z-score")
    ax2.axhline(Z_HIGH, linestyle=":", label=f"Seuil haut {Z_HIGH}")
    ax2.axhline(Z_LOW,  linestyle=":", label=f"Seuil bas  {Z_LOW}")
    ax2.set_title(f"IVMR (z-score) — IV Rank actuel : {last['IV_RANK']:.1f}%")
    ax2.legend(); ax2.grid(True); fig2.tight_layout()
    st.pyplot(fig2)

with tab4:
    st.subheader("Prix + MA20/MA50 + résumé du dernier signal")
    last_idx = df.index[-1]
    last_row = df.iloc[-1]
    fig3, ax3 = plt.subplots(figsize=(10,4))
    ax3.plot(df.index, df["Close"],   label="Cours")
    ax3.plot(df.index, df["MA_FAST"], label=f"MA {MA_FAST}")
    ax3.plot(df.index, df["MA_SLOW"], label=f"MA {MA_SLOW}")
    ax3.axvline(last_idx, color="red", linestyle="--")
    ax3.set_title(
        f"Dernier signal : {last_row['Action_Principale']}  |  Alt: {last_row['Alternatives']}  "
        f"| Trend: {last_row['Trend']}  | IV_Z: {last_row['IV_Z']:.2f}  | IV_RANK: {last_row['IV_RANK']:.1f}%"
    )
    ax3.legend(); ax3.grid(True); fig3.tight_layout()
    st.pyplot(fig3)

with tab5:
    st.subheader("IVMR (z-score) + signaux (achat/vente vol & directionnels)")
    # ====== CLASSIFICATION DES SIGNAUX ======
    df["OK_ACHAT_VOL"]  = df["IV_Z"] <= Z_LOW
    df["OK_VENTE_VOL"]  = df["IV_Z"] >= Z_HIGH
    df["OK_ACHAT_CALL"] = (df["IV_Z"] <= Z_LOW)  & (df["Trend"] == "Haussier")
    df["OK_ACHAT_PUT"]  = (df["IV_Z"] <= Z_LOW)  & (df["Trend"] == "Baissier")
    df["OK_VENTE_CALL"] = (df["IV_Z"] >= Z_HIGH) & (df["Trend"] == "Baissier")
    df["OK_VENTE_PUT"]  = (df["IV_Z"] >= Z_HIGH) & (df["Trend"] == "Haussier")

    fig4, ax4 = plt.subplots(figsize=(10,4))
    ax4.plot(df.index, df["IV_Z"], label="IV z-score")
    ax4.axhline(Z_HIGH, linestyle=":", label=f"Seuil haut {Z_HIGH}")
    ax4.axhline(Z_LOW,  linestyle=":", label=f"Seuil bas  {Z_LOW}")

    # Points de signaux (scatter). Un marqueur par type.
    def sc(mask, marker, lab):
        x = df.index[mask]
        y = df.loc[mask, "IV_Z"]
        ax4.scatter(x, y, marker=marker, s=60, label=lab)

    sc(df["OK_ACHAT_CALL"], "^",  "OK ACHAT CALL")
    sc(df["OK_ACHAT_PUT"],  "v",  "OK ACHAT PUT")
    sc(df["OK_VENTE_CALL"], "x",  "OK VENTE CALL")
    sc(df["OK_VENTE_PUT"],  "s",  "OK VENTE PUT")
    sc(df["OK_ACHAT_VOL"],  "o",  "OK ACHAT VOL")
    sc(df["OK_VENTE_VOL"],  "D",  "OK VENTE VOL")

    ax4.set_title("IVMR (z-score) + signaux (achat/vente vol & directionnels)")
    ax4.legend(ncol=3); ax4.grid(True); fig4.tight_layout()
    st.pyplot(fig4)

st.caption("Note : IV = proxy (vol réalisée + bruit). IV Rank calculé sur la période choisie.")

