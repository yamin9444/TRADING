#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
import streamlit as st
from scipy.stats import norm
from scipy.optimize import brentq
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import date

st.set_page_config(page_title="Black-Scholes + ATR + Target 1M", layout="wide")

# ======================================================================
# Utils sessions
# ======================================================================
if "iv_monthly" not in st.session_state:
    st.session_state.iv_monthly = None
if "atr_monthly" not in st.session_state:
    st.session_state.atr_monthly = None
if "spot" not in st.session_state:
    st.session_state.spot = None

# ======================================================================
# Black-Scholes core
# ======================================================================
def _d1d2(S, K, r, T, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return d1, d2

def _d1d2_q(S, K, r, q, T, sigma):
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return d1, d2

def blackScholes(S, K, r, T, sigma, type_="c"):
    d1, d2 = _d1d2(S, K, r, T, sigma)
    if type_ == "c":
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def blackScholes_q(S, K, r, q, T, sigma, type_="c"):
    d1, d2 = _d1d2_q(S, K, r, q, T, sigma)
    if type_ == "c":
        return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

def implied_volatility(market_price, S, K, r, T, type_='c'):
    f = lambda sig: blackScholes(S, K, r, T, sig, type_) - market_price
    try:
        return brentq(f, 1e-5, 5)
    except ValueError:
        return np.nan

def implied_volatility_q(market_price, S, K, r, q, T, type_='c'):
    f = lambda sig: blackScholes_q(S, K, r, q, T, sig, type_) - market_price
    try:
        return brentq(f, 1e-5, 5)
    except ValueError:
        return np.nan

def greeks_plain(S, K, r, T, sigma, type_="c"):
    d1, d2 = _d1d2(S, K, r, T, sigma)
    if type_ == "c":
        delta = norm.cdf(d1)
        theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2))/365
        rho   =  K*T*np.exp(-r*T)*norm.cdf(d2)/100
    else:
        delta = norm.cdf(d1) - 1
        theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2))/365
        rho   = -K*T*np.exp(-r*T)*norm.cdf(-d2)/100
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    vega  = S*norm.pdf(d1)*np.sqrt(T)/100
    return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}

def greeks_with_q(S, K, r, q, T, sigma, type_="c"):
    d1, d2 = _d1d2_q(S, K, r, q, T, sigma)
    exp_qT = np.exp(-q*T)
    if type_ == "c":
        delta = exp_qT*norm.cdf(d1)
        theta = (-S*exp_qT*norm.pdf(d1)*sigma/(2*np.sqrt(T))
                 + q*S*exp_qT*norm.cdf(d1)
                 - r*K*np.exp(-r*T)*norm.cdf(d2))/365
        rho   =  K*T*np.exp(-r*T)*norm.cdf(d2)/100
    else:
        delta = exp_qT*(norm.cdf(d1) - 1)
        theta = (-S*exp_qT*norm.pdf(d1)*sigma/(2*np.sqrt(T))
                 - q*S*exp_qT*norm.cdf(-d1)
                 + r*K*np.exp(-r*T)*norm.cdf(-d2))/365
        rho   = -K*T*np.exp(-r*T)*norm.cdf(-d2)/100
    gamma = exp_qT*norm.pdf(d1)/(S*sigma*np.sqrt(T))
    vega  = S*exp_qT*norm.pdf(d1)*np.sqrt(T)/100
    return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}

# ======================================================================
# Header
# ======================================================================
st.title("Application Black-Scholes (avec dividendes) + ATR + Target 1M")

# ======================================================================
# Sidebar : spot auto
# ======================================================================
st.sidebar.header("Spot (Yahoo Finance)")
ticker = st.sidebar.text_input("Ticker (ex: AAPL)", value="AAPL")
if st.sidebar.button("Récupérer le spot", key="spotbtn"):
    try:
        t = yf.Ticker(ticker)
        s = t.history(period="1d")["Close"].iloc[-1]
        st.session_state.spot = float(s)
        st.sidebar.success(f"{ticker.upper()} spot: {s:.2f} $")
    except Exception as e:
        st.sidebar.error(f"Erreur: {e}")

# ======================================================================
# Section Black-Scholes
# ======================================================================
st.subheader("Pricing Black-Scholes")
col1, col2 = st.columns([2,1])

with col1:
    S = st.number_input("Spot (S)", value=st.session_state.spot if st.session_state.spot else 100.0, format="%.6f")
    K = st.number_input("Strike (K)", value=100.0, format="%.6f")
    r = st.number_input("Taux sans risque r (ex: 4,49% ⇒ 0.0449)", value=0.03, format="%.6f")
    T_days = st.number_input("Maturité (jours)", value=30, step=1)
    sigma_pct = st.number_input("Volatilité (σ, % annualisée)", value=25.0, min_value=0.0, step=0.1)
    type_opt = st.selectbox("Type d’option", ["Call", "Put"])
    type_clean = "c" if type_opt == "Call" else "p"

    use_div = st.checkbox("Inclure les dividendes (q)", value=False)
    q_pct = st.number_input("Rendement dividendes (%)", value=0.0, min_value=0.0, step=0.1,
                            help="Annualisé. Exemple 3% ⇒ 3.0")
    market_price = st.number_input("Prix observé (pour IV)", value=0.0, min_value=0.0)

with col2:
    st.write("**Exemples**")
    st.markdown(
        "- Spot = 190.12\n"
        "- Strike = 200\n"
        "- r = 0.0449 (4,49%)\n"
        "- T = 30 jours ⇒ 30/365\n"
        "- σ = 25% ⇒ 0.25\n"
        "- q = 3% ⇒ 0.03\n"
        "- Prix observé = 4.50"
    )

T = max(T_days, 0)/365.0
sigma = sigma_pct/100.0
q = q_pct/100.0 if use_div else 0.0

if T > 0 and sigma > 0:
    # Prix
    if use_div:
        price = blackScholes_q(S, K, r, q, T, sigma, type_clean)
    else:
        price = blackScholes(S, K, r, T, sigma, type_clean)
    st.success(f"Prix Black-Scholes{' (avec dividendes)' if use_div else ''} : **{price:.2f} $**")

    # IV
    if market_price > 0:
        if use_div:
            iv = implied_volatility_q(market_price, S, K, r, q, T, type_clean)
        else:
            iv = implied_volatility(market_price, S, K, r, T, type_clean)
        if np.isnan(iv):
            st.warning("IV introuvable (essaie d'autres paramètres).")
        else:
            st.info(f"Volatilité implicite annualisée : **{iv:.2%}**")
            iv_converted = {
                "Annualisée": iv,
                "Quarter": iv*np.sqrt(1/4),
                "Mois": iv*np.sqrt(1/12),
                "Semaine": iv*np.sqrt(1/52),
                "Jour": iv*np.sqrt(1/252)
            }
            # Mémorise la mensuelle pour la target 1M
            st.session_state.iv_monthly = iv_converted["Mois"]*100.0  # en %
            iv_df = pd.DataFrame({
                "Horizon": list(iv_converted.keys()),
                "IV (%)": [f"{v*100:.2f}" for v in iv_converted.values()]
            })
            st.write("IV convertie par période :")
            st.table(iv_df)

    # Greeks
    greeks = greeks_with_q(S, K, r, q, T, sigma, type_clean) if use_div else greeks_plain(S, K, r, T, sigma, type_clean)
    st.write("Greeks")
    st.write(pd.DataFrame(greeks, index=["Valeur"]).T)
else:
    st.warning("Renseigne une maturité (jours) > 0 et une volatilité > 0.")

st.markdown("---")

# ======================================================================
# Section ATR
# ======================================================================
st.subheader("ATR – Volatilité historique (période personnalisée)")

c1, c2 = st.columns(2)
with c1:
    ticker_atr = st.text_input("Ticker (Yahoo Finance) – ATR", value=ticker or "AAPL")
with c2:
    date_min = st.date_input("Début", value=date(2024,1,1))
    date_max = st.date_input("Fin", value=date.today())

if st.button("Afficher ATR sur la période", key="btn_atr"):
    df_atr = yf.download(ticker_atr, start=date_min, end=date_max)
    if df_atr.empty:
        st.warning("Aucune donnée pour ce couple ticker/période.")
    else:
        # Normalise colonnes si MultiIndex
        if isinstance(df_atr.columns, pd.MultiIndex):
            df_atr.columns = df_atr.columns.get_level_values(0)

        # True Range et %TR
        df_atr['H-L']  = df_atr['High'] - df_atr['Low']
        df_atr['H-PC'] = (df_atr['High'] - df_atr['Close'].shift(1)).abs()
        df_atr['L-PC'] = (df_atr['Low']  - df_atr['Close'].shift(1)).abs()
        df_atr['TR']   = df_atr[['H-L','H-PC','L-PC']].max(axis=1)
        df_atr['TR_Perc'] = df_atr['TR'] / df_atr['Close'] * 100

        horizons = {
            "1 Mois": 21,
            "1 Trimestre": 63,
            "1 Semestre": 126,
            "1 An": 252,
            "3 Ans": 756,
            "5 Ans": 1260
        }

        results = []
        for label, n in horizons.items():
            if len(df_atr) >= n:
                tr = df_atr['TR_Perc'][-n:]
                close = df_atr['Close'][-n:]
                atr_pct = tr.mean()
                move_close = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100
                amplitude = (close.max() - close.min()) / close.iloc[0] * 100
                results.append([label, round(atr_pct,2), round(move_close,2), round(amplitude,2)])
            elif len(df_atr) > 0:
                tr = df_atr['TR_Perc']
                close = df_atr['Close']
                atr_pct = tr.mean()
                move_close = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100
                amplitude = (close.max() - close.min()) / close.iloc[0] * 100
                results.append([label + " (max dispo)", round(atr_pct,2), round(move_close,2), round(amplitude,2)])
            else:
                results.append([label, "N/A", "N/A", "N/A"])

        df_result = pd.DataFrame(results, columns=["Horizon", "ATR% Moyen", "Move Close%", "Amplitude%"])
        st.table(df_result)

        # Mémorise l'ATR mensuel si la ligne existe
        try:
            row = df_result[df_result["Horizon"]=="1 Mois"]["ATR% Moyen"]
            if not row.empty and pd.notna(row.values[0]):
                st.session_state.atr_monthly = float(row.values[0])
        except Exception:
            pass

st.markdown("---")

# ======================================================================
# Section Target 1 mois (70% IVm + 30% ATRm)
# ======================================================================
st.subheader("🎯 Target 1 mois (70% IV mensuelle + 30% ATR mensuel)")
c3, c4 = st.columns(2)
with c3:
    iv_m_input = st.number_input("IV mensuelle (%)", value=float(st.session_state.iv_monthly) if st.session_state.iv_monthly is not None else 25.0, min_value=0.0, step=0.1)
with c4:
    atr_m_input = st.number_input("ATR mensuel (%)", value=float(st.session_state.atr_monthly) if st.session_state.atr_monthly is not None else 8.0, min_value=0.0, step=0.1)

if st.button("Calculer la target 1 mois", key="btn_target"):
    target_pct = 0.7*iv_m_input + 0.3*atr_m_input
    st.success(f"Target 1 mois (mouvement attendu) : **{target_pct:.2f}%**")

    # Si spot connu, donne un range
    Sref = st.session_state.spot if st.session_state.spot is not None else S
    if Sref is not None and np.isfinite(Sref):
        up_price = Sref*(1 + target_pct/100.0)
        dn_price = Sref*(1 - target_pct/100.0)
        st.write(f"Autour du spot {Sref:.2f} $ :")
        st.markdown(f"- **Borne basse** ≈ **{dn_price:.2f} $**")
        st.markdown(f"- **Borne haute** ≈ **{up_price:.2f} $**")

st.caption("Tip : l’IV mensuelle se remplit si tu calcules l’IV ci-dessus ; l’ATR mensuel se remplit quand tu affiches l’ATR de la période.")
st.markdown("---\n*Made by GOOFY AH TRADING*")

