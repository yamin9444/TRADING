#!/usr/bin/env python
# coding: utf-8

# app.py — Monte Carlo Pro (Streamlit)
# --------------------------------------------------
# Histogramme + zones 50/80 + résumé automatique
# Support auto (pct / ATR / plus bas N jours / quantile)
# Simulation vectorisée (rapide), normal ou student-t
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

TRADING_DAYS_YEAR = 252

# ---------------- Utils ----------------
@st.cache_data(show_spinner=False, ttl=3600)
def load_data(ticker: str, period: str):
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty or "Close" not in df:
        raise ValueError("Impossible de télécharger les données pour ce ticker/période.")
    price = df["Close"].astype(float).squeeze()
    returns = np.log(price / price.shift(1)).dropna()
    return df, price, returns

def ewma_sigma(returns: pd.Series, lam: float = 0.94) -> float:
    return float(returns.ewm(alpha=(1-lam)).var(bias=False).iloc[-1]**0.5)

def atr_series(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df.get("High", df["Close"])
    low  = df.get("Low",  df["Close"])
    close = df["Close"]
    tr = pd.concat([
        (high - low).abs(),
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()

def simulate_mc(S0, mu_y, sigma_y, T_days, n_sims,
                dist="normal", antithetic=True, seed=42):
    np.random.seed(seed)
    T = T_days / TRADING_DAYS_YEAR
    dt = T / T_days
    def draw(ns, steps):
        if antithetic:
            half = ns // 2
            if dist == "student_t":
                df = 6
                Zh = np.random.standard_t(df, size=(half, steps)) / np.sqrt(df/(df-2))
            else:
                Zh = np.random.standard_normal((half, steps))
            Z = np.vstack([Zh, -Zh])
            if Z.shape[0] < ns:
                extra = (np.random.standard_t(6, size=(1, steps))/np.sqrt(6/4)) if dist=="student_t" \
                        else np.random.standard_normal((1, steps))
                Z = np.vstack([Z, extra])
            return Z[:ns]
        else:
            if dist == "student_t":
                Z = np.random.standard_t(6, size=(ns, steps)) / np.sqrt(6/4)
            else:
                Z = np.random.standard_normal((ns, steps))
            return Z
    Z = draw(n_sims, T_days)
    inc = (mu_y - 0.5 * sigma_y**2) * dt + sigma_y * np.sqrt(dt) * Z
    log_paths = np.cumsum(inc, axis=1)
    S = S0 * np.exp(np.column_stack([np.zeros(n_sims), log_paths]))
    return S

def proba_range(x, low=None, high=None):
    m = np.ones_like(x, dtype=bool)
    if low  is not None: m &= (x >= low)
    if high is not None: m &= (x <= high)
    return 100.0 * m.mean()

def interval_central(samples, mass=0.50):
    a = (1.0 - mass) / 2.0
    b = 1.0 - a
    lo = float(np.quantile(samples, a))
    hi = float(np.quantile(samples, b))
    p = proba_range(samples, lo, hi)  # ~ mass*100, empirique
    return lo, hi, p

# ---------------- UI ----------------
st.set_page_config(page_title="Monte Carlo Stocks", layout="wide")
st.title("Monte Carlo – Simulation de prix (GBM)")

with st.sidebar:
    st.markdown("### Paramètres")
    ticker = st.text_input("Ticker", "AAPL").strip().upper()
    period = st.selectbox("Période historique", ["6mo","1y","2y","5y","10y","max"], index=2)
    T_days = st.slider("Horizon (jours)", 5, 60, 20, 1)
    n_sims = st.slider("# Simulations", 5_000, 100_000, 20_000, 5_000)
    seed   = st.number_input("Seed", value=42, step=1)

    st.markdown("### Modèle")
    drift_mode = st.selectbox("Drift", ["historical", "risk_neutral"])
    r_annual   = st.slider("Taux sans risque r (annualisé)", 0.0, 0.10, 0.03, 0.005)
    vol_mode   = st.selectbox("Volatilité", ["std", "ewma"])
    shock_dist = st.selectbox("Distribution des chocs", ["normal", "student_t"])
    antithetic = st.checkbox("Antithetic variates (stabilise)", True)

    st.markdown("### Support auto")
    support_mode = st.selectbox("Mode support", ["none","pct","atr","hist_low","quantile"], index=1)
    support_pct  = st.slider("Support : % sous spot", 0.02, 0.30, 0.10, 0.01)
    atr_window   = st.slider("ATR window", 5, 30, 14, 1)
    k_atr        = st.slider("k × ATR", 0.5, 4.0, 2.0, 0.5)
    hist_look    = st.slider("Lookback (jours) low/quantile", 20, 252, 60, 10)
    hist_q       = st.slider("Quantile bas (0–1)", 0.05, 0.40, 0.20, 0.05)

    st.markdown("### Questions de proba")
    pct_up = st.slider("Seuil hausse (%)", 0.02, 0.25, 0.10, 0.01)
    range_low  = st.number_input("Plage: bas", value=220.0)
    range_high = st.number_input("Plage: haut", value=240.0)

# ---------------- Core ----------------
try:
    df, price, rets = load_data(ticker, period)
    S0 = float(price.iloc[-1])

    # Volatilité
    sigma_daily = ewma_sigma(rets) if vol_mode == "ewma" else float(rets.std(ddof=1))
    sigma_year  = sigma_daily * np.sqrt(TRADING_DAYS_YEAR)

    # Drift
    mu_year_hist = float(rets.mean()) * TRADING_DAYS_YEAR
    mu_y = r_annual if drift_mode == "risk_neutral" else mu_year_hist

    # Support auto
    def compute_support(mode: str):
        if mode == "none": return None
        if mode == "pct":  return float(S0 * (1 - support_pct))
        if mode == "atr":
            atr = atr_series(df, atr_window)
            if atr.notna().any():
                return float(S0 - k_atr * atr.iloc[-1])
            return None
        look = price.tail(hist_look)
        if look.empty: return None
        if mode == "hist_low": return float(look.min())
        if mode == "quantile": return float(np.quantile(look.values, hist_q))
        return None

    support = compute_support(support_mode)
    if support is not None and support > S0 * 1.5:
        support = None  # garde-fou

    # Simulation
    S = simulate_mc(S0, mu_y, sigma_year, T_days, n_sims,
                    dist=shock_dist, antithetic=antithetic, seed=seed)
    final = S[:, -1]

    # Stats & zones
    mean_ = float(final.mean())
    ci95  = (float(np.quantile(final, 0.025)), float(np.quantile(final, 0.975)))
    q25,q50,q75 = [float(np.quantile(final, q)) for q in (0.25,0.50,0.75)]
    low50, high50, p50 = interval_central(final, 0.50)
    low80, high80, p80 = interval_central(final, 0.80)

    # Probabilités utiles
    p_up        = proba_range(final, S0*(1+pct_up), None)
    p_in_range  = proba_range(final, range_low, range_high) if range_high > range_low else np.nan
    p_below_sup = proba_range(final, None, support) if support is not None else np.nan

    # =========== Encadré "Analytique vs Monte Carlo" ===========
    T_years = T_days / TRADING_DAYS_YEAR
    analytic_mean = float(S0 * np.exp(mu_y * T_years))
    diff_abs = mean_ - analytic_mean
    diff_pct = 100.0 * diff_abs / analytic_mean if analytic_mean != 0 else 0.0

    st.markdown("### 📐 Prix attendu – analytique vs simulation")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Spot S₀", f"{S0:,.2f}")
    m2.metric("E[S_T] analytique", f"{analytic_mean:,.2f}")
    m3.metric("Moyenne Monte Carlo", f"{mean_:,.2f}")
    m4.metric("Écart MC vs analytique", f"{diff_abs:,.2f}", f"{diff_pct:+.2f}%")

    # ----------- Résumé texte -----------
    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown(f"**Ticker** : `{ticker}`")
        st.markdown(f"**Horizon** : {T_days} jours")
        st.markdown(f"**Drift** : {'r (risk-neutral)' if drift_mode=='risk_neutral' else 'μ historique'}")
        st.markdown(f"**Vol annualisée σ** : {sigma_year:.4f}  *(mode: {vol_mode})*")
        st.markdown(f"**IC95%** : [{ci95[0]:,.2f} ; {ci95[1]:,.2f}]")
        st.markdown(f"**Quartiles** Q25/Q50/Q75 : {q25:.2f} / {q50:.2f} / {q75:.2f}")

    with col2:
        st.markdown("**Probas**")
        st.markdown(f"- P(> +{int(pct_up*100)}%) = **{p_up:.2f}%**  (seuil {S0*(1+pct_up):.2f})")
        if support is not None:
            st.markdown(f"- P(< support {support:.2f}) = **{p_below_sup:.2f}%**")
        if not np.isnan(p_in_range):
            st.markdown(f"- P({range_low:.2f} ≤ prix ≤ {range_high:.2f}) = **{p_in_range:.2f}%**")

        st.markdown("**Résumé automatique**")
        st.markdown(
            f"- Il y a **~{p50:.1f}%** de chances que le prix soit **entre {low50:.2f} et {high50:.2f}** dans **{T_days} jours**.  \n"
            f"- Il y a **~{p80:.1f}%** de chances que le prix soit **entre {low80:.2f} et {high80:.2f}** dans **{T_days} jours**."
        )

    # ----------- Histogramme + Trajectoires côte à côte -----------
    c1, c2 = st.columns(2)

    with c1:
        fig1, ax1 = plt.subplots(figsize=(6,4))
        ax1.axvspan(low80, high80, color="skyblue", alpha=0.25, zorder=0, label=f"Zone 80% [{low80:.0f}-{high80:.0f}]")
        ax1.axvspan(low50, high50, color="deepskyblue", alpha=0.35, zorder=1, label=f"Zone 50% [{low50:.0f}-{high50:.0f}]")
        ax1.hist(final, bins=70, edgecolor="black", alpha=0.75, zorder=2)
        ax1.axvline(mean_, color="red", linestyle="--", linewidth=2, label=f"Moyenne MC = {mean_:.2f}", zorder=3)
        ax1.axvline(ci95[0], color="green", linestyle="--", linewidth=2, label=f"IC95% = [{ci95[0]:.2f}; {ci95[1]:.2f}]", zorder=3)
        ax1.axvline(ci95[1], color="green", linestyle="--", linewidth=2, zorder=3)
        ax1.set_title(f"Distribution des prix simulés ({ticker})")
        ax1.set_xlabel("Prix simulé"); ax1.set_ylabel("Fréquence")
        ax1.legend()
        st.pyplot(fig1, clear_figure=True)

    with c2:
        fig2, ax2 = plt.subplots(figsize=(6,4))
        t = np.arange(T_days + 1)
        idx = np.random.choice(S.shape[0], size=min(120, S.shape[0]), replace=False)
        for i in idx:
            ax2.plot(t, S[i, :], linewidth=0.8, alpha=0.6)
        if support is not None:
            ax2.axhline(support, linestyle=":", linewidth=1.8, label=f"Support = {support:.2f}")
        ax2.set_title(f"Trajectoires Monte Carlo ({len(idx)} paths)")
        ax2.set_xlabel("Jour"); ax2.set_ylabel("Prix")
        ax2.legend()
        st.pyplot(fig2, clear_figure=True)

except Exception as e:
    st.error(f"Erreur : {e}")
