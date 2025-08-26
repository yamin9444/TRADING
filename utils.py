import streamlit as st
import pandas as pd

@st.cache_data
def charger_csv(path: str) -> pd.DataFrame:
    """Charge un CSV et le met en cache (recalcul si le fichier change)."""
    return pd.read_csv(path)
