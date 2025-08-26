import streamlit as st

# Configuration de la page
st.set_page_config(page_title="Divitae Capital", page_icon="💹", layout="wide")

# Centrer le texte avec HTML
st.markdown(
    """
    <div style="text-align: center; padding-top: 100px;">
        <h1>Bienvenue sur le Pricing <span style="color:#4F46E5;">DIVITAE CAPITAL</span></h1>
        <br><br>
        <h3>🚀 Étapes à suivre :</h3>
        <p style="font-size:18px;">1️⃣ Classement Secteur / Industry</p>
        <p style="font-size:18px;">2️⃣ Classement Entreprise</p>
        <p style="font-size:18px;">3️⃣ Black-Scholes ATR</p>
        <p style="font-size:18px;">4️⃣ IVRank</p>
    </div>
    """,
    unsafe_allow_html=True
)
