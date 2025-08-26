# Streamlit Multipage Starter

## Lancer en local
```bash
pip install -r requirements.txt
streamlit run Home.py
```

## Ajouter une page
- Crée un fichier Python dans `pages/` avec un préfixe numérique : `4_Ma_Nouvelle_Page.py`.
- Le menu se met à jour automatiquement.

## Partage d'état
- Utilise `st.session_state` pour partager des variables entre pages.
- Exemple : `st.session_state["ticker_global"] = "AAPL"` dans une page, puis accès depuis une autre page.
