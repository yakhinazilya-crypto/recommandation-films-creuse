import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Cinéma Art & Essai – Creuse",
    layout="wide"
)


# SIDEBAR
st.sidebar.title("🎬 Cinéma Art & Essai – Creuse")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Contexte Creuse",
        "📊 Analyse du marché",
        "🎬 Recommandation par genres",
        "🎭 Recommandation par acteur"
    ]
)


# PAGE 1 — CONTEXTE
if page == "🏠 Contexte Creuse":
    st.title("🏠 Contexte socio-culturel de la Creuse")

    st.markdown("""
    La Creuse est un département marqué par :
    - une population vieillissante,
    - un accès culturel limité,
    - un fort potentiel pour le cinéma Art & Essai.
    """)

    st.info("Cette application vise à proposer des recommandations adaptées aux publics locaux.")


# PAGE 2 — ANALYSE DU MARCHÉ
elif page == "📊 Analyse du marché":
    st.title("📊 Analyse du marché du cinéma")

    st.write("Analyse exploratoire des films disponibles depuis 1960.")

# PAGE 3 — GENRES
elif page == "🎬 Recommandation par genres":
    st.title("🎬 Recommandation de films par genres")

    st.write("Sélectionnez un film pour obtenir des recommandations similaires.")


# PAGE 4 — ACTEURS
elif page == "🎭 Recommandation par acteur":
    st.title("🎭 Recommandation de films par acteur")

    st.write("Sélectionnez un acteur pour découvrir des films recommandés.")


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# CONFIGURATION GÉNÉRALE
# ===============================
st.set_page_config(
    page_title="Cinéma en Creuse – Étude de marché",
    layout="wide"
)

# ===============================
# TITRE PRINCIPAL
# ===============================
st.title("🎬 Étude de marché – Cinéma en Creuse")

st.markdown("""
Cette page présente le **contexte démographique et culturel**  
pour justifier un **système de recommandation de films adapté à la Creuse**.
""")

st.divider()

# ===============================
# CONTEXTE NATIONAL
# ===============================
st.header("📊 Contexte national du cinéma (France)")

st.markdown("""
- **Fréquentation nationale 2024** : **181 millions d'entrées**
- **Public 60 ans et +** : **6,3 entrées/an**
- **Croissance du cinéma Art & Essai** : **+2,9%**
""")

st.info(
    "👉 Le public senior est un public actif et fidèle au cinéma, "
    "particulièrement pour les films Art & Essai."
)

st.divider()

# ===============================
# CONTEXTE CREUSE
# ===============================
st.header("🗺️ Spécificités de la Creuse")

st.markdown("""
- Département **rural**
- **Population vieillissante**
- Offre culturelle plus limitée
- Fort potentiel pour une programmation ciblée
""")

st.success(
    "🎯 Objectif du projet : proposer des films adaptés "
    "aux goûts du public senior de la Creuse."
)

st.divider()

# ===============================
# SOURCES
# ===============================
st.header("🔗 Sources officielles")

st.markdown("""
- [Géographie du cinéma – CNC](https://www.cnc.fr/professionnels/etudes-et-rapports/statistiques/geographie-du-cinema)
- [Bilan de fréquentation 2024 – CNC](https://www.cnc.fr/professionnels/actualites/frequentation-cinematographique-en-2024)
""")
