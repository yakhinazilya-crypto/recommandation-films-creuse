import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


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
    ### 🎯 Pourquoi ce projet ?

    La Creuse est un département rural caractérisé par :

    - une **population vieillissante**,
    - une **offre culturelle limitée**,
    - une forte appétence pour les **cinémas de proximité**.

    Le cinéma **Art & Essai** joue un rôle essentiel :
    il favorise le lien social, l’accès à la culture et la diversité cinématographique,
    en particulier pour les publics seniors.
    """)

    st.info(
        "👉 Objectif du projet : proposer un système de recommandation de films "
        "adapté aux goûts du public local de la Creuse."
    )
    st.markdown("### 📌 Indicateurs clés (KPI)")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        label="👵 Population +60 ans",
        value="36 %",
        delta="au-dessus de la moyenne nationale"
    )

    col2.metric(
        label="🎬 Salles de cinéma",
        value="4",
        delta="département rural"
    )

    col3.metric(
        label="📉 Accès à l'offre culturelle",
        value="Faible",
        delta="opportunité Art & Essai"
    )
    st.markdown("### 📊 Répartition de la population par âge (Creuse)")

    # Données simples (exemple INSEE)
    age_groups = ["0–19", "20–39", "40–59", "60+"]
    population = [18, 22, 24, 36]  # en %

    fig, ax = plt.subplots()
    ax.bar(age_groups, population)
    ax.set_ylabel("Pourcentage (%)")
    ax.set_xlabel("Tranches d'âge")
    ax.set_title("Population par tranche d'âge – Creuse")

    st.pyplot(fig)
    st.markdown("### 🎭 Accès aux équipements culturels")

    zones = ["Creuse", "Moyenne nationale"]
    access_rate = [35, 62]  # en %

    fig2, ax2 = plt.subplots()
    ax2.bar(zones, access_rate)
    ax2.set_ylabel("Accès (%)")
    ax2.set_title("Accès aux équipements culturels")

    st.pyplot(fig2)




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




    


st.header("🔗 Sources officielles")

st.markdown("""
- [Géographie du cinéma – CNC](https://www.cnc.fr/professionnels/etudes-et-rapports/statistiques/geographie-du-cinema)
- [Bilan de fréquentation 2024 – CNC](https://www.cnc.fr/professionnels/actualites/frequentation-cinematographique-en-2024)
""")
