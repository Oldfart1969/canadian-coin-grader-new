import streamlit as st
from PIL import Image
from grading_engine import grade_coin_v2, validate_image

st.set_page_config(
    page_title="Canadian Coin Grader",
    page_icon="🪙",
    layout="wide"
)

st.title("🇨🇦 Canadian Coin Grader — Reference Calibrated")
st.caption(
    "Estimation photographique basée sur l’analyse d’image et une banque locale "
    "de références canadiennes gradées. Ce résultat n’est pas une certification ICCS/PCGS/NGC."
)

with st.sidebar:
    st.header("Identification de la pièce")
    denomination = st.selectbox(
        "Dénomination",
        ["1-cent", "5-cent", "10-cent", "25-cent", "50-cent", "1-dollar"]
    )
    year = st.number_input("Année", min_value=1858, max_value=2026, value=1936, step=1)
    strike = st.selectbox(
        "Type de frappe",
        ["Business strike", "Proof-Like", "Specimen", "Proof"]
    )

    st.divider()
    st.markdown("**Pour de meilleurs résultats**")
    st.markdown(
        "- Une photo nette de chaque côté\n"
        "- Pièce centrée et photographiée à plat\n"
        "- Éclairage diffus, sans reflet brûlé\n"
        "- Résolution élevée\n"
        "- Pas de filtre ni d’accentuation artificielle"
    )

col1, col2 = st.columns(2)

with col1:
    st.subheader("Avers")
    obv_file = st.file_uploader(
        "Téléverser l’avers",
        type=["jpg", "jpeg", "png", "webp"],
        key="obverse"
    )

with col2:
    st.subheader("Revers")
    rev_file = st.file_uploader(
        "Téléverser le revers",
        type=["jpg", "jpeg", "png", "webp"],
        key="reverse"
    )

if obv_file and rev_file:
    obv = Image.open(obv_file).convert("RGB")
    rev = Image.open(rev_file).convert("RGB")

    p1, p2 = st.columns(2)
    p1.image(obv, caption="Avers", use_container_width=True)
    p2.image(rev, caption="Revers", use_container_width=True)

    issues = validate_image(obv, "avers") + validate_image(rev, "revers")
    for issue in issues:
        st.warning(issue)

    if st.button("Grader la pièce", type="primary", use_container_width=True):
        result = grade_coin_v2(
            obv,
            rev,
            denomination,
            int(year),
            strike
        )

        st.divider()
        a, b, c = st.columns(3)
        a.metric("Grade estimé", result["grade"])
        b.metric("Fourchette plausible", f'{result["range"][0]} – {result["range"][1]}')
        c.metric("Confiance photographique", f'{result["confidence"]:.0%}')

        st.subheader("Évaluation des deux faces")
        s1, s2 = st.columns(2)
        s1.metric("Avers", result["side_estimates"]["obverse"])
        s2.metric("Revers", result["side_estimates"]["reverse"])

        st.subheader("Analyse visuelle")
        metrics = result["metrics"]
        cols = st.columns(5)
        labels = [
            ("Détails", metrics["detail"]),
            ("Surfaces", metrics["surface"]),
            ("Contraste", metrics["contrast"]),
            ("Lustre", metrics["luster"]),
            ("Qualité des marques", metrics["marks_quality"]),
        ]
        for col, (label, value) in zip(cols, labels):
            col.metric(label, f"{value}/10")

        st.subheader("Pourquoi ce grade?")
        for reason in result["reasons"]:
            st.write("• " + reason)

        st.subheader("Références canadiennes les plus proches")
        if result["comparables"]:
            st.dataframe(
                result["comparables"],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Aucune référence suffisamment proche n’a été trouvée pour cette sélection.")

        st.caption(
            f"Banque locale utilisée : {result['meta']['reference_count']} références gradées. "
            "Les grades Mint State restent plus difficiles à établir sur deux photos fixes, "
            "notamment à cause du lustre en mouvement et des hairlines."
        )
else:
    st.info("Téléverse une photo de l’avers et une photo du revers pour commencer.")
