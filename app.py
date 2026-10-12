import html
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Paths & constants
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best_lightgbm_regressor.pkl"
SCALER_PATH = BASE_DIR / "scaler_advanced.pkl"
DATA_PATH = BASE_DIR / "cutoffs_app.csv.gz"  # slim, compressed copy of cutoffs.csv
ENCODINGS_PATH = BASE_DIR / "encodings.json"  # optional, see README

# Order must match the scaler / model training order
FEATURES = ["year_numeric", "source_page", "univ_target_enc", "page_target_enc"]
YEARS = [2024, 2025, 2026]
CSV_COLS = [
    "university_id", "university", "year", "programme_id", "programme",
    "category", "figure", "source_page",
]

CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {
    padding: 1.8rem 2rem; border-radius: 18px; margin-bottom: 1.4rem; color: #fff;
    background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
    box-shadow: 0 10px 30px rgba(79, 70, 229, .25);
}
.hero h1 {margin: 0; font-size: 2.1rem; color: #fff; padding: 0;}
.hero p {margin: .5rem 0 0; opacity: .92; font-size: 1.02rem;}
.result {
    border-radius: 18px; padding: 1.6rem 1.2rem; text-align: center;
    background: rgba(124, 58, 237, .10); border: 1px solid rgba(124, 58, 237, .35);
}
.result .label {font-size: .8rem; letter-spacing: .1em; text-transform: uppercase; opacity: .75;}
.result .value {font-size: 3.4rem; font-weight: 800; line-height: 1.15; color: #7c3aed;}
.result .sub {font-size: .92rem; opacity: .8; margin-top: .2rem;}
.chip {
    display: inline-block; padding: .2rem .7rem; margin: .15rem .25rem 0 0;
    border-radius: 999px; font-size: .78rem; background: rgba(124, 58, 237, .14);
}
div[data-testid="stMetric"] {
    background: rgba(128, 128, 128, .08); border-radius: 14px; padding: .8rem 1rem;
}
</style>
"""


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


@st.cache_data(show_spinner="Loading cutoff data…")
def load_data():
    df = pd.read_csv(DATA_PATH, usecols=CSV_COLS)
    df = df[df["figure"].notna() & df["source_page"].notna()].copy()
    df["source_page"] = df["source_page"].astype(int)

    univ_enc = df.groupby("university_id")["figure"].mean().to_dict()
    page_enc = df.groupby("source_page")["figure"].mean().to_dict()
    prog_page = (
        df.groupby("programme_id")["source_page"].median().round().astype(int).to_dict()
    )
    from_file = False
    if ENCODINGS_PATH.exists():
        saved = json.loads(ENCODINGS_PATH.read_text())
        univ_enc.update({k: float(v) for k, v in saved.get("university", {}).items()})
        page_enc.update({int(k): float(v) for k, v in saved.get("source_page", {}).items()})
        from_file = True

    lookups = {
        "univ_enc": univ_enc,
        "page_enc": page_enc,
        "prog_page": prog_page,
        "global_mean": float(df["figure"].mean()),
        "from_file": from_file,
    }
    return df, lookups


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
def predict_scores(model, scaler, years, page, univ_enc, page_enc):
    X = pd.DataFrame(
        {
            "year_numeric": list(years),
            "source_page": page,
            "univ_target_enc": univ_enc,
            "page_target_enc": page_enc,
        }
    )[FEATURES]
    preds = model.predict(scaler.transform(X))
    return np.clip(preds, 0, None)


def short(text, n=90):
    text = str(text)
    return text if len(text) <= n else text[: n - 1] + "…"


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="CUET Cutoff Predictor", page_icon="🎓", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    try:
        model, scaler = load_artifacts()
        df, lk = load_data()
    except Exception as e:
        st.error(f"Could not load the app files: {e}")
        st.info(
            "Make sure `best_lightgbm_regressor.pkl`, `scaler_advanced.pkl` and "
            "`cutoffs_app.csv.gz` are in the same folder as `app.py`."
        )
        st.stop()

    # ---- Sidebar -----------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🎓 About")
        st.write(
            "Estimate the admission cutoff score for a programme using a LightGBM "
            "model trained on published university cutoff lists."
        )
        st.divider()
        st.metric("Cutoff records", f"{len(df):,}")
        st.metric("Universities", df["university_id"].nunique())
        st.metric("Programmes", df["programme_id"].nunique())
        st.divider()
        st.caption(
            "Estimates only. Always check the official counselling portal of the "
            "university for actual cutoffs."
        )

    # ---- Hero --------------------------------------------------------------
    st.markdown(
        """
        <div class="hero">
          <h1>🎓 University Cutoff Predictor</h1>
          <p>Pick a university and programme to get an estimated cutoff score,
          plus how it compares with past years.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_predict, tab_explore, tab_about = st.tabs(
        ["🔮 Predict", "📊 Explore data", "ℹ️ How it works"]
    )

    # ======================= Predict =======================================
    with tab_predict:
        left, right = st.columns([1, 1], gap="large")

        with left:
            st.subheader("Choose a course")
            unis = (
                df[["university_id", "university"]]
                .drop_duplicates()
                .sort_values("university")
            )
            uni_names = dict(zip(unis["university_id"], unis["university"]))
            uid = st.selectbox(
                "University", list(uni_names), format_func=lambda x: uni_names[x]
            )

            progs = (
                df[df["university_id"] == uid][["programme_id", "programme"]]
                .drop_duplicates("programme_id")
                .sort_values("programme")
            )
            prog_names = dict(zip(progs["programme_id"], progs["programme"]))
            pid = st.selectbox(
                "Programme",
                list(prog_names),
                format_func=lambda x: short(prog_names[x]),
            )
            year = st.select_slider("Admission year", options=YEARS, value=YEARS[-1])

            # Derived model inputs
            page = lk["prog_page"].get(pid, 1)
            u_enc = lk["univ_enc"].get(uid, lk["global_mean"])
            p_enc = lk["page_enc"].get(page, lk["global_mean"])

            with st.expander("Advanced: model inputs"):
                st.caption(
                    "These are filled in automatically from your selection. "
                    "Turn on manual mode to override them."
                )
                manual = st.toggle("Enter values manually", value=False)
                if manual:
                    page = int(
                        st.number_input("source_page", 1, 1000, int(page), key=f"sp_{pid}")
                    )
                    u_enc = st.number_input(
                        "univ_target_enc", value=float(u_enc), key=f"ue_{uid}"
                    )
                    p_enc = st.number_input(
                        "page_target_enc", value=float(p_enc), key=f"pe_{pid}"
                    )
                else:
                    st.write(
                        f"source_page = **{page}**, univ_target_enc = **{u_enc:.1f}**, "
                        f"page_target_enc = **{p_enc:.1f}**"
                    )

        preds = predict_scores(model, scaler, YEARS, page, u_enc, p_enc)
        pred_by_year = dict(zip(YEARS, preds))
        pred = pred_by_year[year]
        hist = df[df["programme_id"] == pid]

        with right:
            st.subheader("Estimated cutoff")
            st.markdown(
                f"""
                <div class="result">
                  <div class="label">Predicted cutoff score · {year}</div>
                  <div class="value">{pred:,.1f}</div>
                  <div class="sub">{html.escape(short(prog_names[pid], 70))}</div>
                  <div class="sub">{html.escape(uni_names[uid])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            m1, m2, m3 = st.columns(3)
            if len(hist):
                m1.metric("Past average", f"{hist['figure'].mean():,.0f}")
                m2.metric("Past highest", f"{hist['figure'].max():,.0f}")
            else:
                m1.metric("Past average", "n/a")
                m2.metric("Past highest", "n/a")
            m3.metric(
                "University avg",
                f"{df[df['university_id'] == uid]['figure'].mean():,.0f}",
            )

        st.divider()
        st.subheader("Trend 2024 – 2026")
        actual = hist.groupby("year")["figure"].mean()
        trend = pd.DataFrame(
            {
                "Model estimate": [pred_by_year[y] for y in YEARS],
                "Historical average": [actual.get(y, np.nan) for y in YEARS],
            },
            index=[str(y) for y in YEARS],
        )
        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            st.bar_chart(trend)
        with c2:
            st.caption(
                "Historical average is the mean of all listed cutoffs for this "
                "programme (all categories and rounds). The model does not use "
                "category or round, so its estimate is a general level for the "
                "programme, not a number for one specific category."
            )

        if len(hist):
            with st.expander("Past cutoffs by category"):
                pivot = (
                    hist.pivot_table(
                        index="category", columns="year", values="figure", aggfunc="mean"
                    )
                    .round(1)
                    .sort_index()
                )
                st.dataframe(pivot)

    # ======================= Explore =======================================
    with tab_explore:
        st.subheader("Browse historical cutoffs")
        f1, f2, f3, f4 = st.columns([2, 1, 2, 2])
        with f1:
            ex_uid = st.selectbox(
                "University",
                ["All"] + list(uni_names),
                format_func=lambda x: "All universities" if x == "All" else uni_names[x],
                key="ex_uni",
            )
        with f2:
            ex_year = st.selectbox("Year", YEARS[::-1], key="ex_year")
        with f3:
            cats = sorted(df["category"].unique())
            ex_cats = st.multiselect(
                "Category", cats, default=["UR"] if "UR" in cats else [], key="ex_cat"
            )
        with f4:
            q = st.text_input("Search programme", placeholder="e.g. Economics")

        sub = df[df["year"] == ex_year]
        if ex_uid != "All":
            sub = sub[sub["university_id"] == ex_uid]
        if ex_cats:
            sub = sub[sub["category"].isin(ex_cats)]
        if q:
            sub = sub[sub["programme"].str.contains(q, case=False, regex=False)]

        if sub.empty:
            st.info("No records match these filters.")
        else:
            agg = (
                sub.groupby(["university", "programme"])["figure"]
                .agg(Average="mean", Highest="max", Lowest="min", Records="count")
                .round(1)
                .reset_index()
                .sort_values("Average", ascending=False)
                .rename(columns={"university": "University", "programme": "Programme"})
            )
            st.caption(f"{len(agg):,} programmes · {len(sub):,} records")
            top = agg.head(15).copy()
            top["Programme"] = top["Programme"].map(lambda s: short(s, 40))
            st.bar_chart(top.set_index("Programme")["Average"])
            st.dataframe(agg, hide_index=True)
            st.download_button(
                "Download table (CSV)",
                agg.to_csv(index=False).encode("utf-8"),
                file_name="cutoffs_filtered.csv",
                mime="text/csv",
            )

    # ======================= About =========================================
    with tab_about:
        st.subheader("How it works")
        st.markdown(
            """
            1. You choose a **university**, **programme** and **year**.
            2. The app builds four numeric inputs for the model:
               `year_numeric`, `source_page`, `univ_target_enc` and `page_target_enc`.
            3. The inputs are scaled with the saved `StandardScaler`.
            4. A **LightGBM regressor** returns the estimated cutoff score.

            **Good to know**
            - The model does not take category (UR, OBC, SC, ST…) or counselling round
              as inputs, so the estimate is a general level for the programme.
            - Only score-based cutoffs are included; rank-based lists are excluded.
            - Treat results as a guide, not an official figure.
            """
        )
        if lk["from_file"]:
            st.success("Using target encodings from `encodings.json`.")
        else:
            st.warning(
                "Target encodings are computed from the cutoff data. For results that "
                "exactly match training, add the `encodings.json` saved from your "
                "training notebook."
            )


if __name__ == "__main__":
    main()
