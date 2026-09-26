from __future__ import annotations

import base64
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# Resolve artifacts from the repository root so the app can be launched from any
# working directory with: streamlit run app.py
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
METRICS_PATH = ROOT / "data" / "results" / "metrics_summary.csv"
PRICE_BAND_PATH = ROOT / "data" / "results" / "price_band_summary.csv"
APE_DISTRIBUTION_PATH = ROOT / "data" / "results" / "ape_distribution_summary.csv"
ERROR_BUCKETS_PATH = ROOT / "data" / "results" / "lightgbm_error_buckets.csv"

# Keep user-facing model names separate from their local artifact filenames.
MODEL_FILES = {
    "LightGBM": "lightgbm.joblib",
    "XGBoost": "xgboost.joblib",
    "Random Forest": "random_forest.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Linear Regression": "linear_regression.joblib",
}

# Parameters fitted on the Nov 2023-Apr 2026 training partition.
SCALER = {
    "LivingArea": (2009.9928049303055, 934.8753700663731),
    "LotSizeSquareFeet": (257832.1489571616, 14396407.851855999),
    "AssociationFee": (68.37882000983043, 245.34309013708688),
    "BathroomsPerBedroom": (0.75004755030323, 0.25066943803682734),
    "PropertyAge": (49.11489237012578, 27.44935641091912),
    "LivingAreaPerBedroom": (574.6531555228848, 187.99780140212113),
    "LivingAreaLotRatio": (11.816692220707267, 599.497812456248),
    "LivingAreaPerBathroom": (789.6230722918415, 197.7717325802333),
    "LotSizePerBedroom": (78370.75627942286, 4645198.253898754),
    "GarageSpacesPerBedroom": (0.5842926508717189, 0.2738121655500193),
}

FLOORING_OPTIONS = [
    "Bamboo", "Brick", "Carpet", "Concrete", "Laminate", "SeeRemarks",
    "Stone", "Tile", "Unknown", "Vinyl", "Wood",
]
LEVEL_OPTIONS = ["MultiSplit", "One", "ThreeOrMore", "Two", "Unknown"]


@st.cache_resource
def load_artifacts() -> tuple[list[str], dict[str, object]]:
    """Load the saved feature order and fitted models once per app process."""
    features = list(joblib.load(MODEL_DIR / "model_features.joblib"))
    models = {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in MODEL_FILES.items()
    }
    return features, models


@st.cache_data
def load_metrics() -> pd.DataFrame:
    """Load the held-out test metrics used by the performance page."""
    return pd.read_csv(METRICS_PATH)


@st.cache_data
def load_segment_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the Week 8 price-band and error-distribution summaries."""
    return (
        pd.read_csv(PRICE_BAND_PATH),
        pd.read_csv(APE_DISTRIBUTION_PATH),
        pd.read_csv(ERROR_BUCKETS_PATH),
    )


@st.cache_data
def load_hero_data_url() -> str:
    """Return an optimized project image as an embeddable CSS data URL."""
    encoded = base64.b64encode((ROOT / "assets" / "california-home-hero.jpg").read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return a ratio while preserving zero-denominator cases for imputation."""
    return numerator / denominator if denominator > 0 else np.nan


def scale_value(name: str, value: float) -> float:
    """Apply the same training-fitted standardization used in Notebook 2."""
    mean, scale = SCALER[name]
    if pd.isna(value):
        # Training medians expressed in raw units.
        medians = {
            "BathroomsPerBedroom": 2 / 3,
            "LivingAreaPerBedroom": 538.3333333333334,
            "LivingAreaPerBathroom": 767.3333333333334,
            "LivingAreaLotRatio": 0.236257061122561,
            "LotSizePerBedroom": 2178.0,
            "GarageSpacesPerBedroom": 2 / 3,
        }
        value = medians.get(name, mean)
    return (float(value) - mean) / scale


def make_feature_row(values: dict, features: list[str]) -> pd.DataFrame:
    """Convert one raw property form submission into the 127-feature schema."""
    # Start at zero so absent one-hot categories and false indicators use the
    # same representation as the encoded training data.
    row = {feature: 0.0 for feature in features}
    close_date = values["CloseDate"]
    bedrooms = float(values["BedroomsTotal"])
    bathrooms = float(values["BathroomsTotalInteger"])
    living_area = float(values["LivingArea"])
    lot_size = float(values["LotSizeSquareFeet"])
    garage = float(values["GarageSpaces"])
    property_age = max(close_date.year - int(values["YearBuilt"]), 0)

    # Copy direct model inputs before deriving ratios and interactions.
    booleans = [
        "ViewYN", "WaterfrontYN", "BasementYN", "PoolPrivateYN",
        "AttachedGarageYN", "FireplaceYN", "NewConstructionYN",
    ]
    numeric = [
        "Latitude", "Longitude", "ParkingTotal", "YearBuilt",
        "BathroomsTotalInteger", "BedroomsTotal", "Stories", "LotSizeArea",
        "MainLevelBedrooms", "GarageSpaces", "AmenityCount",
        "DistrictType_Elementary", "DistrictType_High", "DistrictType_Unified",
    ]
    for name in booleans:
        row[name] = int(values[name])
    for name in numeric:
        row[name] = float(values[name])

    # Recreate the temporal, ratio, amenity, and interaction features from the
    # leakage-aware preprocessing notebook.
    row.update({
        "LivingArea": living_area,
        "AssociationFee": float(values["AssociationFee"]),
        "LotSizeSquareFeet": lot_size,
        "CloseYear": close_date.year,
        "CloseMonth": close_date.month,
        "PropertyAge": property_age,
        "BathroomsPerBedroom": safe_ratio(bathrooms, bedrooms),
        "LivingAreaPerBedroom": safe_ratio(living_area, bedrooms),
        "LivingAreaPerBathroom": safe_ratio(living_area, bathrooms),
        "LivingAreaLotRatio": safe_ratio(living_area, lot_size),
        "LotSizePerBedroom": safe_ratio(lot_size, bedrooms),
        "GarageSpacesPerBedroom": safe_ratio(garage, bedrooms),
        "PoolAndWaterfront": int(values["PoolPrivateYN"] and values["WaterfrontYN"]),
        "WaterfrontAndView": int(values["WaterfrontYN"] and values["ViewYN"]),
        "PoolAndGarage": int(values["PoolPrivateYN"] and values["AttachedGarageYN"]),
        "FireplaceAndBasement": int(values["FireplaceYN"] and values["BasementYN"]),
        "NewConstructionAndGarage": int(values["NewConstructionYN"] and values["AttachedGarageYN"]),
        "PremiumAmenityCombo": int(values["PoolPrivateYN"] and values["ViewYN"] and values["FireplaceYN"]),
    })

    # Activate only categories that were present in the training schema.
    age_bucket = "New" if property_age <= 5 else "Modern" if property_age <= 20 else "Mature" if property_age <= 50 else "Historic"
    categorical = [
        f"CountyOrParish_{values['CountyOrParish']}",
        f"AssociationFeeFrequency_{values['AssociationFeeFrequency']}",
        f"AgeBucket_{age_bucket}",
        f"Levels_{values['Levels']}",
    ] + [f"Flooring_{item}" for item in values["Flooring"]]
    for name in categorical:
        if name in row:
            row[name] = 1

    # Standardize the same ten numerical columns scaled during training, then
    # enforce the exact saved feature order expected by every model.
    for name in SCALER:
        row[name] = scale_value(name, row[name])
    return pd.DataFrame([row], columns=features).astype(float)


def validate_encoded_frame(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Validate and order an uploaded encoded batch before prediction."""
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        preview = ", ".join(missing[:8])
        raise ValueError(f"Missing {len(missing)} required model features, including: {preview}")
    result = frame[features].apply(pd.to_numeric, errors="coerce")
    if result.isna().any().any():
        bad = result.columns[result.isna().any()].tolist()
        raise ValueError(f"These features contain missing or non-numeric values: {', '.join(bad[:8])}")
    return result


# Configure the page before rendering any Streamlit elements.
st.set_page_config(page_title="California Home Intelligence", layout="wide")

# Apply a compact editorial design system across the public product pages.
st.markdown(
    """
    <style>
    :root {
        --ink: #14231d;
        --muted: #617169;
        --forest: #123d2f;
        --sage: #ddefe6;
        --paper: #f7f8f4;
        --line: #d9e1dc;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"], .stApp > header { display: none !important; }
    [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
    .block-container {
        max-width: 1180px;
        padding-top: 1.25rem;
        padding-bottom: 3rem;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: var(--ink);
        font-family: Georgia, "Times New Roman", serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.025em;
    }
    p, label, [data-testid="stCaptionContainer"] { color: var(--muted); }
    .site-header {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        min-height: 3.4rem;
    }
    .brand-mark {
        display: flex;
        width: 2.45rem;
        height: 2.45rem;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        background: var(--forest);
        color: white;
        font-family: Georgia, serif;
        font-size: 0.88rem;
        font-weight: 700;
    }
    .brand-name { color: var(--ink) !important; font-size: 0.98rem; font-weight: 750; }
    .brand-subtitle { color: var(--muted) !important; font-size: 0.72rem; }
    [data-testid="stSegmentedControl"], [data-testid="stButtonGroup"] div[role="radiogroup"] {
        justify-content: flex-end;
        gap: 0.25rem;
    }
    [data-testid="stSegmentedControl"] button,
    [data-testid="stButtonGroup"] button,
    button[role="radio"] {
        min-height: 2.45rem;
        border: 0 !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        color: var(--muted) !important;
        font-size: 0.86rem;
    }
    [data-testid="stSegmentedControl"] button[aria-pressed="true"],
    [data-testid="stButtonGroup"] button[aria-checked="true"],
    button[role="radio"][aria-checked="true"] {
        border-bottom-color: var(--forest) !important;
        background: transparent !important;
        color: var(--forest) !important;
        font-weight: 750;
    }
    .top-rule { margin: 0.35rem 0 0; border-top: 1px solid var(--line); }
    .eyebrow {
        margin-bottom: 1rem;
        color: #356651;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }
    .hero { padding: 4rem 0 3.5rem; }
    .hero h1 {
        max-width: 800px;
        margin: 0;
        font-family: Georgia, "Times New Roman", serif !important;
        font-weight: 600 !important;
        font-size: clamp(3rem, 5.2vw, 4.7rem);
        line-height: 0.98;
        overflow-wrap: normal;
        word-break: normal;
        hyphens: none;
    }
    .hero-copy {
        max-width: 680px;
        margin-top: 1.6rem;
        color: var(--muted);
        font-size: 1.14rem;
        line-height: 1.75;
    }
    .st-key-home_hero {
        min-height: 38rem;
        margin: 2.5rem 0 3rem;
        padding: 2.25rem;
        border-radius: 20px;
        background-position: center;
        background-size: cover;
        overflow: hidden;
    }
    .st-key-home_hero [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {
        padding: 0 2.25rem 2.25rem;
        border: 1px solid rgba(255, 255, 255, 0.55);
        border-radius: 16px;
        background: rgba(247, 248, 244, 0.92);
        backdrop-filter: blur(4px);
    }
    .st-key-home_hero .hero {
        padding: 3rem 0 2rem;
    }
    .section { padding: 4rem 0; border-top: 1px solid var(--line); }
    .section h2 { max-width: 760px; margin: 0 0 1rem; font-size: 2.7rem; line-height: 1.08; }
    .evidence-strip {
        display: grid;
        grid-template-columns: 1.2fr repeat(3, 1fr);
        margin: 0 0 4rem;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: white;
        overflow: hidden;
    }
    .evidence-item { min-height: 8.5rem; padding: 1.45rem; border-right: 1px solid var(--line); }
    .evidence-item:last-child { border-right: 0; }
    .evidence-label { color: var(--muted); font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
    .evidence-value { margin-top: 0.55rem; color: var(--ink); font-family: Georgia, serif; font-size: 2rem; line-height: 1.05; }
    .evidence-note { margin-top: 0.4rem; color: var(--muted); font-size: 0.78rem; line-height: 1.4; }
    .feature-card {
        min-height: 13.5rem;
        padding: 1.65rem;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: white;
    }
    .feature-number { color: #47745f; font-size: 0.7rem; font-weight: 800; letter-spacing: 0.12em; }
    .feature-card h3 { margin: 1.2rem 0 0.75rem; font-size: 1.55rem; }
    .feature-card p { font-size: 0.92rem; line-height: 1.65; }
    .page-intro { max-width: 820px; padding: 4.25rem 0 2.5rem; }
    .page-intro h1 { margin: 0; font-size: clamp(2.7rem, 5vw, 4.7rem); line-height: 1; }
    .page-intro p { max-width: 680px; margin-top: 1.25rem; font-size: 1.05rem; line-height: 1.7; }
    [data-testid="stForm"] {
        padding: 1.5rem;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: white;
    }
    [data-testid="stForm"] h3 { margin-top: 0.4rem; font-family: inherit; font-size: 1.1rem; letter-spacing: 0; }
    [data-testid="stExpander"] { border-color: var(--line); background: #fbfcfa; }
    .prediction-card {
        margin: 1.6rem 0 1rem;
        padding: 2.4rem;
        border: 1px solid #b8cfc3;
        border-radius: 16px;
        background: var(--sage);
        text-align: left;
    }
    .prediction-label {
        color: #416655;
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }
    .prediction-value {
        margin-top: 0.45rem;
        color: var(--forest);
        font-size: clamp(2.75rem, 5vw, 4.75rem);
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -0.04em;
    }
    .prediction-meta {
        margin-top: 0.65rem;
        color: #4c695c;
        font-size: 0.95rem;
    }
    .site-footer {
        margin-top: 5rem;
        padding: 1.7rem 0 0.5rem;
        border-top: 1px solid var(--line);
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.6;
    }
    .site-footer strong { color: var(--ink); }
    .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"],
    [data-testid="stFormSubmitButton"] button {
        min-height: 3rem;
        border: 1px solid var(--forest);
        border-radius: 9px;
        background: var(--forest);
        color: white;
        font-weight: 700;
    }
    .stButton > button[kind="primary"] p,
    .stDownloadButton > button[kind="primary"] p,
    [data-testid="stFormSubmitButton"] button p {
        color: white !important;
    }
    button[kind="primary"] * { color: white !important; }
    @media (max-width: 900px) {
        .st-key-home_hero { min-height: 34rem; padding: 1rem; background-position: 64% center; }
        .st-key-home_hero [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
        .st-key-home_hero [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {
            min-width: 100% !important;
            flex: 1 1 100% !important;
            padding: 0 1.25rem 1.5rem;
        }
        .st-key-home_hero [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child { display: none; }
    }
    @media (max-width: 760px) {
        .block-container { padding: 0.8rem 1rem 2rem; }
        .site-header { min-height: 2.8rem; }
        .hero { padding: 3rem 0 2.25rem; }
        .hero h1 { font-size: 2.8rem; }
        .evidence-strip { grid-template-columns: 1fr 1fr; }
        .evidence-item { border-bottom: 1px solid var(--line); }
        .page-intro { padding-top: 3rem; }
        [data-testid="stSegmentedControl"], [data-testid="stButtonGroup"] div[role="radiogroup"] {
            justify-content: flex-start;
            flex-wrap: wrap;
        }
        [data-testid="stSegmentedControl"] button,
        [data-testid="stButtonGroup"] button,
        button[role="radio"] {
            flex: 1 1 calc(50% - 0.25rem);
            padding: 0.4rem 0.48rem;
            font-size: 0.76rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Stop with an actionable message when local model artifacts are unavailable.
try:
    FEATURES, MODELS = load_artifacts()
    METRICS = load_metrics()
    PRICE_BANDS, APE_DISTRIBUTION, ERROR_BUCKETS = load_segment_analysis()
except Exception as exc:
    st.error(f"Failed to load model artifacts: {exc}")
    st.info("Confirm that models/ contains all five .joblib models and model_features.joblib, and that requirements.txt is installed.")
    st.stop()

# A compact top navigation makes the app read like a public product rather than
# an internal dashboard while preserving Streamlit's rerun-safe state model.
def navigate_to(page_name: str) -> None:
    """Update the navigation widget from a callback before widgets render."""
    st.session_state["page"] = page_name


if "page" not in st.session_state:
    st.session_state["page"] = "Home"


brand_col, nav_col = st.columns([1.05, 2.1], vertical_alignment="center")
with brand_col:
    st.markdown(
        """
        <div class="site-header">
            <div class="brand-mark">CHI</div>
            <div>
                <div class="brand-name">California Home Intelligence</div>
                <div class="brand-subtitle">Residential price review</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with nav_col:
    page = st.segmented_control(
        "Primary navigation",
        ["Home", "Estimate", "Model Insights", "Analyst Tools"],
        required=True,
        label_visibility="collapsed",
        key="page",
        width="stretch",
    )
st.markdown('<div class="top-rule"></div>', unsafe_allow_html=True)

# LightGBM is the primary estimator because it achieved the strongest overall
# held-out performance; the other models remain available for comparison.
primary_metrics = METRICS.loc[METRICS["Model"] == "LightGBM"].iloc[0]

if page == "Home":
    hero_data_url = load_hero_data_url()
    st.markdown(
        f'<style>.st-key-home_hero {{ background-image: url("{hero_data_url}"); }}</style>',
        unsafe_allow_html=True,
    )
    with st.container(key="home_hero"):
        hero_copy, hero_space = st.columns([1.08, 0.92], gap="large", vertical_alignment="center")
        with hero_copy:
            st.markdown(
                """
                <section class="hero">
                    <div class="eyebrow">California homes · evidence-led pricing</div>
                    <h1>See the value.<br>Know the risk.</h1>
                    <p class="hero-copy">
                        A practical decision-support tool for reviewing California home values,
                        comparing model estimates, and recognizing when professional judgment matters.
                    </p>
                </section>
                """,
                unsafe_allow_html=True,
            )
            st.button(
                "Start a property estimate",
                type="primary",
                key="home_cta",
                on_click=navigate_to,
                args=("Estimate",),
            )

    st.markdown(
        """
        <div class="evidence-strip">
            <div class="evidence-item">
                <div class="evidence-label">Held-out evidence</div>
                <div class="evidence-value">June 2026</div>
                <div class="evidence-note">Untouched test month used for model comparison.</div>
            </div>
            <div class="evidence-item">
                <div class="evidence-label">Primary model</div>
                <div class="evidence-value">LightGBM</div>
                <div class="evidence-note">Selected for the strongest overall test performance.</div>
            </div>
            <div class="evidence-item">
                <div class="evidence-label">Explained variance</div>
                <div class="evidence-value">0.906 R²</div>
                <div class="evidence-note">Approximately 91% of observed price variation.</div>
            </div>
            <div class="evidence-item">
                <div class="evidence-label">Typical percentage miss</div>
                <div class="evidence-value">8.3%</div>
                <div class="evidence-note">LightGBM median absolute percentage error.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <section class="section">
            <div class="eyebrow">What the product supports</div>
            <h2>One estimate, three better-informed decisions.</h2>
            <p class="hero-copy">The model is a review signal—not an automated appraisal. It is most useful when paired with local context and comparable-sale judgment.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    cards = [
        (c1, "01 · PRICE REVIEW", "Challenge a proposed price", "Compare the primary estimate with a listing or expected sale price and decide whether deeper comparable-sale review is justified."),
        (c2, "02 · CONSENSUS", "Compare five models", "See whether independent model families agree. A wide spread is a clear signal to interpret the estimate with additional caution."),
        (c3, "03 · RISK", "Route unusual properties", "Recognize the limits of an educational AVM and send luxury, highly unusual, or out-of-distribution homes to professional review."),
    ]
    for column, number, title, copy in cards:
        with column:
            st.markdown(
                f'<div class="feature-card"><div class="feature-number">{number}</div><h3>{title}</h3><p>{copy}</p></div>',
                unsafe_allow_html=True,
            )

elif page == "Estimate":
    st.markdown(
        """
        <div class="page-intro">
            <div class="eyebrow">Single-property review</div>
            <h1>A practical price check, grounded in property details.</h1>
            <p>Start with the essential facts. Additional location, HOA, construction, and amenity fields are available when you need a more complete estimate.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Derive the county selector from the saved schema to avoid a duplicated,
    # manually maintained list of one-hot categories.
    county_features = [f.removeprefix("CountyOrParish_") for f in FEATURES if f.startswith("CountyOrParish_")]
    default_county = county_features.index("Los Angeles") if "Los Angeles" in county_features else 0
    with st.form("prediction_form"):
        st.subheader("Essential property details")
        c1, c2, c3, c4 = st.columns(4)
        living_area = c1.number_input("Living area (sq ft)", 100.0, 50000.0, 1800.0, 50.0)
        lot_size = c2.number_input("Lot size (sq ft)", 100.0, 10000000.0, 7200.0, 100.0)
        bedrooms = c3.number_input("Bedrooms", 0, 30, 3)
        bathrooms = c4.number_input("Bathrooms", 0, 30, 2)
        c1, c2, c3, c4 = st.columns(4)
        year_built = c1.number_input("Year built", 1800, date.today().year + 2, 1976)
        county = c2.selectbox("County", county_features, default_county)
        garage = c3.number_input("Garage spaces", 0.0, 20.0, 2.0, 1.0)
        stories = c4.number_input("Stories", 0.0, 20.0, 1.0, 0.5)

        with st.expander("Additional details", expanded=False):
            st.caption("Use these fields when the information is available. The defaults represent a typical starting point for this prototype.")
            c1, c2, c3 = st.columns(3)
            latitude = c1.number_input("Latitude", 32.0, 42.5, 34.104689, format="%.6f", help="Geographic latitude of the property.")
            longitude = c2.number_input("Longitude", -125.0, -113.0, -118.075548, format="%.6f", help="Geographic longitude of the property.")
            close_date = c3.date_input("Valuation date", date(2026, 6, 15))
            c1, c2, c3, c4 = st.columns(4)
            parking = c1.number_input("Total parking", 0.0, 50.0, 2.0, 1.0)
            hoa_fee = c2.number_input("HOA fee", 0.0, 100000.0, 0.0, 25.0)
            hoa_frequency = c3.selectbox("HOA frequency", ["None", "Monthly", "Quarterly", "SemiAnnually", "Annually"])
            levels = c4.selectbox("Home levels", LEVEL_OPTIONS, index=1)
            main_bedrooms = st.number_input("Main-level bedrooms", 0.0, 30.0, float(min(bedrooms, 3)), 1.0)
            flooring = st.multiselect("Flooring", FLOORING_OPTIONS, default=["Unknown"])
            st.markdown("**Features and amenities**")
            amenity_cols = st.columns(4)
            labels = ["View", "Waterfront", "Basement", "Private pool", "Attached garage", "Fireplace", "New construction"]
            amenity_values = [amenity_cols[index % 4].checkbox(label) for index, label in enumerate(labels)]
            st.markdown("**School district indicators**")
            d1, d2, d3 = st.columns(3)
            district_elementary = d1.checkbox("Elementary district")
            district_high = d2.checkbox("High-school district")
            district_unified = d3.checkbox("Unified district", value=True)
        submitted = st.form_submit_button("Estimate Home Price", type="primary", width="stretch")

    if submitted:
        if close_date.year < year_built:
            st.error("Year built cannot be later than the valuation date.")
        else:
            view, waterfront, basement, pool, attached, fireplace, new_construction = amenity_values
            values = {
                "LivingArea": living_area, "LotSizeSquareFeet": lot_size, "LotSizeArea": lot_size,
                "BedroomsTotal": bedrooms, "BathroomsTotalInteger": bathrooms,
                "YearBuilt": year_built, "GarageSpaces": garage, "ParkingTotal": parking,
                "Stories": stories, "MainLevelBedrooms": main_bedrooms,
                "CountyOrParish": county, "Latitude": latitude, "Longitude": longitude,
                "CloseDate": close_date, "AssociationFee": hoa_fee,
                "AssociationFeeFrequency": hoa_frequency, "Levels": levels, "Flooring": flooring,
                "ViewYN": view, "WaterfrontYN": waterfront, "BasementYN": basement,
                "PoolPrivateYN": pool, "AttachedGarageYN": attached,
                "FireplaceYN": fireplace, "NewConstructionYN": new_construction,
                "AmenityCount": sum(amenity_values),
                "DistrictType_Elementary": district_elementary,
                "DistrictType_High": district_high, "DistrictType_Unified": district_unified,
            }
            # Encode the form once and send the identical row to all five models.
            feature_row = make_feature_row(values, FEATURES)
            predictions = {
                name: float(model.predict(feature_row)[0])
                for name, model in MODELS.items()
            }
            prediction = predictions["LightGBM"]
            # This MAPE-based range is a descriptive guide, not a calibrated
            # prediction or confidence interval.
            mape = float(primary_metrics["MAPE (%)"]) / 100
            lo, hi = max(0, prediction * (1 - mape)), prediction * (1 + mape)
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="prediction-label">Estimated Sale Price</div>
                    <div class="prediction-value">${prediction:,.0f}</div>
                    <div class="prediction-meta">LightGBM primary estimate · Reference range ${lo:,.0f}–${hi:,.0f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("The reference range uses test-set MAPE as a simple guide; it is not a statistical confidence interval.")

            comparison = pd.DataFrame({
                "Model": list(predictions),
                "Estimated Price": list(predictions.values()),
            })
            comparison["Difference vs. LightGBM"] = (
                (comparison["Estimated Price"] - prediction) / prediction
            )
            median_prediction = float(comparison["Estimated Price"].median())
            spread = (
                comparison["Estimated Price"].max() - comparison["Estimated Price"].min()
            ) / median_prediction
            comparison["Estimated Price"] = comparison["Estimated Price"].map(lambda value: f"${value:,.0f}")
            comparison["Difference vs. LightGBM"] = comparison["Difference vs. LightGBM"].map(
                lambda value: "—" if abs(value) < 1e-12 else f"{value:+.1%}"
            )
            r1, r2, r3 = st.columns(3)
            r1.metric("Reference low", f"${lo:,.0f}")
            r2.metric("Reference high", f"${hi:,.0f}")
            r3.metric("Model agreement", "Lower" if spread > 0.25 else "Higher", help="Based on the spread across all five model estimates.")
            st.markdown("#### How to read this result")
            if spread > 0.25:
                st.warning("The five models disagree substantially for this property. Treat the estimate as a review signal and prioritize comparable-sale or professional appraisal evidence.")
            else:
                st.info("The five models show relatively close agreement for this property. The estimate is still a decision-support signal, not a professional appraisal.")
            with st.expander("Technical model comparison", expanded=False):
                st.dataframe(comparison, hide_index=True, width="stretch")
                st.caption(f"Median estimate across all models: ${median_prediction:,.0f}")
            with st.expander("View the 127 encoded model features"):
                st.dataframe(feature_row.T.rename(columns={0: "Value"}), width="stretch")

elif page == "Analyst Tools":
    st.markdown(
        """
        <div class="page-intro">
            <div class="eyebrow">Analyst workflow</div>
            <h1>Run the saved models across an encoded portfolio.</h1>
            <p>This technical workflow expects the same 127-feature schema used during training. It is intended for analysts, not general property CSV uploads.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Step 1 · Prepare the encoded file")
    st.write("Download the schema or a one-row example. This workflow does not accept a standard listing export: every model feature must already be encoded and scaled.")
    template = pd.DataFrame(columns=FEATURES)
    analyst_counties = [f.removeprefix("CountyOrParish_") for f in FEATURES if f.startswith("CountyOrParish_")]
    example_values = {
        "LivingArea": 1800.0, "LotSizeSquareFeet": 7200.0, "LotSizeArea": 7200.0,
        "BedroomsTotal": 3, "BathroomsTotalInteger": 2, "YearBuilt": 1976,
        "GarageSpaces": 2.0, "ParkingTotal": 2.0, "Stories": 1.0,
        "MainLevelBedrooms": 3.0, "CountyOrParish": "Los Angeles" if "Los Angeles" in analyst_counties else analyst_counties[0],
        "Latitude": 34.104689, "Longitude": -118.075548, "CloseDate": date(2026, 6, 15),
        "AssociationFee": 0.0, "AssociationFeeFrequency": "None", "Levels": "One",
        "Flooring": ["Unknown"], "ViewYN": False, "WaterfrontYN": False,
        "BasementYN": False, "PoolPrivateYN": False, "AttachedGarageYN": True,
        "FireplaceYN": False, "NewConstructionYN": False, "AmenityCount": 1,
        "DistrictType_Elementary": False, "DistrictType_High": False,
        "DistrictType_Unified": True,
    }
    example = make_feature_row(example_values, FEATURES)
    d1, d2 = st.columns(2)
    d1.download_button("Download empty schema", template.to_csv(index=False), "prediction_template.csv", "text/csv", width="stretch")
    d2.download_button("Download encoded example", example.to_csv(index=False), "encoded_example.csv", "text/csv", width="stretch")

    st.markdown("### Step 2 · Validate and run predictions")
    upload = st.file_uploader("Upload the encoded CSV", type="csv", help="The file must include all 127 saved model features. Identifier and ClosePrice columns are allowed.")
    if upload is not None:
        try:
            uploaded = pd.read_csv(upload)
            encoded = validate_encoded_frame(uploaded, FEATURES)
            extra_columns = [column for column in uploaded.columns if column not in FEATURES]
            v1, v2, v3 = st.columns(3)
            v1.metric("Rows ready", f"{len(encoded):,}")
            v2.metric("Required features", f"{len(FEATURES)} / {len(FEATURES)}")
            v3.metric("Extra columns retained", f"{len(extra_columns)}")
            st.success("Schema validation passed. All required features are numeric and ready for prediction.")
            output = uploaded.copy()
            for name, model in MODELS.items():
                output[f"PredictedPrice_{name.replace(' ', '')}"] = model.predict(encoded)
            st.markdown("### Step 3 · Review and export")
            st.success(f"Completed five-model predictions for {len(output):,} records.")
            st.dataframe(output.head(100), width="stretch")
            st.download_button("Download Prediction Results", output.to_csv(index=False), "home_price_predictions.csv", "text/csv", type="primary")
        except Exception as exc:
            st.error(f"Unable to generate predictions: {exc}")

elif page == "Model Insights":
    st.markdown(
        """
        <div class="page-intro">
            <div class="eyebrow">Trust and method</div>
            <h1>Performance evidence, with the limits kept visible.</h1>
            <p>All five models were compared on the same untouched June 2026 test month.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Present the saved Week 8 test metrics without recomputing model results.
    st.info("LightGBM is the primary model because it explains approximately 91% of observed price variation and produced the lowest overall dollar error on the held-out test month.")
    st.subheader("Overall Model Comparison")
    st.write("All models were evaluated on the same untouched June 2026 test month.")

    best_r2 = METRICS.loc[METRICS["R²"].idxmax()]
    best_mae = METRICS.loc[METRICS["MAE"].idxmin()]
    best_mdape = METRICS.loc[METRICS["MdAPE (%)"].idxmin()]
    k1, k2, k3 = st.columns(3)
    k1.metric("Best R²", f"{best_r2['R²']:.3f}", best_r2["Model"])
    k2.metric("Lowest MAE", f"${best_mae['MAE']:,.0f}", best_mae["Model"])
    k3.metric("Lowest Median APE", f"{best_mdape['MdAPE (%)']:.1f}%", best_mdape["Model"])

    display = METRICS.copy()
    display["MAE"] = display["MAE"].map(lambda x: f"${x:,.0f}")
    display["RMSE"] = display["RMSE"].map(lambda x: f"${x:,.0f}")
    display["R²"] = display["R²"].map(lambda x: f"{x:.3f}")
    display["MAPE (%)"] = display["MAPE (%)"].map(lambda x: f"{x:.1f}%")
    display["MdAPE (%)"] = display["MdAPE (%)"].map(lambda x: f"{x:.1f}%")
    st.dataframe(display, hide_index=True, width="stretch")

    st.markdown("#### Explained Variance")
    st.caption("Higher R² is better.")
    r2_chart = METRICS.set_index("Model")[["R²"]].sort_values("R²")
    st.bar_chart(r2_chart, horizontal=True, color="#3b8c6e")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Dollar Error")
        st.caption("Lower values are better. MAE shows typical dollar error; RMSE penalizes large misses more heavily.")
        dollar_errors = METRICS.set_index("Model")[["MAE", "RMSE"]]
        st.bar_chart(dollar_errors, color=["#3b8c6e", "#9bbbad"])
    with right:
        st.markdown("#### Percentage Error")
        st.caption("Lower values are better. MdAPE is less sensitive to extreme properties than MAPE.")
        percentage_errors = METRICS.set_index("Model")[["MAPE (%)", "MdAPE (%)"]]
        st.bar_chart(percentage_errors, color=["#466b9c", "#9aafd0"])

    st.markdown("---")
    st.subheader("Error Distribution")
    st.write(
        "A single average can hide the shape of model error. These views separate the typical property "
        "from the upper tail of harder-to-predict homes."
    )

    lightgbm_distribution = APE_DISTRIBUTION.loc[
        APE_DISTRIBUTION["Model"] == "LightGBM"
    ].iloc[0]
    within_ten = float(ERROR_BUCKETS.loc[
        ERROR_BUCKETS["Error Bucket"].isin(["Under 5%", "5–10%"]), "Share (%)"
    ].sum())
    within_twenty = float(ERROR_BUCKETS.loc[
        ERROR_BUCKETS["Error Bucket"].isin(["Under 5%", "5–10%", "10–20%"]), "Share (%)"
    ].sum())
    e1, e2, e3 = st.columns(3)
    e1.metric("Within 10%", f"{within_ten:.1f}%", "of test homes")
    e2.metric("Within 20%", f"{within_twenty:.1f}%", "of test homes")
    e3.metric("LightGBM P95 error", f"{lightgbm_distribution['P95 (%)']:.1f}%", "lowest of five models")

    distribution_left, distribution_right = st.columns([1.05, 1])
    with distribution_left:
        st.markdown("#### LightGBM Error Buckets")
        st.caption("Share of June 2026 homes by absolute percentage error.")
        error_bucket_chart = ERROR_BUCKETS.set_index("Error Bucket")[["Share (%)"]]
        st.bar_chart(error_bucket_chart, color="#3b8c6e")
    with distribution_right:
        st.markdown("#### Typical vs. Tail Error")
        st.caption("P90 and P95 show the error level below which 90% and 95% of homes fall.")
        tail_chart = APE_DISTRIBUTION.set_index("Model")[["Median (%)", "P90 (%)", "P95 (%)"]]
        st.bar_chart(tail_chart, color=["#9bbbad", "#6688a8", "#264f73"])

    st.info(
        "LightGBM keeps 83.3% of test predictions within 20% of the actual close price and has the "
        "lowest P90 and P95 error. Random Forest still has the slightly lower median error, so the "
        "primary-model choice reflects stronger overall fit and better control of severe misses—not a win on every metric."
    )

    st.markdown("---")
    st.subheader("Performance by Price Band")
    st.write(
        "To test how well each model distinguishes lower- and higher-value homes, the June 2026 test "
        "set is divided into five equal-frequency bands using actual close price."
    )

    band_order = ["Q1 - Lowest", "Q2 - Low", "Q3 - Middle", "Q4 - High", "Q5 - Highest"]
    nonlinear_models = ["Decision Tree", "Random Forest", "XGBoost", "LightGBM"]
    band_mdape = (
        PRICE_BANDS.loc[PRICE_BANDS["Model"].isin(nonlinear_models)]
        .pivot(index="Price Band", columns="Model", values="MdAPE (%)")
        .reindex(band_order)[nonlinear_models]
    )
    st.markdown("#### Median Percentage Error by Actual Price Quintile")
    st.caption("Lower is better. Linear Regression is omitted here so its much larger error does not flatten the differences among the four non-linear models.")
    st.line_chart(
        band_mdape,
        color=["#b89a68", "#6e927f", "#5379a1", "#123d2f"],
    )

    lightgbm_bands = (
        PRICE_BANDS.loc[PRICE_BANDS["Model"] == "LightGBM"]
        .set_index("Price Band")
        .reindex(band_order)
    )
    band_left, band_right = st.columns([1.05, 1])
    with band_left:
        st.markdown("#### LightGBM Median Residual")
        st.caption("Actual minus predicted. Positive values mean underprediction; negative values mean overprediction.")
        st.bar_chart(lightgbm_bands[["Median Residual ($)"]], color="#466b9c")
    with band_right:
        st.markdown("#### Test-Month Band Boundaries")
        band_table = lightgbm_bands.reset_index()[
            ["Price Band", "Count", "Min Actual", "Median Actual", "Max Actual"]
        ].copy()
        for column in ["Min Actual", "Median Actual", "Max Actual"]:
            band_table[column] = band_table[column].map(lambda value: f"${value:,.0f}")
        band_table = band_table.rename(columns={
            "Count": "Homes",
            "Min Actual": "From",
            "Median Actual": "Median",
            "Max Actual": "To",
        })
        st.dataframe(band_table, hide_index=True, width="stretch")

    takeaway_1, takeaway_2, takeaway_3 = st.columns(3)
    with takeaway_1:
        st.markdown("#### Lower price bands")
        st.write("Random Forest records the lowest median percentage error in Q1–Q3, with its best result in Q2 at 5.7%.")
    with takeaway_2:
        st.markdown("#### Upper-middle band")
        st.write("XGBoost narrowly leads Q4 at 8.7% MdAPE; LightGBM follows closely at 8.9%.")
    with takeaway_3:
        st.markdown("#### Highest price band")
        st.write("LightGBM leads Q5 at 10.9% MdAPE, but its positive USD 106,676 median residual shows systematic underprediction.")

    st.warning(
        "Price-band boundaries are specific to this June 2026 test month, not permanent business thresholds. "
        "The USD 1.65M–8.20M highest band remains the clearest review zone: use comparable-sale evidence or a professional appraisal before relying on an automated estimate."
    )

    st.markdown("#### Accuracy–Error Trade-Off")
    st.caption("Models closest to the upper-left combine higher explained variance with lower absolute error.")
    tradeoff = METRICS.rename(columns={"R²": "Test R²", "MAE": "MAE ($)"}).copy()
    st.scatter_chart(
        tradeoff,
        x="MAE ($)",
        y="Test R²",
        color="Model",
        size="RMSE",
    )

    st.info("LightGBM leads on R², MAE, MAPE, and RMSE. Random Forest has the lowest MdAPE, indicating slightly better typical percentage error.")

st.markdown(
    """
    <footer class="site-footer">
        <strong>California Home Intelligence</strong><br>
        Educational AVM prototype built by Jasper Fan-Chiang · USC Applied Data Science · IDX Exchange internship.<br>
        Estimates are decision-support signals, not professional appraisals or lending or investment advice.
    </footer>
    """,
    unsafe_allow_html=True,
)
