from __future__ import annotations

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
st.set_page_config(page_title="California Home Price Estimator", page_icon="🏠", layout="wide")

# Apply the project-specific prediction card and sidebar theme.
st.markdown(
    """
    <style>
    .prediction-card {
        margin: 1.25rem 0 0.75rem;
        padding: 2rem 2.25rem;
        border: 1px solid rgba(38, 102, 79, 0.25);
        border-radius: 18px;
        background: linear-gradient(135deg, #f4fbf7 0%, #e8f5ee 100%);
        box-shadow: 0 10px 28px rgba(31, 82, 63, 0.10);
        text-align: center;
    }
    .prediction-label {
        margin-bottom: 0.35rem;
        color: #4b6d60;
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }
    .prediction-value {
        color: #123d2f;
        font-size: clamp(2.75rem, 5vw, 4.75rem);
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -0.04em;
    }
    .prediction-meta {
        margin-top: 0.65rem;
        color: #577468;
        font-size: 0.95rem;
    }
    section[data-testid="stSidebar"] {
        background: #123d2f;
        border-right: 0;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1.75rem;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #ffffff;
    }
    section[data-testid="stSidebar"] button[kind="secondary"] {
        min-height: 2.8rem;
        padding: 0.65rem 0.85rem;
        border: 0;
        border-left: 3px solid transparent;
        border-radius: 8px;
        background: transparent;
        color: #d9e7e1;
        font-weight: 500;
        text-align: left;
        justify-content: flex-start;
        transition: background-color 120ms ease, color 120ms ease;
    }
    section[data-testid="stSidebar"] button[kind="secondary"]:hover {
        border-color: transparent;
        background: rgba(255, 255, 255, 0.08);
        color: #ffffff;
    }
    section[data-testid="stSidebar"] button[kind="primary"] {
        min-height: 2.8rem;
        padding: 0.65rem 0.85rem;
        border: 0;
        border-left-color: #88c7aa;
        border-left-style: solid;
        border-left-width: 3px;
        border-radius: 8px;
        background: #ddefe6;
        color: #123d2f;
        font-weight: 700;
        text-align: left;
        justify-content: flex-start;
    }
    section[data-testid="stSidebar"] button[kind="primary"]:hover {
        border-color: #88c7aa;
        background: #e8f5ee;
        color: #123d2f;
    }
    .sidebar-brand {
        margin: 0 0 2.1rem;
    }
    .sidebar-brand-mark {
        margin-bottom: 0.8rem;
        font-size: 2rem;
        line-height: 1;
    }
    .sidebar-brand-name {
        color: #ffffff;
        font-size: 1.03rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        line-height: 1.35;
    }
    .sidebar-brand-subtitle {
        margin-top: 0.2rem;
        color: #a9c3b8;
        font-size: 0.78rem;
        letter-spacing: 0.05em;
    }
    .sidebar-section-label {
        margin: 1.15rem 0 0.45rem 0.2rem;
        color: #88a99c;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.13em;
    }
    .sidebar-footer {
        margin-top: 2.4rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.14);
        color: #88a99c;
        font-size: 0.72rem;
        line-height: 1.5;
    }
    .sidebar-author {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.14);
    }
    .sidebar-author-avatar {
        display: flex;
        flex: 0 0 2rem;
        align-items: center;
        justify-content: center;
        width: 2rem;
        height: 2rem;
        border-radius: 50%;
        background: #ddefe6;
        color: #123d2f;
        font-size: 0.72rem;
        font-weight: 800;
    }
    .sidebar-author-label {
        color: #88a99c;
        font-size: 0.62rem;
        font-weight: 700;
        letter-spacing: 0.11em;
    }
    .sidebar-author-name {
        margin-top: 0.08rem;
        color: #ffffff;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .sidebar-author-detail {
        margin-top: 0.3rem;
        color: #a9c3b8;
        font-size: 0.67rem;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🏠 California Home Price Estimator")
st.caption("A home-price estimation tool trained on historical CRMLS sales data | California residential properties only")

# Stop with an actionable message when local model artifacts are unavailable.
try:
    FEATURES, MODELS = load_artifacts()
    METRICS = load_metrics()
except Exception as exc:
    st.error(f"Failed to load model artifacts: {exc}")
    st.info("Confirm that models/ contains all five .joblib models and model_features.joblib, and that requirements.txt is installed.")
    st.stop()

# Sidebar buttons store the selected view in Streamlit session state.
with st.sidebar:
    if "page" not in st.session_state:
        st.session_state.page = "Single Property"

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-mark">⌂</div>
            <div class="sidebar-brand-name">HOME PRICE<br>ESTIMATOR</div>
            <div class="sidebar-brand-subtitle">CALIFORNIA AVM</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-section-label">PREDICTION</div>', unsafe_allow_html=True)

    if st.button(
        "⌂   Single Property", key="nav_single", width="stretch",
        type="primary" if st.session_state.page == "Single Property" else "secondary",
    ):
        st.session_state.page = "Single Property"
        st.rerun()

    if st.button(
        "▤   Batch Prediction", key="nav_batch", width="stretch",
        type="primary" if st.session_state.page == "Batch Prediction" else "secondary",
    ):
        st.session_state.page = "Batch Prediction"
        st.rerun()

    st.markdown('<div class="sidebar-section-label">ANALYSIS</div>', unsafe_allow_html=True)
    if st.button(
        "▥   Model Performance", key="nav_performance", width="stretch",
        type="primary" if st.session_state.page == "Model Performance" else "secondary",
    ):
        st.session_state.page = "Model Performance"
        st.rerun()

    st.markdown(
        """
        <div class="sidebar-footer">CRMLS · CALIFORNIA<br>Educational AVM prototype</div>
        <div class="sidebar-author">
            <div class="sidebar-author-avatar">JF</div>
            <div>
                <div class="sidebar-author-label">AUTHOR</div>
                <div class="sidebar-author-name">Jasper Fan-Chiang</div>
                <div class="sidebar-author-detail">
                    M.S. in Applied Data Science<br>
                    University of Southern California<br><br>
                    IDX Exchange — Data Science Internship
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

page = st.session_state.page

# LightGBM is the primary estimator because it achieved the strongest overall
# held-out performance; the other models remain available for comparison.
primary_metrics = METRICS.loc[METRICS["Model"] == "LightGBM"].iloc[0]

if page == "Single Property":
    # Derive the county selector from the saved schema to avoid a duplicated,
    # manually maintained list of one-hot categories.
    county_features = [f.removeprefix("CountyOrParish_") for f in FEATURES if f.startswith("CountyOrParish_")]
    default_county = county_features.index("Los Angeles") if "Los Angeles" in county_features else 0
    with st.form("prediction_form"):
        st.subheader("Property Details")
        c1, c2, c3, c4 = st.columns(4)
        living_area = c1.number_input("Living area (sq ft)", 100.0, 50000.0, 1800.0, 50.0)
        lot_size = c2.number_input("Lot size (sq ft)", 100.0, 10000000.0, 7200.0, 100.0)
        bedrooms = c3.number_input("Bedrooms", 0, 30, 3)
        bathrooms = c4.number_input("Bathrooms", 0, 30, 2)
        c1, c2, c3, c4 = st.columns(4)
        year_built = c1.number_input("Year built", 1800, date.today().year + 2, 1976)
        garage = c2.number_input("Garage spaces", 0.0, 20.0, 2.0, 1.0)
        parking = c3.number_input("Total parking", 0.0, 50.0, 2.0, 1.0)
        stories = c4.number_input("Stories", 0.0, 20.0, 1.0, 0.5)

        st.subheader("Location and Transaction Details")
        c1, c2, c3, c4 = st.columns(4)
        county = c1.selectbox("County", county_features, default_county)
        latitude = c2.number_input("Latitude", 32.0, 42.5, 34.104689, format="%.6f")
        longitude = c3.number_input("Longitude", -125.0, -113.0, -118.075548, format="%.6f")
        close_date = c4.date_input("Valuation / close date", date(2026, 6, 15))
        c1, c2, c3, c4 = st.columns(4)
        hoa_fee = c1.number_input("HOA fee", 0.0, 100000.0, 0.0, 25.0)
        hoa_frequency = c2.selectbox("HOA frequency", ["None", "Monthly", "Quarterly", "SemiAnnually", "Annually"])
        levels = c3.selectbox("Levels", LEVEL_OPTIONS, index=1)
        main_bedrooms = c4.number_input("Main-level bedrooms", 0.0, 30.0, float(min(bedrooms, 3)), 1.0)

        st.subheader("Features and Amenities")
        flooring = st.multiselect("Flooring", FLOORING_OPTIONS, default=["Unknown"])
        amenity_cols = st.columns(7)
        labels = ["View", "Waterfront", "Basement", "Private pool", "Attached garage", "Fireplace", "New construction"]
        amenity_values = [col.checkbox(label) for col, label in zip(amenity_cols, labels)]
        d1, d2, d3 = st.columns(3)
        district_elementary = d1.checkbox("Elementary district match")
        district_high = d2.checkbox("High-school district match")
        district_unified = d3.checkbox("Unified district match", value=True)
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
            with st.expander("Compare All Model Estimates", expanded=True):
                st.dataframe(comparison, hide_index=True, width="stretch")
                st.caption(f"Median estimate across all models: ${median_prediction:,.0f}")
                if spread > 0.25:
                    st.warning("Model predictions vary substantially for this property. Treat the estimate with additional caution.")
            with st.expander("View the 127 features submitted to the model"):
                st.dataframe(feature_row.T.rename(columns={0: "Value"}), width="stretch")

elif page == "Batch Prediction":
    # Batch mode intentionally accepts already encoded and scaled features; it
    # does not attempt to reproduce raw-data preprocessing for arbitrary files.
    st.subheader("Upload an Encoded CSV")
    st.write("The CSV must contain all 127 features defined in `model_features.joblib`. It may also contain `ClosePrice` or identifier columns.")
    template = pd.DataFrame(columns=FEATURES)
    st.download_button("Download Feature Template", template.to_csv(index=False), "prediction_template.csv", "text/csv")
    upload = st.file_uploader("Choose a CSV file", type="csv")
    if upload is not None:
        try:
            uploaded = pd.read_csv(upload)
            encoded = validate_encoded_frame(uploaded, FEATURES)
            output = uploaded.copy()
            for name, model in MODELS.items():
                output[f"PredictedPrice_{name.replace(' ', '')}"] = model.predict(encoded)
            st.success(f"Completed five-model predictions for {len(output):,} records.")
            st.dataframe(output.head(100), width="stretch")
            st.download_button("Download Prediction Results", output.to_csv(index=False), "home_price_predictions.csv", "text/csv", type="primary")
        except Exception as exc:
            st.error(f"Unable to generate predictions: {exc}")

elif page == "Model Performance":
    # Present the saved Week 8 test metrics without recomputing model results.
    st.subheader("Held-Out Test-Month Performance")
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

st.divider()
st.caption("Educational AVM prototype — Do not treat this estimate as an appraisal or as lending or investment advice.")
