
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Project configuration
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

DEMO_DATA_FILE = PROJECT_DIR / "data" / "housing_sample.csv"
TRAINING_DATA_FILE = (
    PROJECT_DIR / "data" / "processed" / "housing_cleaned.csv"
)
MODEL_FILE = PROJECT_DIR / "models" / "house_price_sensitivity_a.joblib"

st.set_page_config(
    page_title="Nigerian House Price Prediction",
    page_icon="🏠",
    layout="wide",
)


# --------------------------------------------------
# Cached loaders: avoid repeatedly reading files
# --------------------------------------------------
@st.cache_data(show_spinner="Loading training dataset...")
def load_training_data(file_path, modified_time):
    return pd.read_csv(file_path)


@st.cache_data(show_spinner="Loading demonstration dataset...")
def load_demo_data(file_path, modified_time):
    return pd.read_csv(file_path)


@st.cache_resource
def load_saved_model(file_path, modified_time):
    return joblib.load(file_path)


@st.cache_data
def get_options(values):
    """Return sorted, unique, non-empty dropdown values."""
    return sorted({
        str(value).strip()
        for value in values
        if pd.notna(value) and str(value).strip()
    })


@st.cache_data
def get_towns_by_state(data):
    """Build a cached mapping of states to available towns."""
    locations = data[["state", "town"]].copy()

    locations["state"] = (
        locations["state"].fillna("").astype(str).str.strip()
    )
    locations["town"] = (
        locations["town"].fillna("").astype(str).str.strip()
    )

    locations = locations[
        (locations["state"] != "")
        & (locations["town"] != "")
    ].drop_duplicates()

    return {
        state_name: sorted(group["town"].unique().tolist())
        for state_name, group in locations.groupby("state", sort=True)
    }


# --------------------------------------------------
# Page heading
# --------------------------------------------------
st.title("Nigerian House Price Prediction")

st.caption(
    "Machine learning project for residential property "
    "price analysis in Nigeria."
)

st.info(
    "The dashboard uses demonstration data. The prediction form "
    "uses the saved experimental model trained on the processed "
    "housing dataset. Its output is a model estimate, not a "
    "verified market valuation."
)


# --------------------------------------------------
# Load model and training dataset
# --------------------------------------------------
bundle = None
training_df = None

if not MODEL_FILE.exists():
    st.error(f"Saved model not found: {MODEL_FILE}")

elif not TRAINING_DATA_FILE.exists():
    st.error(f"Training dataset not found: {TRAINING_DATA_FILE}")

else:
    try:
        training_df = load_training_data(
            str(TRAINING_DATA_FILE),
            TRAINING_DATA_FILE.stat().st_mtime,
        )

        bundle = load_saved_model(
            str(MODEL_FILE),
            MODEL_FILE.stat().st_mtime,
        )

        if not isinstance(bundle, dict) or "model" not in bundle:
            raise ValueError(
                "The saved model file does not contain the expected model bundle."
            )

        required_model_keys = [
            "feature_columns",
            "numeric_features",
            "categorical_features",
            "target_transform",
        ]

        missing_keys = [
            key for key in required_model_keys
            if key not in bundle
        ]

        if missing_keys:
            raise ValueError(
                "Model bundle is missing: "
                + ", ".join(missing_keys)
            )

        if bundle["target_transform"] != "log1p":
            raise ValueError(
                "This app expects a model trained with a log1p target."
            )

        missing_features = [
            column for column in bundle["feature_columns"]
            if column not in training_df.columns
        ]

        if missing_features:
            raise ValueError(
                "Training data is missing model features: "
                + ", ".join(missing_features)
            )

        required_dropdown_columns = {
            "state", "town", "property_type"
        }

        missing_dropdown_columns = (
            required_dropdown_columns - set(training_df.columns)
        )

        if missing_dropdown_columns:
            raise ValueError(
                "Training data is missing dropdown columns: "
                + ", ".join(sorted(missing_dropdown_columns))
            )

    except Exception as error:
        st.error(f"Could not load the prediction model or data: {error}")
        bundle = None
        training_df = None


# --------------------------------------------------
# Prediction form
# --------------------------------------------------
st.header("Predict a House Price")

if bundle is not None and training_df is not None:

    numeric_features = bundle["numeric_features"]
    categorical_features = bundle["categorical_features"]
    feature_columns = bundle["feature_columns"]

    supported_categorical = {
        "state", "town", "property_type"
    }

    supported_numeric = {
        "bedrooms", "bathrooms", "toilets", "parking_space"
    }

    unsupported = (
        set(categorical_features) - supported_categorical
    ) | (
        set(numeric_features) - supported_numeric
    )

    if unsupported:
        st.error(
            "This form does not yet support these model features: "
            + ", ".join(sorted(unsupported))
        )

    else:
        # These choices are cached to avoid rebuilding them on every rerun.
        states = get_options(tuple(training_df["state"].tolist()))

        property_types = get_options(
            tuple(training_df["property_type"].tolist())
        )

        towns_by_state = get_towns_by_state(
            training_df[["state", "town"]]
        )

        if not states:
            st.error(
                "No state values were found in the training dataset."
            )

        elif not property_types:
            st.error(
                "No property types were found in the training dataset."
            )

        else:
            default_state = (
                states.index("Lagos") if "Lagos" in states else 0
            )

            # Keep location selectors OUTSIDE the form.
            # This lets Streamlit update the town dropdown immediately.
            st.subheader("Property Location")

            location_left, location_right = st.columns(2)

            with location_left:
                state = st.selectbox(
                    "State / FCT",
                    options=states,
                    index=default_state,
                    key="prediction_state",
                )

            towns = towns_by_state.get(state, [])

            # Clear a previous town selection when the state changes.
            # The state-specific key gives each state's dropdown its own value.
            town_widget_key = f"prediction_town_{state}"

            if not towns:
                towns = get_options(tuple(training_df["town"].tolist()))

            default_town = (
                towns.index("Lekki") if "Lekki" in towns else 0
            )

            with location_right:
                town = st.selectbox(
                    "Town",
                    options=towns,
                    index=default_town if towns else None,
                    disabled=not towns,
                    key=town_widget_key,
                    help="Town options are based on the selected state.",
                )

            default_type = (
                property_types.index("Detached Duplex")
                if "Detached Duplex" in property_types
                else 0
            )

            # Numeric inputs and property type remain in the form.
            with st.form("house_prediction_form"):

                st.subheader("Enter the Property Details")

                property_type = st.selectbox(
                    "Property type",
                    options=property_types,
                    index=default_type,
                )

                left, right = st.columns(2)

                with left:
                    bedrooms = st.number_input(
                        "Bedrooms",
                        min_value=1,
                        max_value=9,
                        value=4,
                        step=1,
                    )

                    bathrooms = st.number_input(
                        "Bathrooms",
                        min_value=1,
                        max_value=9,
                        value=4,
                        step=1,
                    )

                with right:
                    toilets = st.number_input(
                        "Toilets",
                        min_value=1,
                        max_value=9,
                        value=4,
                        step=1,
                    )

                    parking_space = st.number_input(
                        "Parking spaces",
                        min_value=1,
                        max_value=9,
                        value=2,
                        step=1,
                    )

                submitted = st.form_submit_button(
                    "Predict House Price",
                    type="primary",
                    disabled=not towns or not property_types,
                )

            # --------------------------------------------------
            # Make prediction
            # --------------------------------------------------
            if submitted:
                input_values = {
                    "bedrooms": int(bedrooms),
                    "bathrooms": int(bathrooms),
                    "toilets": int(toilets),
                    "parking_space": int(parking_space),
                    "state": state,
                    "town": town,
                    "property_type": property_type,
                }

                try:
                    input_df = pd.DataFrame(
                        [{
                            column: input_values[column]
                            for column in feature_columns
                        }],
                        columns=feature_columns,
                    )

                    predicted_log_price = float(
                        np.asarray(
                            bundle["model"].predict(input_df)
                        ).ravel()[0]
                    )

                    # The saved model predicts log(1 + price).
                    predicted_price = float(
                        np.expm1(predicted_log_price)
                    )

                    if not np.isfinite(predicted_price):
                        raise ValueError(
                            "The model returned a non-finite price."
                        )

                    predicted_price = max(0.0, predicted_price)

                    st.divider()
                    st.subheader("Model Prediction")

                    st.metric(
                        "Estimated house price",
                        f"₦{predicted_price:,.2f}",
                    )

                    st.write("Details submitted")

                    st.dataframe(
                        input_df,
                        width="stretch",
                        hide_index=True,
                    )

                    st.caption(
                        "Predictions are estimates and may differ "
                        "from actual market prices."
                    )

                except Exception as error:
                    st.error(f"Prediction failed: {error}")

else:
    st.warning(
        "The prediction form is unavailable until both the saved model "
        "and its matching processed training dataset can be loaded."
    )


# --------------------------------------------------
# Demonstration dataset dashboard
# --------------------------------------------------
st.divider()
st.header("Demonstration Dataset Dashboard")

if not DEMO_DATA_FILE.exists():
    st.warning(
        "The demonstration dataset was not found. "
        "The prediction form above uses a separate training dataset."
    )

else:
    try:
        df = load_demo_data(
            str(DEMO_DATA_FILE),
            DEMO_DATA_FILE.stat().st_mtime,
        )

        required_columns = [
            "property_id",
            "state",
            "city",
            "property_type",
            "bedrooms",
            "bathrooms",
            "land_size_sqm",
            "house_age_years",
            "condition",
            "price_naira",
        ]

        missing_columns = [
            column for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            st.error(
                "Demonstration dataset is missing: "
                + ", ".join(missing_columns)
            )

        elif df.empty:
            st.warning(
                "The demonstration dataset contains no records."
            )

        else:
            df["price_naira"] = pd.to_numeric(
                df["price_naira"],
                errors="coerce",
            )

            valid_prices = df["price_naira"].dropna()
            valid_prices = valid_prices[valid_prices > 0]

            st.subheader("Dataset Overview")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Total Records", f"{len(df):,}")
            col2.metric("Features", f"{len(df.columns) - 1:,}")
            col3.metric(
                "States Represented",
                f"{df['state'].nunique():,}",
            )

            if not valid_prices.empty:
                col4.metric(
                    "Average Demo Price",
                    f"₦{valid_prices.mean():,.0f}",
                )
            else:
                col4.metric(
                    "Average Demo Price",
                    "Unavailable",
                )

            left, right = st.columns(2)

            with left:
                st.subheader(
                    "Average Demonstration Price by State"
                )

                state_prices = (
                    df.groupby("state", as_index=False)["price_naira"]
                    .mean()
                    .dropna()
                    .sort_values(
                        "price_naira",
                        ascending=False,
                    )
                    .set_index("state")
                )

                if not state_prices.empty:
                    st.bar_chart(state_prices)
                else:
                    st.warning(
                        "No valid prices are available for the chart."
                    )

            with right:
                st.subheader("Bedroom Distribution")

                bedroom_counts = (
                    df["bedrooms"].value_counts().sort_index()
                )

                st.bar_chart(bedroom_counts)

            st.subheader("Explore Demonstration Records")

            demo_states = [
                "All states"
            ] + sorted(
                df["state"].dropna().astype(str).unique()
            )

            selected_demo_state = st.selectbox(
                "Filter demonstration data by state",
                options=demo_states,
                key="demo_state_filter",
            )

            if selected_demo_state != "All states":
                filtered_df = df[
                    df["state"].astype(str) == selected_demo_state
                ]
            else:
                filtered_df = df

            st.write(
                f"Records displayed: {len(filtered_df):,}"
            )

            st.dataframe(
                filtered_df,
                width="stretch",
                hide_index=True,
            )

            with st.expander(
                "View demonstration data quality checks"
            ):
                st.write("Missing values per column")

                st.dataframe(
                    df.isnull().sum().rename("Missing values"),
                    width="stretch",
                )

                st.write("Dataset data types")

                st.dataframe(
                    df.dtypes.astype(str).rename("Data type"),
                    width="stretch",
                )

                st.write(
                    f"Valid positive prices: {len(valid_prices):,}"
                )

                st.write(
                    "Missing or non-positive prices: "
                    f"{len(df) - len(valid_prices):,}"
                )

    except (OSError, pd.errors.ParserError) as error:
        st.error(
            f"Could not read the demonstration dataset: {error}"
        )
