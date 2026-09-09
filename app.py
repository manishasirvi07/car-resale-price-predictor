import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Car Resale Price Predictor",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Car Resale Price Predictor")

st.write(
    "Enter the details of a used car to estimate its resale price."
)


# ==========================================
# LOAD DATASET
# ==========================================

@st.cache_data
def load_data():

    df = pd.read_csv("cardekho_dataset.csv")

    return df


df = load_data()


# ==========================================
# CLEAN COLUMN NAMES
# ==========================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)


# ==========================================
# TARGET
# ==========================================

target = "selling_price"


# Remove missing target values
df = df.dropna(subset=[target])


# ==========================================
# FEATURES
# ==========================================

features = [
    "brand",
    "model",
    "vehicle_age",
    "km_driven",
    "seller_type",
    "fuel_type",
    "transmission_type",
    "mileage",
    "engine",
    "max_power",
    "seats"
]


# Keep only columns available in dataset
features = [
    column
    for column in features
    if column in df.columns
]


X = df[features]

y = df[target]


# ==========================================
# CATEGORICAL & NUMERICAL FEATURES
# ==========================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


# ==========================================
# PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_features
        ),

        (
            "numerical",

            "passthrough",

            numerical_features
        )
    ]
)


# ==========================================
# MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)


# ==========================================
# TRAIN MODEL
# ==========================================

with st.spinner("Training model..."):

    pipeline.fit(
        X_train,
        y_train
    )


# ==========================================
# MODEL EVALUATION
# ==========================================

y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)


# ==========================================
# CAR DETAILS
# ==========================================

st.divider()

st.subheader("🚘 Enter Car Details")


inputs = {}


for feature in features:

    # Categorical input
    if feature in categorical_features:

        values = sorted(
            df[feature]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        inputs[feature] = st.selectbox(
            feature.replace(
                "_",
                " "
            ).title(),

            values
        )

    # Numerical input
    else:

        minimum = float(
            df[feature].min()
        )

        maximum = float(
            df[feature].max()
        )

        default = float(
            df[feature].median()
        )

        inputs[feature] = st.number_input(

            feature.replace(
                "_",
                " "
            ).title(),

            min_value=minimum,

            max_value=maximum,

            value=default
        )


# ==========================================
# PREDICT PRICE
# ==========================================

if st.button(
    "🚀 Predict Resale Price",
    use_container_width=True
):

    input_df = pd.DataFrame(
        [inputs]
    )


    # Prediction
    prediction = pipeline.predict(
        input_df
    )[0]


    # Prevent negative prediction
    prediction = max(
        0,
        prediction
    )


    # ======================================
    # RESULT
    # ======================================

    st.divider()

    st.subheader(
        "💰 Estimated Resale Price"
    )


    st.success(
        f"₹{prediction:,.0f}"
    )


    st.markdown(
        f"### Approximately ₹{prediction / 100000:.2f} Lakhs"
    )


    # ======================================
    # MODEL ERROR
    # ======================================

    st.caption(
        f"Typical prediction error (MAE): "
        f"₹{mae:,.0f}"
    )


    # ======================================
    # PRICE CATEGORY
    # ======================================

    if prediction >= 1000000:

        st.info(
            "💎 Premium Used Car"
        )

    elif prediction >= 500000:

        st.info(
            "🚗 Mid-Range Used Car"
        )

    else:

        st.info(
            "🚙 Budget Used Car"
        )