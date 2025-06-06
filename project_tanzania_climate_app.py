# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

st.title("Climate Temperature & Rainfall Analysis (Tanzania)")

# File Upload
uploaded_file = st.file_uploader("Upload CSV File (Temperature & Rainfall)", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    data = pd.read_csv(uploaded_file, on_bad_lines='skip', engine='python')

    # Clean columns
    data.rename(columns=lambda x: x.strip().capitalize(), inplace=True)
    data.rename(columns={'Ann': 'Temperature'}, inplace=True)  # Temperature column

    # Attempt to detect rainfall column
    rainfall_col = next((col for col in data.columns if 'rain' in col.lower()), None)

    required_cols = ['Year', 'Temperature']
    if rainfall_col:
        required_cols.append(rainfall_col)
        data.rename(columns={rainfall_col: 'Rainfall'}, inplace=True)

    if all(col in data.columns for col in required_cols):
        data = data[required_cols].copy()

        # Convert year to datetime
        data['Date'] = pd.to_datetime(data['Year'].astype(str), format='%Y')
        data.dropna(inplace=True)

        st.subheader("Cleaned Data Preview")
        st.write(data.head())

        # Temperature Trend
        st.subheader("Annual Temperature Trend")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(data['Date'], data['Temperature'], marker='o', label='Temp (\u00b0C)', color='orange')
        ax.set_xlabel("Year")
        ax.set_ylabel("Temperature (\u00b0C)")
        ax.set_title("Annual Temperature Over Time")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        # Rainfall Trend (if exists)
        if 'Rainfall' in data.columns:
            st.subheader("Annual Rainfall Trend")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(data['Date'], data['Rainfall'], marker='s', label='Rainfall (mm)', color='blue')
            ax.set_xlabel("Year")
            ax.set_ylabel("Rainfall (mm)")
            ax.set_title("Annual Rainfall Over Time")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

        # Correlation
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots()
        corr = data[['Year', 'Temperature'] + (['Rainfall'] if 'Rainfall' in data.columns else [])].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Model Training
        st.subheader("Model Training")
        features = data[['Year'] + (['Rainfall'] if 'Rainfall' in data.columns else [])]
        target = data['Temperature']

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        lr = LinearRegression()    
        lr.fit(data[['Year', 'Rainfall']], data['Temperature'])  # Make sure your data includes 'Rainfall'

        # --- Real-time Prediction Interface ---
        st.subheader("📈 Predict Temperature Using Custom Input")

        with st.form(key='prediction_form'):
            st.markdown("### Enter Prediction Parameters:")
            input_year = st.number_input("Year (e.g., 2027)", min_value=1900, max_value=2100, value=2025)
    
            # Check if Rainfall column exists before asking for input
            if 'Rainfall' in data.columns:
               input_rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, max_value=5000.0, value=1000.0)
               input_data = pd.DataFrame({'Year': [input_year], 'Rainfall': [input_rainfall]})
            else:
               input_data = pd.DataFrame({'Year': [input_year]})
    
            submit_button = st.form_submit_button(label="Predict")

        if submit_button:
            #prediction = model.predict(input_data)
            prediction = lr.predict(input_data)
            st.success(f"🌡️ Predicted Temperature for {input_year}: **{prediction[0]:.2f} °C**")

        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_preds = lr.predict(X_test)
        lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))

        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
        rf_r2 = r2_score(y_test, rf_preds)

        st.write(f"**Linear Regression RMSE:** {lr_rmse:.2f}")
        st.write(f"**Random Forest RMSE:** {rf_rmse:.2f}")
        st.write(f"**Random Forest R2 Score:** {rf_r2:.2f}")

        # Forecasting
        st.subheader("Forecast Future Annual Temperatures (2025–2030)")
        future_years = pd.DataFrame({"Year": np.arange(2025, 2031)})

        if 'Rainfall' in data.columns:
            avg_rainfall = data['Rainfall'].mean()
            future_years['Rainfall'] = avg_rainfall

        future_preds = lr.predict(future_years)

        # Combine for plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data['Year'], data['Temperature'], label='Historical Temp', color='blue', marker='o')
        ax.plot(future_years['Year'], future_preds, label='Forecasted Temp', color='green', marker='o', linestyle='--')
        ax.set_title("Forecasted Annual Temperatures (2025–2030)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Temperature (\u00b0C)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Optional: Polynomial Regression
        st.subheader("Polynomial Trend Fit")
        poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        poly_model.fit(features, target)
        poly_preds = poly_model.predict(future_years)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data['Year'], data['Temperature'], label='Historical', color='blue')
        ax.plot(future_years['Year'], poly_preds, label='Polynomial Forecast', color='purple', linestyle='--', marker='o')
        ax.set_title("Polynomial Forecast (Degree 2)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Temperature (\u00b0C)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    else:
        st.error(f"Uploaded file must contain the columns: {required_cols}")
else:
    st.info("Please upload a CSV file containing Year, Temperature, and optionally Rainfall.")

st.markdown("---")
st.header("📦 Project Deliverables")

st.markdown("""
- ✅ **Final Report**: Summarizing goals, methods, EDA, and model performance.
- ✅ **Python Code**: Scripts for cleaning, analysis, prediction, and deployment.
- ✅ **Streamlit Web App**: This interactive tool.
- ✅ **Conclusion**: Built skills in data analysis, modeling, and visualization using climate data from Tanzania.
""")

st.header("🔍 Next Steps / Future Work")
st.markdown("""
- Add new climate variables like humidity and wind speed.
- Analyze seasonal/long-term effects.
- Expand the project to include other regions.
""")