# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose

st.title("Climate Temperature Analysis & Forecasting (Tanzania)")

# File Upload
uploaded_file = st.file_uploader("Upload Temperature CSV File", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file, on_bad_lines='skip', engine='python')

    # Clean and rename columns
    data.rename(columns=lambda x: x.strip().capitalize(), inplace=True)
    data.rename(columns={'ANN': 'Mean'}, inplace=True)  # Ensure case matches after capitalize()

    # Show data preview
    st.subheader("Raw Data Preview")
    st.write(data.head())

    # Column check
    required_cols = ['Year', 'Mean']
    if all(col in data.columns for col in required_cols):
        data = data[required_cols].copy()

        """

        data['Year'] = pd.to_datetime(data['Year'])
        data.rename(columns={'Year': 'Date', 'Mean': 'Temperature'}, inplace=True)
        data.dropna(inplace=True)

        data = data.set_index('Date').resample('MS').mean().reset_index()

        """
        # Convert year to datetime (assumes annual data)
        data.rename(columns={'Mean': 'Temperature'}, inplace=True)
        data['Date'] = pd.to_datetime(data['Year'].astype(str), format='%Y')

        # You CANNOT resample yearly data monthly
        # So skip resampling if you don't have monthly data

        data = data[['Date', 'Temperature']].dropna()
        st.write("Final data shape:", data.shape)

        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Season'] = data['Month'].apply(lambda x: 'Dry' if x in [6,7,8,9] else 'Wet')

        """
        # Plot time series
        st.subheader("Temperature Trend Over Time")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(data['Date'], data['Temperature'], color='orange', label='Monthly Avg Temp')
        ax.set_title('Monthly Temperature Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Temperature (°C)')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        """

        # Temperature trend plot
        st.subheader("Temperature Trend Over Time")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(data['Date'], data['Temperature'], marker='o', linestyle='-', color='orange', label='Annual Temp')
        ax.set_title('Annual Temperature Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Temperature (°C)')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        
        """
        # Seasonal decomposition
        st.subheader("Seasonal Decomposition")
        data_indexed = data.set_index('Date').asfreq('MS')
        decomp = seasonal_decompose(data_indexed['Temperature'], model='additive', period=12)
        fig = decomp.plot()
        st.pyplot(fig)

        """

        # Seasonal Decomposition (only if >=24 observations)
        if len(data) >= 24:
            st.subheader("Seasonal Decomposition")
            data_indexed = data.set_index('Date').asfreq('MS')  # Monthly frequency
            try:
                decomp = seasonal_decompose(data_indexed['Temperature'], model='additive', period=12)
                fig = decomp.plot()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Decomposition error: {e}")
        else:
            st.warning("Not enough data for seasonal decomposition (requires at least 24 monthly points).")

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        corr = data[['Temperature', 'Year', 'Month']].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Machine Learning Models
        st.subheader("Model Training & Evaluation")
        features = data[['Year', 'Month']]
        target = data['Temperature']
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

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
        st.subheader("Forecast Future Temperatures (2025–2030)")
        future_years = pd.DataFrame({
            'Year': np.repeat(np.arange(2025, 2031), 12),
            'Month': np.tile(np.arange(1, 13), 6)
        })
        future_preds = rf.predict(future_years)
        future_dates = pd.date_range(start='2025-01', end='2030-12', freq='MS')

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(future_dates, future_preds, label='Predicted Temp (2025-2030)', color='green')
        ax.set_title('Forecasted Monthly Temperatures (2025–2030)')
        ax.set_xlabel('Year')
        ax.set_ylabel('Predicted Temperature (°C)')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.error(f"Uploaded file must contain the columns: {required_cols}")
else:
    st.info("Please upload a CSV file to start analysis.")




