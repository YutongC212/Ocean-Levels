import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

def main():
    st.title("Global Ocean Levels Predictor 🌊")
    st.write("""
    This app uses historical global ocean level data to predict future trends.
    The dataset is loaded automatically as part of the app.
    """)

    file_path = "Global_sea_level_rise.csv"
    try:
        data = pd.read_csv(file_path)
        st.success("dataset loaded successfully!")
    except Exception as e:
        st.error(f"error loading dataset: {e}")
        return

    st.write("### dataset column names")
    st.write(data.columns)

    st.write("### data preview")
    st.dataframe(data.head())

    if 'year' not in data.columns or 'mmfrom1993-2008average' not in data.columns:
        st.error("the dataset must contain 'year' and 'mmfrom1993-2008average' columns.")
        return

    X = data['year'].values.reshape(-1, 1)
    y = data['mmfrom1993-2008average'].values.reshape(-1, 1)

    degree = 3
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    years_to_predict = st.slider("years into the future to predict:", min_value=1, max_value=100, value=50)

    current_year = int(data['year'].max())
    future_years = np.arange(current_year + 1, current_year + years_to_predict + 1).reshape(-1, 1)
    future_years_poly = poly.transform(future_years)
    future_sea_levels = model.predict(future_years_poly)

    future_data = pd.DataFrame({'year': future_years.flatten(), 'mmfrom1993-2008average': future_sea_levels.flatten()})
    combined_data = pd.concat([data, future_data], ignore_index=True)

    plt.figure(figsize=(10, 6))
    plt.scatter(data['year'], data['mmfrom1993-2008average'], color='blue', label='historical data')
    plt.plot(combined_data['year'], combined_data['mmfrom1993-2008average'], color='red', label='predicted trend')
    plt.xlabel("year")
    plt.ylabel("sea level (mm from 1993-2008 average)")
    plt.title("global ocean levels prediction (polynomial regression model)")
    plt.legend()
    st.pyplot(plt)

if __name__ == "__main__":
    main()
