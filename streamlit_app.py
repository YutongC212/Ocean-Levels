import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the dataset directly from the app
@st.cache_data
def load_data():
    # Path to the dataset stored in your app
    file_path = "data/sea_level_data.csv"
    data = pd.read_csv(file_path)
    return data

# Main function for the Streamlit app
def main():
    st.title("Global Sea Level Rise Predictor 🌊")
    st.write("""
    This app uses historical global sea level data to predict future trends.
    Adjust the slider below to predict sea level rise over the coming years.
    """)

    # Load data
    data = load_data()

    # Check the first few rows of the data
    st.write("### Historical Data Preview")
    st.dataframe(data.head())

    # Prepare data for modeling (assuming the dataset has columns 'Year' and 'mmfrom1993-2008average')
    data.columns = ['year', 'sea_level']  # Modify as needed based on your dataset
    X = data['year'].values.reshape(-1, 1)
    y = data['sea_level'].values.reshape(-1, 1)

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Slider to select the number of years to predict
    years_to_predict = st.slider("Years into the future to predict:", min_value=1, max_value=100, value=50)

    # Generate future predictions
    current_year = int(data['year'].max())
    future_years = np.arange(current_year + 1, current_year + years_to_predict + 1).reshape(-1, 1)
    future_sea_levels = model.predict(future_years)

    # Combine historical and predicted data
    future_data = pd.DataFrame({'Year': future_years.flatten(), 'Sea Level': future_sea_levels.flatten()})
    combined_data = pd.concat([data, future_data], ignore_index=True)

    # Plot the historical and predicted data
    plt.figure(figsize=(10, 6))
    plt.scatter(data['year'], data['sea_level'], color='blue', label='Historical Data')
    plt.plot(combined_data['Year'], combined_data['Sea Level'], color='red', label='Predicted Trend')
    plt.xlabel("Year")
    plt.ylabel("Sea Level (mm)")
    plt.title("Global Sea Level Rise Prediction")
    plt.legend()
    st.pyplot(plt)

# Run the app
if __name__ == "__main__":
    main()
