import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Main function for the Streamlit app
def main():
    st.title("Global Ocean Levels Predictor 🌊")
    st.write("""
    This app uses historical global ocean level data to predict future trends.
    The dataset is loaded automatically as part of the app.
    """)

    # Load the dataset from the app's local directory
    file_path = "Global_sea_level_rise.csv"
    try:
        data = pd.read_csv(file_path)
        st.success("Dataset loaded successfully!")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return

    # Display the column names for debugging
    st.write("### Dataset Column Names")
    st.write(data.columns)

    # Show a preview of the data
    st.write("### Data Preview")
    st.dataframe(data.head())

    # Check if the expected columns are present
    if 'year' not in data.columns or 'mmfrom1993-2008average' not in data.columns:
        st.error("The dataset must contain 'year' and 'mmfrom1993-2008average' columns.")
        return

    # Prepare data for modeling
    X = data['year'].values.reshape(-1, 1)
    y = data['mmfrom1993-2008average'].values.reshape(-1, 1)

    # Logarithmic transformation of X
    X_log = np.log(X)

    # Fit a linear regression model to the logarithmic transformation
    model = LinearRegression()
    model.fit(X_log, y)

    # Slider to select the number of years to predict
    years_to_predict = st.slider("Years into the future to predict:", min_value=1, max_value=100, value=50)

    # Generate future predictions
    current_year = int(data['year'].max())
    future_years = np.arange(current_year + 1, current_year + years_to_predict + 1).reshape(-1, 1)
    future_years_log = np.log(future_years)  # Apply log transformation to future years
    future_sea_levels = model.predict(future_years_log)

    # Combine historical and predicted data
    future_data = pd.DataFrame({'year': future_years.flatten(), 'mmfrom1993-2008average': future_sea_levels.flatten()})
    combined_data = pd.concat([data, future_data], ignore_index=True)

    # Plot the historical and predicted data
    plt.figure(figsize=(10, 6))
    plt.scatter(data['year'], data['mmfrom1993-2008average'], color='blue', label='Historical Data')
    plt.plot(combined_data['year'], combined_data['mmfrom1993-2008average'], color='red', label='Predicted Trend')
    plt.xlabel("Year")
    plt.ylabel("Sea Level (mm from 1993-2008 average)")
    plt.title("Global Ocean Levels Prediction (Logarithmic Model)")
    plt.legend()
    st.pyplot(plt)

# Run the app
if __name__ == "__main__":
    main()
