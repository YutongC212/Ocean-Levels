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
    Upload your dataset in CSV format to analyze and predict future sea levels.
    """)

    # File uploader for the dataset
    uploaded_file = st.file_uploader("Upload the 'Global Ocean Levels' dataset (CSV format):", type="csv")
    if uploaded_file is not None:
        # Read the uploaded file
        try:
            data = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return

        # Display the column names for debugging
        st.write("### Dataset Column Names")
        st.write(data.columns)

        # Show a preview of the data
        st.write("### Data Preview")
        st.dataframe(data.head())

        # Check if the expected columns are present (with lowercase 'year')
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
    else:
        st.warning("Please upload the 'Global Ocean Levels' dataset to proceed.")

# Run the app
if __name__ == "__main__":
    main()
