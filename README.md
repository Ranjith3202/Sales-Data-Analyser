import streamlit as st
import pandas as pd
import numpy as np
import mysql.connector
import matplotlib.pyplot as plt
import io
from sklearn.linear_model import LinearRegression
from datetime import timedelta


st.title("📊 Sales Data Analytics Dashboard")

# -------------------------------
# File Upload
# -------------------------------
file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if file:
    # Read file
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.subheader("📌 Original Data")
    st.write(df)

    # Original Shape
    st.subheader("📏 Original Data Shape")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # -------------------------------
    # Data Cleaning Options
    # -------------------------------
    st.subheader("🧹 Data Cleaning Options")

    remove_dup = st.checkbox("Remove Duplicates")

    null_option = st.selectbox(
        "Handle Missing Values",
        ["None", "Mean", "Median", "Zero"]
    )

    # -------------------------------
    # Apply Cleaning
    # -------------------------------
    if st.button("Apply Cleaning"):

        df_clean = df.copy()

        # Remove duplicates
        if remove_dup:
            df_clean = df_clean.drop_duplicates(subset=['OrderID'])

        # Handle null values
        if null_option != "None":
            for col in df_clean.columns:
                if df_clean[col].dtype != "object":
                    if null_option == "Mean":
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                    elif null_option == "Median":
                        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                    elif null_option == "Zero":
                        df_clean[col] = df_clean[col].fillna(0)
                else:
                    df_clean[col] = df_clean[col].fillna("Unknown")

        # Save cleaned data
        st.session_state["cleaned_data"] = df_clean

        st.success("✅ Data cleaned successfully!")

# -------------------------------
# Show Cleaned Data (ONLY AFTER BUTTON)
# -------------------------------
if "cleaned_data" in st.session_state:

    df = st.session_state["cleaned_data"]

    st.subheader("✅ Cleaned Data")
    st.write(df)

    # Cleaned Shape
    st.subheader("📏 Cleaned Data Shape")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # -------------------------------
    # Visualization
    # -------------------------------
    st.subheader("📊 Visualization")

    col1 = st.selectbox("Select X column", df.columns)
    col2 = st.selectbox("Select Y column", df.columns)

    chart_type = st.selectbox("Select Chart Type", ["Bar", "Line", "Pie"])

    fig, ax = plt.subplots()

    if chart_type == "Bar":
        df.groupby(col1)[col2].sum().plot(kind='bar', ax=ax)

    elif chart_type == "Line":
        df.groupby(col1)[col2].sum().plot(kind='line', ax=ax)

    elif chart_type == "Pie":
        df.groupby(col1)[col2].sum().plot(kind='pie', autopct='%1.1f%%', ax=ax)

    st.pyplot(fig)

    # -------------------------------
    # Store in MySQL
    # -------------------------------
    st.subheader("💾 Store Data in MySQL")

    if st.button("Upload to MySQL"):

        try:
            conn = mysql.connector.connect(
                host="localhost",
                user="root",
                password="12345678",
                database="sales_db"
            )

            cursor = conn.cursor()

            cols = ", ".join([f"{col} TEXT" for col in df.columns])
            cursor.execute(f"CREATE TABLE IF NOT EXISTS cleaned_data ({cols})")

            for _, row in df.iterrows():
                values = tuple(str(x) for x in row)
                placeholders = ", ".join(["%s"] * len(values))
                cursor.execute(
                    f"INSERT INTO cleaned_data VALUES ({placeholders})",
                    values
                )

            conn.commit()
            st.success("Data stored successfully in MySQL!")

        except Exception as e:
            st.error(f"Error: {e}")

        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

    # -------------------------------
    # Download Excel
    # -------------------------------
    st.subheader("📥 Download Cleaned Data")


    def convert_to_excel(data):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            data.to_excel(writer, index=False, sheet_name='Cleaned Data')
        return output.getvalue()


    excel_file = convert_to_excel(df)

    st.download_button(
        label="⬇️ Download Cleaned Data as Excel",
        data=excel_file,
        file_name="cleaned_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# -------------------------------
# ML Prediction Section
# -------------------------------
st.subheader("🤖 Sales Prediction")

# User input for number of days
num_days = st.number_input("Enter number of days to predict", min_value=1, max_value=30, value=4)

if st.button("Predict Future Sales"):

    try:
        df_ml = df.copy()

        # Convert Date column
        df_ml['Date'] = pd.to_datetime(df_ml['Date'], errors='coerce')

        # Drop invalid dates
        df_ml = df_ml.dropna(subset=['Date'])

        # Aggregate daily sales
        df_grouped = df_ml.groupby('Date')['Amount'].sum().reset_index()

        # Convert date to numeric
        df_grouped['Date_ordinal'] = df_grouped['Date'].map(pd.Timestamp.toordinal)

        # Features & target
        X = df_grouped[['Date_ordinal']]
        y = df_grouped['Amount']

        # Train model
        model = LinearRegression()
        model.fit(X, y)

        # Generate future dates dynamically
        last_date = df_grouped['Date'].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, int(num_days) + 1)]

        future_ordinal = [[d.toordinal()] for d in future_dates]

        predictions = model.predict(future_ordinal)

        # Create result DataFrame
        pred_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Amount": predictions
        })

        st.subheader("📅 Predicted Sales")
        st.write(pred_df)

        # -------------------------------
        # Visualization
        # -------------------------------
        fig2, ax2 = plt.subplots()

        # Actual data
        ax2.plot(df_grouped['Date'], df_grouped['Amount'], label='Actual')

        # Predicted data
        ax2.plot(pred_df['Date'], pred_df['Predicted Amount'], linestyle='dashed', label='Predicted')

        ax2.legend()
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Prediction Error: {e}")
