import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Pandas Cheat Sheet",
    page_icon="🐼",
    layout="wide"
)

# --- Sidebar Navigation ---
st.sidebar.title("🐼 Pandas Cheat Sheet")
st.sidebar.markdown("Use this cheat sheet to quickly look up common Pandas operations.")

section = st.sidebar.radio("Jump to Section", [
    "📘 Introduction",
    "📋 Creating DataFrames",
    "🔍 Data Inspection",
    "🎯 Indexing & Selection",
    "🧹 Data Cleaning",
    "📊 Aggregation & Grouping",
    "🔗 Merging & Joining",
    "⏱️ Time Series Basics"
])

# --- Sample DataFrame for Demos ---
def get_sample_df():
    df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'Salary': [50000, 60000, 70000, 80000],
        'Department': ['HR', 'IT', 'Finance', 'IT']
    })
    return df

# --- Content Area ---
if section == "📘 Introduction":
    st.title("🐼 Introduction to Pandas")
    st.markdown("""
    Pandas is a Python library for data manipulation and analysis.

    **Core Data Structures:**
    - `Series`: One-dimensional labeled array.
    - `DataFrame`: Two-dimensional labeled table (like a spreadsheet).

    It's widely used in data analysis, ML pipelines, and even finance.
    """)
    st.code("""
import pandas as pd
import numpy as np
    """, language="python")

elif section == "📋 Creating DataFrames":
    st.title("📋 Creating DataFrames")

    st.subheader("From Dictionary")
    st.code("""
data = {
    'Name': ['Alice', 'Bob'],
    'Age': [25, 30]
}
df = pd.DataFrame(data)
    """, language="python")
    st.dataframe(pd.DataFrame({'Name': ['Alice', 'Bob'], 'Age': [25, 30]}))

    st.subheader("From NumPy Array")
    st.code("""
np_data = np.array([[1, 2], [3, 4]])
df = pd.DataFrame(np_data, columns=['A', 'B'])
    """, language="python")
    st.dataframe(pd.DataFrame(np.array([[1, 2], [3, 4]]), columns=['A', 'B']))

    st.subheader("From CSV File")
    st.code("""
df = pd.read_csv("your_file.csv")
    """, language="python")

elif section == "🔍 Data Inspection":
    st.title("🔍 Inspecting Your Data")
    df = get_sample_df()

    st.subheader("👀 Preview Data")
    st.dataframe(df.head())

    st.subheader("🔎 Info and Types")
    st.code("""
df.info()
df.dtypes
    """, language="python")

    st.subheader("📐 Summary Stats")
    st.dataframe(df.describe(include='all'))

elif section == "🎯 Indexing & Selection":
    st.title("🎯 Indexing & Selection")
    df = get_sample_df()

    st.subheader("Select Columns")
    st.code("""
df['Name']
df[['Name', 'Age']]
    """, language="python")

    st.subheader("Select Rows")
    st.code("""
df.iloc[0]         # by position
df.loc[0]          # by label
    """, language="python")

    st.subheader("Boolean Filtering")
    st.code("""
df[df['Salary'] > 60000]
    """, language="python")

elif section == "🧹 Data Cleaning":
    st.title("🧹 Data Cleaning")
    df = get_sample_df()
    df.loc[2, 'Salary'] = np.nan
    st.dataframe(df)

    st.subheader("Detecting & Filling Missing Values")
    st.code("""
df.isna()
df.fillna(0)
    """, language="python")

    st.subheader("Dropping Missing Values")
    st.code("df.dropna()", language="python")

elif section == "📊 Aggregation & Grouping":
    st.title("📊 Aggregation & Grouping")
    df = get_sample_df()

    st.subheader("Using groupby")
    st.code("df.groupby('Department')['Salary'].mean()", language="python")
    st.dataframe(df.groupby('Department')['Salary'].mean().reset_index())

elif section == "🔗 Merging & Joining":
    st.title("🔗 Merging & Joining DataFrames")

    st.subheader("Basic Merge")
    st.code("""
df1 = pd.DataFrame({'ID': [1, 2], 'Name': ['Alice', 'Bob']})
df2 = pd.DataFrame({'ID': [1, 2], 'Score': [90, 85]})
pd.merge(df1, df2, on='ID')
    """, language="python")

elif section == "⏱️ Time Series Basics":
    st.title("⏱️ Time Series Basics")

    st.subheader("Creating a Time Index")
    st.code("""
dates = pd.date_range('2023-01-01', periods=5)
df = pd.DataFrame({'Date': dates, 'Value': range(5)})
df.set_index('Date', inplace=True)
    """, language="python")

    dates = pd.date_range('2023-01-01', periods=5)
    ts_df = pd.DataFrame({'Date': dates, 'Value': range(5)}).set_index('Date')
    st.line_chart(ts_df)

# Footer
st.markdown("""
---
Made with ❤️ by **Aditya's ML Cheat Sheet** 
[📘 GitHub](https://github.com/) | [📬 Feedback](mailto:you@example.com)
""")
