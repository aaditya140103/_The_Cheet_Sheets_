import streamlit as st
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(
    page_title="Pandas Cheat Sheet",
    page_icon="🐼",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("📚 Pandas Cheat Sheet")
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

# Sample DataFrame for demos
def get_sample_df():
    df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'Salary': [50000, 60000, 70000, 80000],
        'Department': ['HR', 'IT', 'Finance', 'IT']
    })
    return df

# Introduction
if section == "📘 Introduction":
    st.title("🐼 Introduction to Pandas")
    st.markdown("""
    **Pandas** is a powerful Python library for data analysis and manipulation.

    - `Series`: One-dimensional labeled array.
    - `DataFrame`: Two-dimensional labeled data structure.

    > Think of it like Excel in Python, but way more powerful!
    """)

    st.code("""
import pandas as pd
import numpy as np
    """, language="python")

# Creating DataFrames
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
import numpy as np
np_data = np.array([[1, 2], [3, 4]])
df = pd.DataFrame(np_data, columns=['A', 'B'])
    """, language="python")

# Data Inspection
elif section == "🔍 Data Inspection":
    st.title("🔍 Inspecting Your Data")
    df = get_sample_df()
    st.dataframe(df)

    st.subheader("Head & Tail")
    st.code("""
df.head()
df.tail()
    """, language="python")

    st.subheader("Data Types & Info")
    st.code("""
df.dtypes
df.info()
    """, language="python")

    st.subheader("Summary Statistics")
    st.code("df.describe(include='all')")
    st.dataframe(df.describe(include='all'))

# Indexing & Selection
elif section == "🎯 Indexing & Selection":
    st.title("🎯 Indexing & Selection")
    df = get_sample_df()
    st.dataframe(df)

    st.subheader("Using `[]`, `.loc`, and `.iloc`")
    st.code("""
df['Name']            # Column

# Row by label and position
df.loc[0]
df.iloc[0]

# Subset
df.loc[0:2, ['Name', 'Salary']]
    """, language="python")

# Data Cleaning
elif section == "🧹 Data Cleaning":
    st.title("🧹 Cleaning Data")
    df = get_sample_df()
    df.loc[2, 'Salary'] = np.nan
    st.dataframe(df)

    st.subheader("Handling Missing Values")
    st.code("""
df.isna()
df.fillna(0)
df.dropna()
    """, language="python")

# Aggregation & Grouping
elif section == "📊 Aggregation & Grouping":
    st.title("📊 Aggregation & Grouping")
    df = get_sample_df()
    st.dataframe(df)

    st.subheader("Groupby")
    st.code("""
df.groupby('Department')['Salary'].mean()
    """, language="python")
    st.write(df.groupby('Department')['Salary'].mean())

# Merging & Joining
elif section == "🔗 Merging & Joining":
    st.title("🔗 Merging & Joining DataFrames")
    st.subheader("Merge on Key")
    st.code("""
df1 = pd.DataFrame({'ID': [1, 2], 'Name': ['Alice', 'Bob']})
df2 = pd.DataFrame({'ID': [1, 2], 'Score': [90, 85]})
merged = pd.merge(df1, df2, on='ID')
    """, language="python")

# Time Series Basics
elif section == "⏱️ Time Series Basics":
    st.title("⏱️ Time Series Basics")
    st.code("""
dates = pd.date_range('2023-01-01', periods=5)
df = pd.DataFrame({'Date': dates, 'Value': range(5)})
df.set_index('Date', inplace=True)
    """, language="python")

    dates = pd.date_range('2023-01-01', periods=5)
    ts_df = pd.DataFrame({'Date': dates, 'Value': range(5)}).set_index('Date')
    st.line_chart(ts_df)

st.markdown("---")
st.caption("Created with ❤️ by Aditya's ML Cheat Sheet")
