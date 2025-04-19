import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set page configuration
st.set_page_config(
    page_title="Pandas Syntax Reference",
    page_icon="📊",
    layout="wide"
)

# Main title
st.title("📊 Comprehensive Pandas Syntax Reference")
st.markdown("""
This interactive app provides a complete reference for pandas syntax commonly used in data analysis,
data science, and machine learning workflows. Use the navigation menu to explore different sections.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Select a Section",
    ["Introduction", 
     "Creating DataFrames", 
     "Data Inspection", 
     "Selection & Indexing",
     "Data Cleaning",
     "Data Manipulation",
     "Grouping & Aggregation",
     "Merging & Joining",
     "Time Series",
     "Visualization",
     "Statistical Analysis",
     "Machine Learning Integration",
     "Input/Output Operations",
     "Advanced Operations"]
)

# Sample data for examples
def generate_sample_data():
    # Create a basic DataFrame
    df = pd.DataFrame({
        'A': np.random.rand(10),
        'B': np.random.randn(10),
        'C': np.random.randint(0, 10, 10),
        'D': pd.date_range(start='2023-01-01', periods=10),
        'E': ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'alpha', 'beta', 'gamma', 'delta', 'epsilon'],
        'F': [True, False, True, True, False, False, True, False, True, False]
    })
    
    # Create some missing values
    df.loc[3:5, 'A'] = np.nan
    df.loc[7, 'C'] = np.nan
    
    return df

df = generate_sample_data()

# Create sales data for time series examples
def generate_sales_data():
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    sales = np.random.randint(100, 1000, size=len(dates))
    sales_data = pd.DataFrame({'date': dates, 'sales': sales})
    sales_data['day_of_week'] = sales_data['date'].dt.day_name()
    sales_data['month'] = sales_data['date'].dt.month_name()
    sales_data['quarter'] = sales_data['date'].dt.quarter
    sales_data['year'] = sales_data['date'].dt.year
    
    # Add some seasonality
    sales_data.loc[sales_data['day_of_week'].isin(['Saturday', 'Sunday']), 'sales'] *= 1.5
    sales_data.loc[sales_data['month'].isin(['December']), 'sales'] *= 2
    
    return sales_data

sales_df = generate_sales_data()

# Code display function
def display_code(code, language="python"):
    st.code(code, language=language)

# Section: Introduction
if section == "Introduction":
    st.header("Introduction to Pandas")
    
    st.subheader("What is Pandas?")
    st.write("""
    Pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool,
    built on top of the Python programming language. It's particularly well suited for working with tabular data.
    """)
    
    st.subheader("Key Components")
    st.write("""
    - **Series**: 1D labeled array
    - **DataFrame**: 2D labeled data structure with columns of potentially different types
    - **Index**: Labels for rows and columns
    """)
    
    st.subheader("Import Pandas")
    display_code("""
    import pandas as pd
    import numpy as np  # Often used alongside pandas
    """)
    
    st.subheader("Basic Series & DataFrame Examples")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Creating a Series:")
        display_code("""
        # From a list
        s = pd.Series([1, 3, 5, np.nan, 6, 8])
        
        # With custom index
        s = pd.Series([1, 3, 5, 7], index=['a', 'b', 'c', 'd'])
        
        # From dictionary
        s = pd.Series({'a': 1, 'b': 3, 'c': 5})
        """)
    
    with col2:
        st.write("Creating a DataFrame:")
        display_code("""
        # From a dictionary of Series or arrays
        df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': pd.Timestamp('20230101'),
            'C': pd.Series(1, index=list(range(4)), dtype='float'),
            'D': np.array([3] * 4, dtype='int32'),
            'E': pd.Categorical(["test", "train", "test", "train"]),
            'F': 'foo'
        })
        """)

# Section: Creating DataFrames
elif section == "Creating DataFrames":
    st.header("Creating DataFrames")
    
    st.subheader("From Dictionary")
    display_code("""
    # From dictionary of lists
    df = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': ['a', 'b', 'c', 'd'],
        'C': [True, False, True, False]
    })
    
    # From dictionary of Series
    df = pd.DataFrame({
        'A': pd.Series([1, 2, 3, 4]),
        'B': pd.Series(['a', 'b', 'c', 'd'])
    })
    """)
    
    st.subheader("From Lists")
    display_code("""
    # From list of lists
    df = pd.DataFrame([
        [1, 'a', True],
        [2, 'b', False],
        [3, 'c', True],
        [4, 'd', False]
    ], columns=['A', 'B', 'C'])
    
    # With custom index
    df = pd.DataFrame([
        [1, 'a', True],
        [2, 'b', False],
        [3, 'c', True],
        [4, 'd', False]
    ], columns=['A', 'B', 'C'], index=['w', 'x', 'y', 'z'])
    """)
    
    st.subheader("From NumPy Array")
    display_code("""
    # From NumPy array
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    df = pd.DataFrame(array, columns=['A', 'B', 'C'])
    
    # From random data
    df = pd.DataFrame(np.random.randn(5, 3), columns=['A', 'B', 'C'])
    """)
    
    st.subheader("From CSV and Other Data Sources")
    display_code("""
    # From CSV file
    df = pd.read_csv('filename.csv')
    
    # From Excel file
    df = pd.read_excel('filename.xlsx', sheet_name='Sheet1')
    
    # From SQL query
    import sqlite3
    conn = sqlite3.connect('database.db')
    df = pd.read_sql_query("SELECT * FROM table", conn)
    
    # From JSON
    df = pd.read_json('filename.json')
    
    # From HTML table
    df = pd.read_html('https://example.com/table.html')[0]
    """)
    
    st.subheader("Empty DataFrame with Defined Structure")
    display_code("""
    # Empty DataFrame with column names
    df = pd.DataFrame(columns=['A', 'B', 'C'])
    
    # With specific dtypes
    df = pd.DataFrame({
        'A': pd.Series(dtype='int'),
        'B': pd.Series(dtype='float'),
        'C': pd.Series(dtype='str'),
        'D': pd.Series(dtype='datetime64[ns]')
    })
    """)

# Section: Data Inspection
elif section == "Data Inspection":
    st.header("Data Inspection")
    
    st.subheader("Basic Inspection")
    st.write("Sample DataFrame for examples:")
    st.dataframe(df.head())
    
    display_code("""
    # View first n rows (default 5)
    df.head()
    df.head(10)
    
    # View last n rows (default 5)
    df.tail()
    df.tail(10)
    
    # Random sample of rows
    df.sample(5)
    df.sample(frac=0.25)  # 25% of the data
    
    # DataFrame dimensions (rows, columns)
    df.shape
    
    # Basic information about DataFrame
    df.info()
    
    # Column data types
    df.dtypes
    
    # Index information
    df.index
    
    # Column names
    df.columns
    
    # Quick statistical summary
    df.describe()
    
    # Include categorical columns in describe
    df.describe(include='all')
    
    # Memory usage
    df.memory_usage()
    """)
    
    st.subheader("Count Values")
    display_code("""
    # Count non-NA values in each column
    df.count()
    
    # Count unique values in a column
    df['E'].nunique()
    
    # Count occurrences of each value
    df['E'].value_counts()
    
    # With normalize option for percentage
    df['E'].value_counts(normalize=True)
    
    # Get unique values
    df['E'].unique()
    """)
    
    st.subheader("Missing Values")
    display_code("""
    # Check for missing values
    df.isna()  # or df.isnull()
    
    # Count missing values per column
    df.isna().sum()
    
    # Percentage of missing values
    df.isna().mean() * 100
    
    # Drop missing values
    df.dropna()
    
    # Fill missing values
    df.fillna(0)  # fill with a constant
    df.fillna(method='ffill')  # forward fill
    df.fillna(method='bfill')  # backward fill
    df.fillna(df.mean())  # fill with column mean
    """)

# Section: Selection & Indexing
elif section == "Selection & Indexing":
    st.header("Selection & Indexing")
    
    st.subheader("Sample DataFrame:")
    st.dataframe(df.head())
    
    st.subheader("Basic Selection")
    display_code("""
    # Select a single column (returns Series)
    df['A']
    
    # Select multiple columns (returns DataFrame)
    df[['A', 'B']]
    
    # Select rows by position
    df.iloc[0]  # First row
    df.iloc[0:5]  # First five rows
    df.iloc[[0, 3, 6]]  # Specific rows
    
    # Select rows by label
    df.loc[0]  # Row with index 0
    df.loc[0:5]  # Rows with index 0 through 5 (inclusive)
    df.loc[[0, 3, 6]]  # Specific rows by index
    
    # Select both rows and columns
    df.iloc[0:5, 0:3]  # First 5 rows, first 3 columns
    df.loc[0:5, ['A', 'C']]  # Mix of index and column names
    """)
    
    st.subheader("Boolean Indexing")
    display_code("""
    # Filter rows based on a column value
    df[df['A'] > 0.5]
    
    # Multiple conditions with & (and) and | (or)
    df[(df['A'] > 0.5) & (df['C'] < 5)]
    df[(df['A'] > 0.5) | (df['C'] < 5)]
    
    # Using query method
    df.query('A > 0.5 and C < 5')
    
    # Check if values are in a list
    df[df['E'].isin(['alpha', 'beta'])]
    
    # Filter based on string operations
    df[df['E'].str.contains('al')]
    df[df['E'].str.startswith('a')]
    """)
    
    st.subheader("Setting & Resetting Index")
    display_code("""
    # Set a column as index
    df_indexed = df.set_index('E')
    
    # Reset index (turn index into column)
    df_reset = df_indexed.reset_index()
    
    # Multi-level indexing
    df_multi = df.set_index(['E', 'F'])
    
    # Access with multi-level index
    df_multi.loc['alpha']
    df_multi.loc[('alpha', True)]
    
    # Cross-section on multi-level index
    df_multi.xs('alpha', level='E')
    """)
    
    st.subheader("Advanced Indexing")
    display_code("""
    # Get rows where column value is maximum
    df.loc[df['A'].idxmax()]
    
    # Where values match a condition
    df.where(df > 0)  # Replaces non-matching with NaN
    
    # Mask values based on condition
    df.mask(df > 0)  # Replaces matching with NaN
    
    # Get a cross-section of the data
    df.xs(0)  # Cross-section for index value 0
    
    # Get the diagonal values
    df.values.diagonal()  # Only works for square DataFrames
    """)

# Section: Data Cleaning
elif section == "Data Cleaning":
    st.header("Data Cleaning")
    
    st.subheader("Sample DataFrame with Issues:")
    messy_df = pd.DataFrame({
        'name': ['John Smith', ' jane  Doe', 'Bob Johnson ', None],
        'age': [25, '30', np.nan, '45'],
        'email': ['john@example.com', 'jane@example', 'bob', None],
        'date': ['2023-01-01', '01/15/2023', '2023.02.28', None]
    })
    st.dataframe(messy_df)
    
    st.subheader("Handling Missing Values")
    display_code("""
    # Identify missing values
    df.isna()  # or df.isnull()
    
    # Count missing values in each column
    df.isna().sum()
    
    # Drop rows with any missing values
    df.dropna()
    
    # Drop rows where all values are missing
    df.dropna(how='all')
    
    # Drop columns with missing values
    df.dropna(axis=1)
    
    # Only keep rows with at least 2 non-NA values
    df.dropna(thresh=2)
    
    # Fill missing values with a specific value
    df.fillna(0)
    
    # Fill with mean, median, or mode
    df['A'].fillna(df['A'].mean())
    df['A'].fillna(df['A'].median())
    df['A'].fillna(df['A'].mode()[0])
    
    # Forward fill (use previous value)
    df.fillna(method='ffill')
    
    # Backward fill (use next value)
    df.fillna(method='bfill')
    
    # Fill with different values for each column
    df.fillna({'A': 0, 'B': 'unknown'})
    
    # Interpolate missing values
    df.interpolate()
    df.interpolate(method='linear')  # default
    df.interpolate(method='polynomial', order=2)
    """)
    
    st.subheader("Data Type Conversion")
    display_code("""
    # Convert a column to numeric
    pd.to_numeric(df['age'])
    pd.to_numeric(df['age'], errors='coerce')  # Invalid becomes NaN
    
    # Convert to datetime
    pd.to_datetime(df['date'])
    pd.to_datetime(df['date'], errors='coerce')
    pd.to_datetime(df['date'], format='%Y-%m-%d')
    
    # Convert to string
    df['age'].astype(str)
    
    # Convert entire DataFrame
    df = df.astype({'A': 'int64', 'B': 'float64', 'E': 'category'})
    
    # Infer better data types (reduce memory)
    df.convert_dtypes()
    """)
    
    st.subheader("String Cleaning")
    display_code("""
    # Strip whitespace
    df['name'].str.strip()
    df['name'].str.lstrip()  # left only
    df['name'].str.rstrip()  # right only
    
    # Change case
    df['name'].str.lower()
    df['name'].str.upper()
    df['name'].str.title()
    
    # Replace values
    df['name'].str.replace('John', 'Jonathan')
    df['name'].str.replace(r'\\d+', '', regex=True)  # Remove digits
    
    # Extract using regex
    df['name'].str.extract(r'(\\w+)\\s(\\w+)')  # Extract first and last name
    
    # Split strings
    df['name'].str.split(' ', expand=True)  # Split into columns
    
    # Pad strings
    df['name'].str.pad(10, side='left', fillchar='0')
    
    # Check if string contains pattern
    df['name'].str.contains('John')
    df['name'].str.match(r'^J.*h$')  # Regex pattern
    """)
    
    st.subheader("Removing Duplicates")
    display_code("""
    # Check for duplicates
    df.duplicated()
    df.duplicated(subset=['A', 'B'])  # Consider only specific columns
    
    # Count duplicates
    df.duplicated().sum()
    
    # Remove duplicates
    df.drop_duplicates()
    df.drop_duplicates(subset=['A', 'B'])
    df.drop_duplicates(keep='last')  # Keep last occurrence
    df.drop_duplicates(keep=False)   # Drop all duplicates
    """)
    
    st.subheader("Handling Outliers")
    display_code("""
    # Identify outliers using IQR method
    Q1 = df['A'].quantile(0.25)
    Q3 = df['A'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter out outliers
    df_filtered = df[(df['A'] >= lower_bound) & (df['A'] <= upper_bound)]
    
    # Cap outliers (winsorization)
    df['A_capped'] = df['A'].clip(lower=lower_bound, upper=upper_bound)
    
    # Z-score method
    from scipy import stats
    z_scores = stats.zscore(df['A'].dropna())
    df_filtered = df[(z_scores < 3) & (z_scores > -3)]
    """)

# Section: Data Manipulation
elif section == "Data Manipulation":
    st.header("Data Manipulation")
    
    st.subheader("Column Operations")
    display_code("""
    # Add a new column
    df['G'] = df['A'] + df['B']
    
    # Apply a function to a column
    df['A_squared'] = df['A'].apply(lambda x: x**2)
    
    # Apply a function element-wise to the DataFrame
    df.applymap(lambda x: x*2 if isinstance(x, (int, float)) else x)
    
    # Rename columns
    df.rename(columns={'A': 'Alpha', 'B': 'Beta'})
    
    # Drop columns
    df.drop(['A', 'B'], axis=1)
    
    # Reorder columns
    df = df[['C', 'A', 'B', 'D', 'E', 'F']]
    
    # Select columns by data type
    df.select_dtypes(include=['number'])
    df.select_dtypes(exclude=['object'])
    """)
    
    st.subheader("Row Operations")
    display_code("""
    # Add a new row
    new_row = pd.Series(['value1', 'value2'], index=['A', 'B'])
    df = df.append(new_row, ignore_index=True)  # Pre pandas 2.0
    
    # More modern approach - use concat
    new_row_df = pd.DataFrame([['value1', 'value2']], columns=['A', 'B'])
    df = pd.concat([df, new_row_df], ignore_index=True)
    
    # Drop rows
    df.drop([0, 1])  # Drop by index
    
    # Sort rows
    df.sort_values('A')
    df.sort_values(['A', 'B'], ascending=[True, False])
    df.sort_index()  # Sort by index
    """)
    
    st.subheader("Numerical Operations")
    display_code("""
    # Basic math operations
    df['A'] + df['B']
    df['A'] - df['B']
    df['A'] * df['B']
    df['A'] / df['B']
    df['A'] ** 2  # Exponentiation
    
    # Cumulative operations
    df['A'].cumsum()
    df['A'].cumprod()
    
    # Round values
    df.round(2)
    df['A'].round(2)
    
    # Absolute values
    df['B'].abs()
    
    # Log, exp, etc.
    np.log(df['A'])
    np.exp(df['A'])
    np.sqrt(df['A'])
    """)
    
    st.subheader("Apply & Map Functions")
    display_code("""
    # Apply a function to each element in a Series
    df['A'].apply(lambda x: x*2)
    
    # Apply a function along an axis of a DataFrame
    df.apply(np.sum, axis=0)  # Sum of each column
    df.apply(np.sum, axis=1)  # Sum of each row
    
    # Apply a function element-wise
    df.applymap(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
    
    # Map values in a Series
    mapping = {'alpha': 'a', 'beta': 'b', 'gamma': 'g'}
    df['E'].map(mapping)
    
    # Replace values
    df['E'].replace({'alpha': 'a', 'beta': 'b'})
    """)
    
    st.subheader("Binning & Cutting")
    display_code("""
    # Cut data into bins
    pd.cut(df['A'], bins=3)
    pd.cut(df['A'], bins=[0, 0.25, 0.5, 0.75, 1.0])
    
    # Cut with labels
    pd.cut(df['A'], bins=3, labels=['Low', 'Medium', 'High'])
    
    # Equal-frequency bins (quantiles)
    pd.qcut(df['A'], q=4)  # Quartiles
    pd.qcut(df['A'], q=[0, 0.25, 0.5, 0.75, 1.0])
    
    # Bin with labels
    pd.qcut(df['A'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    """)
    
    st.subheader("Pivoting & Reshaping")
    display_code("""
    # Pivot: reshape from long to wide format
    pivoted = df.pivot(index='D', columns='E', values='A')
    
    # Pivot table with aggregation
    pivot_table = pd.pivot_table(df, values='A', index='D', 
                                columns='E', aggfunc=np.mean)
    
    # Unpivot: reshape from wide to long format
    melted = pd.melt(df, id_vars=['D'], value_vars=['A', 'B', 'C'])
    
    # Stack: pivot columns to index
    stacked = df.set_index(['D']).stack()
    
    # Unstack: pivot index levels to columns
    unstacked = stacked.unstack()
    
    # Explode a column with lists into multiple rows
    df['list_col'] = [[1, 2], [3, 4], [5, 6]]
    df.explode('list_col')
    """)

# Section: Grouping & Aggregation
elif section == "Grouping & Aggregation":
    st.header("Grouping & Aggregation")
    
    st.subheader("Basic GroupBy Operations")
    display_code("""
    # Group by a single column
    grouped = df.groupby('E')
    
    # Group by multiple columns
    grouped = df.groupby(['E', 'F'])
    
    # Basic aggregate functions
    grouped.mean()
    grouped.sum()
    grouped.count()
    grouped.min()
    grouped.max()
    grouped.median()
    grouped.std()
    grouped.var()
    
    # Get group sizes
    grouped.size()
    
    # First/last row of each group
    grouped.first()
    grouped.last()
    
    # Iterate through groups
    for name, group in df.groupby('E'):
        print(name)
        print(group)
    """)
    
    st.subheader("Advanced Aggregations")
    display_code("""
    # Multiple aggregations on one column
    df.groupby('E')['A'].agg(['mean', 'sum', 'count'])
    
    # Different aggregations for different columns
    df.groupby('E').agg({
        'A': 'sum',
        'B': 'mean',
        'C': ['min', 'max']
    })
    
    # Named aggregations
    df.groupby('E').agg(
        A_mean=('A', 'mean'),
        A_sum=('A', 'sum'),
        B_mean=('B', 'mean'),
        C_max=('C', 'max')
    )
    
    # Custom aggregation functions
    df.groupby('E').agg(lambda x: x.max() - x.min())
    
    # Aggregate with filter
    df.groupby('E').filter(lambda x: x['A'].mean() > 0.5)
    """)
    
    st.subheader("Transformation & Window Functions")
    display_code("""
    # Apply transformation to groups
    df.groupby('E')['A'].transform('mean')  # Replace with group mean
    
    # Calculate percentage of group total
    df['A'] / df.groupby('E')['A'].transform('sum') * 100
    
    # Rank within groups
    df.groupby('E')['A'].transform('rank')
    
    # Cumulative sum within groups
    df.groupby('E')['A'].transform('cumsum')
    
    # Custom transformation
    df.groupby('E')['A'].transform(lambda x: (x - x.mean()) / x.std())
    
    # Rolling window functions
    df.groupby('E')['A'].rolling(window=2).mean()
    
    # Expanding window 
    df.groupby('E')['A'].expanding().mean()
    """)
    
    st.subheader("Groupby with Multiple Keys")
    display_code("""
    # Group by multiple columns
    multi_grouped = df.groupby(['E', 'F'])
    
    # Aggregate
    multi_grouped.sum()
    
    # Select a group
    multi_grouped.get_group(('alpha', True))
    
    # Unstack results
    multi_grouped.sum().unstack()
    
    # Reset index to columns
    multi_grouped.sum().reset_index()
    """)

# Section: Merging & Joining
elif section == "Merging & Joining":
    st.header("Merging & Joining")
    
    st.write("Creating sample DataFrames for joins:")
    display_code("""
    # Sample DataFrames
    df1 = pd.DataFrame({
        'key': ['A', 'B', 'C', 'D'],
        'value1': [1, 2, 3, 4]
    })
    
    df2 = pd.DataFrame({
        'key': ['B', 'D', 'E', 'F'],
        'value2': [5, 6, 7, 8]
    })
    """)
    
    st.subheader("Concatenation")
    display_code("""
    # Vertical concatenation (append)
    pd.concat([df1, df2])
    
    # Horizontal concatenation (side by side)
    pd.concat([df1, df2], axis=1)
    
    # With join options
    pd.concat([df1, df2], join='inner')  # only matching columns
    pd.concat([df1, df2], join='outer')  # all columns (default)
    
    # Ignore index
    pd.concat([df1, df2], ignore_index=True)
    
    # Add keys to create hierarchical index
    pd.concat([df1, df2], keys=['df1', 'df2'])
    """)
    
    st.subheader("Merge & Join Operations")
    display_code("""
    # Inner join (only matching keys)
    pd.merge(df1, df2, on='key')
    
    # Outer join (all keys from both)
    pd.merge(df1, df2, on='key', how='outer')
    
    # Left join (all from left, matching from right)
    pd.merge(df1, df2, on='key', how='left')
    
    # Right join (all from right, matching from left)
    pd.merge(df1, df2, on='key', how='right')
    
    # Join on different column names
    pd.merge(df1, df2, left_on='key1', right_on='key2')
    
    # Join on index
    pd.merge(df1, df2, left_index=True, right_index=True)
    
    # Join with both index and columns
    pd.merge(df1, df2, left_on='key', right_index=True)
    
    # DataFrame join method
    df1.join(df2, lsuffix='_left', rsuffix='_right')
    """)
    
    st.subheader("Advanced Join Operations")
    display_code("""
    # Join with indicator
    pd.merge(df1, df2, on='key', how='outer', indicator=True)
    
    # Join with suffixes for duplicate columns
    pd.merge(df1, df2, on='key', suffixes=('_left', '_right'))
    
    # Multiple keys
    pd.merge(df1, df2, on=['key1', 'key2'])
    """)
  # Section: Time Series
elif section == "Time Series":
    st.header("Time Series Analysis")
    
    st.subheader("Sample Time Series Data:")
    st.dataframe(sales_df.head())
    
    st.subheader("Creating and Converting Time Series")
    display_code("""
    # Create a date range
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    
    # Common frequencies
    pd.date_range('2023-01-01', periods=12, freq='M')  # Month end
    pd.date_range('2023-01-01', periods=12, freq='MS')  # Month start
    pd.date_range('2023-01-01', periods=4, freq='Q')   # Quarter end
    pd.date_range('2023-01-01', periods=4, freq='QS')  # Quarter start
    pd.date_range('2023-01-01', periods=52, freq='W')  # Week end
    pd.date_range('2023-01-01', periods=12, freq='B')  # Business days
    pd.date_range('2023-01-01', periods=24, freq='H')  # Hourly
    pd.date_range('2023-01-01', periods=60, freq='T')  # Minutely
    
    # Convert string to datetime
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Invalid becomes NaN
    
    # Set datetime as index
    df_ts = df.set_index('date')
    """)
    
    st.subheader("DateTime Components")
    display_code("""
    # Extract components
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_name'] = df['date'].dt.day_name()
    df['quarter'] = df['date'].dt.quarter
    df['is_month_end'] = df['date'].dt.is_month_end
    df['is_month_start'] = df['date'].dt.is_month_start
    df['week_of_year'] = df['date'].dt.isocalendar().week
    
    # Hour, minute, second for timestamp data
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['second'] = df['datetime'].dt.second
    """)
    
    st.subheader("Time Series Resampling")
    display_code("""
    # Downsample (reduce frequency)
    df_ts.resample('M').mean()  # Monthly average
    df_ts.resample('Q').sum()   # Quarterly sum
    df_ts.resample('A').max()   # Annual maximum
    
    # Multiple aggregations
    df_ts.resample('M').agg(['mean', 'min', 'max', 'sum'])
    
    # Upsample (increase frequency)
    df_monthly = df_ts.resample('M').mean()
    df_monthly.resample('D').ffill()  # Forward fill daily data
    df_monthly.resample('D').bfill()  # Backward fill
    df_monthly.resample('D').interpolate()  # Interpolate
    """)
    
    st.subheader("Time Series Shifting & Rolling")
    display_code("""
    # Shift values
    df_ts.shift(1)  # Shift 1 period forward (lag)
    df_ts.shift(-1)  # Shift 1 period backward (lead)
    
    # Percent change
    df_ts.pct_change()  # Period-over-period percent change
    df_ts.pct_change(periods=12)  # Year-over-year percent change
    
    # Rolling windows
    df_ts.rolling(window=7).mean()  # 7-day moving average
    df_ts.rolling(window=30).std()  # 30-day standard deviation
    
    # Exponentially weighted moving average
    df_ts.ewm(span=7).mean()  # Exponential moving average
    
    # Expanding window
    df_ts.expanding().mean()  # Cumulative mean
    df_ts.expanding().max()   # Cumulative maximum
    """)
    
    st.subheader("Time Series Operations")
    display_code("""
    # Date filtering
    df[(df['date'] >= '2023-01-01') & (df['date'] <= '2023-06-30')]
    
    # Between dates
    df[df['date'].between('2023-01-01', '2023-06-30')]
    
    # Filter by day of week
    df[df['date'].dt.dayofweek == 0]  # Mondays
    
    # Group by time components
    df.groupby(df['date'].dt.month).mean()
    df.groupby([df['date'].dt.year, df['date'].dt.month]).sum()
    
    # Time zone handling
    df['date_utc'] = df['date'].dt.tz_localize('UTC')
    df['date_est'] = df['date_utc'].dt.tz_convert('US/Eastern')
    
    # Handling business days
    from pandas.tseries.offsets import BDay
    df['next_bday'] = df['date'] + BDay(1)
    """)

# Section: Visualization
elif section == "Visualization":
    st.header("Visualization with Pandas")
    
    st.subheader("Basic Plotting")
    display_code("""
    # Import visualization libraries
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')  # or other styles
    
    # Line plot
    df.plot()  # Plot all numeric columns
    df.plot(y='A')  # Plot specific column
    df.plot(y=['A', 'B'])  # Plot multiple columns
    
    # With more options
    df.plot(
        y='A',
        figsize=(10, 6),
        title='Column A Over Time',
        grid=True,
        legend=True
    )
    
    # Common plot types
    df.plot.line()
    df.plot.bar()
    df.plot.barh()  # Horizontal bar
    df.plot.hist()
    df.plot.box()
    df.plot.kde()  # Kernel Density Estimate
    df.plot.area()
    df.plot.scatter(x='A', y='B')
    df.plot.pie(y='A')
    """)
    
    st.subheader("Time Series Visualization")
    display_code("""
    # Time series line plot
    sales_df.set_index('date')['sales'].plot()
    
    # Seasonal plots
    sales_df.groupby(sales_df['date'].dt.month)['sales'].mean().plot.bar()
    
    # Plot with date formatting
    fig, ax = plt.subplots(figsize=(12, 6))
    sales_df.set_index('date')['sales'].plot(ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.set_title('Daily Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Heatmap for time series
    pivot = sales_df.pivot_table(
        index=sales_df['date'].dt.day_name(),
        columns=sales_df['date'].dt.month_name(),
        values='sales',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='viridis', ax=ax)
    ax.set_title('Average Sales by Day and Month')
    st.pyplot(fig)
    """)
    
    st.subheader("Statistical Visualization")
    display_code("""
    # Histogram
    fig, ax = plt.subplots()
    df['A'].plot.hist(bins=20, alpha=0.5, ax=ax)
    st.pyplot(fig)
    
    # Kernel Density Estimate (KDE)
    fig, ax = plt.subplots()
    df['A'].plot.kde(ax=ax)
    st.pyplot(fig)
    
    # Box plot
    fig, ax = plt.subplots()
    df.plot.box(ax=ax)
    st.pyplot(fig)
    
    # Violin plot using seaborn
    fig, ax = plt.subplots()
    sns.violinplot(data=df[['A', 'B', 'C']], ax=ax)
    st.pyplot(fig)
    
    # Pair plot
    pair_fig = sns.pairplot(df[['A', 'B', 'C']])
    st.pyplot(pair_fig)
    
    # Correlation heatmap
    fig, ax = plt.subplots()
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    """)
    
    st.subheader("Grouped Visualization")
    display_code("""
    # Grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    df.groupby('E')[['A', 'B']].mean().plot.bar(ax=ax)
    st.pyplot(fig)
    
    # Stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    df.groupby('E')[['A', 'B']].mean().plot.bar(stacked=True, ax=ax)
    st.pyplot(fig)
    
    # Grouped box plot
    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(by='E', column=['A', 'B'], ax=ax)
    st.pyplot(fig)
    
    # Facet grid with seaborn
    g = sns.FacetGrid(df, col='E', height=4, aspect=.7)
    g.map(sns.histplot, 'A')
    st.pyplot(g.fig)
    """)
    
    st.subheader("Advanced Visualization")
    display_code("""
    # Customize plots with matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(y='A', ax=ax, color='blue', linewidth=2)
    ax.set_title('Custom Plot', fontsize=16)
    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add horizontal line
    ax.axhline(y=df['A'].mean(), color='r', linestyle='--', label='Mean')
    ax.legend()
    st.pyplot(fig)
    
    # Multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Top left: Line plot
    df['A'].plot(ax=axes[0, 0], title='Line Plot')
    
    # Top right: Histogram
    df['B'].plot.hist(ax=axes[0, 1], bins=20, title='Histogram')
    
    # Bottom left: Scatter plot
    df.plot.scatter(x='A', y='B', ax=axes[1, 0], title='Scatter Plot')
    
    # Bottom right: Box plot
    df[['A', 'B', 'C']].plot.box(ax=axes[1, 1], title='Box Plot')
    
    plt.tight_layout()
    st.pyplot(fig)
    """)

# Section: Statistical Analysis
elif section == "Statistical Analysis":
    st.header("Statistical Analysis")
    
    st.subheader("Descriptive Statistics")
    display_code("""
    # Basic statistics
    df.describe()  # Numeric columns
    df.describe(include='all')  # All columns
    
    # Individual statistics
    df['A'].mean()
    df['A'].median()
    df['A'].mode()
    df['A'].min()
    df['A'].max()
    df['A'].std()  # Standard deviation
    df['A'].var()  # Variance
    df['A'].sem()  # Standard error of mean
    df['A'].skew()  # Skewness
    df['A'].kurt()  # Kurtosis
    
    # Count values
    df['E'].value_counts()
    df['E'].value_counts(normalize=True)  # Proportions
    
    # Quantiles
    df['A'].quantile()  # 0.5 by default (median)
    df['A'].quantile([0.25, 0.5, 0.75, 0.9])
    
    # Summary by group
    df.groupby('E')['A'].describe()
    """)
    
    st.subheader("Correlation and Covariance")
    display_code("""
    # Correlation matrix
    df.corr()  # Pearson correlation
    df.corr(method='spearman')  # Spearman rank correlation
    df.corr(method='kendall')  # Kendall Tau correlation
    
    # Correlation between specific columns
    df['A'].corr(df['B'])
    
    # Covariance matrix
    df.cov()
    
    # Covariance between specific columns
    df['A'].cov(df['B'])
    
    # Correlation with p-values using scipy
    from scipy import stats
    stats.pearsonr(df['A'], df['B'])
    stats.spearmanr(df['A'], df['B'])
    """)
    
    st.subheader("Hypothesis Testing")
    display_code("""
    # Import stats modules
    from scipy import stats
    
    # One-sample t-test
    stats.ttest_1samp(df['A'], 0)  # Test if mean equals 0
    
    # Two-sample t-test
    stats.ttest_ind(df[df['E'] == 'alpha']['A'], 
                   df[df['E'] == 'beta']['A'])
    
    # Paired t-test
    stats.ttest_rel(df['A'], df['B'])
    
    # ANOVA
    groups = [df[df['E'] == 'alpha']['A'], 
              df[df['E'] == 'beta']['A'], 
              df[df['E'] == 'gamma']['A']]
    stats.f_oneway(*groups)
    
    # Chi-squared test of independence
    crosstab = pd.crosstab(df['E'], df['F'])
    stats.chi2_contingency(crosstab)
    
    # Shapiro-Wilk test for normality
    stats.shapiro(df['A'])
    
    # Mann-Whitney U test (non-parametric)
    stats.mannwhitneyu(df[df['E'] == 'alpha']['A'], 
                      df[df['E'] == 'beta']['A'])
    """)
    
    st.subheader("Regression Analysis")
    display_code("""
    # Simple linear regression using numpy
    import numpy as np
    x = df['A'].values
    y = df['B'].values
    mask = ~np.isnan(x) & ~np.isnan(y)  # Remove NaN values
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    
    # Simple linear regression with stats
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
    r_squared = r_value**2
    
    # Multiple regression with statsmodels
    import statsmodels.api as sm
    
    X = df[['A', 'C']].dropna()
    y = df['B'].loc[X.index]
    X = sm.add_constant(X)  # Add intercept
    
    model = sm.OLS(y, X).fit()
    print(model.summary())
    
    # Predictions
    predictions = model.predict(X)
    residuals = y - predictions
    """)
    
    st.subheader("Time Series Analysis")
    display_code("""
    # Autocorrelation
    from pandas.plotting import autocorrelation_plot
    
    # Serial correlation
    df['A'].autocorr()  # Lag 1 autocorrelation
    df['A'].autocorr(lag=2)  # Lag 2 autocorrelation
    
    # Autocorrelation and partial autocorrelation plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    pd.plotting.autocorrelation_plot(df['A'], ax=ax1)
    sm.graphics.tsa.plot_pacf(df['A'].dropna(), lags=10, ax=ax2)
    st.pyplot(fig)
    
    # Decompose time series
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    ts = df.set_index('D')['A'].dropna()
    decomposition = seasonal_decompose(ts, model='additive', period=7)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
    decomposition.observed.plot(ax=ax1)
    ax1.set_title('Observed')
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonality')
    decomposition.resid.plot(ax=ax4)
    ax4.set_title('Residuals')
    plt.tight_layout()
    st.pyplot(fig)
    """)

# Section: Machine Learning Integration
elif section == "Machine Learning Integration":
    st.header("Machine Learning Integration")
    
    st.subheader("Data Preparation")
    display_code("""
    # Import necessary libraries
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    
    # Prepare data for machine learning
    # Split data into features and target
    X = df[['A', 'B', 'C']]
    y = df['F']
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # One-hot encode categorical variables
    categorical_cols = ['E']
    categorical_data = pd.get_dummies(df[categorical_cols], drop_first=True)
    
    # Combine with numerical data
    preprocessed_df = pd.concat([df[['A', 'B', 'C']], categorical_data], axis=1)
    """)
    
    st.subheader("Pipeline Creation")
    display_code("""
    # Create data preprocessing pipeline
    numeric_features = ['A', 'B', 'C']
    categorical_features = ['E']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Combine preprocessing with a model
    from sklearn.linear_model import LogisticRegression
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])
    
    # Fit model
    model.fit(df[numeric_features + categorical_features], df['F'])
    """)
    
    st.subheader("Linear Models")
    display_code("""
    # Linear Regression
    from sklearn.linear_model import LinearRegression
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Model evaluation
    from sklearn.metrics import mean_squared_error, r2_score
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Get coefficients
    coefs = pd.DataFrame(model.coef_, index=X.columns, columns=['Coefficient'])
    intercept = model.intercept_
    
    # Ridge Regression (L2 regularization)
    from sklearn.linear_model import Ridge
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    
    # Lasso Regression (L1 regularization)
    from sklearn.linear_model import Lasso
    
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)
    """)
    
    st.subheader("Classification Models")
    display_code("""
    # Logistic Regression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    # Create binary target (for example purposes)
    binary_target = (df['C'] > df['C'].median()).astype(int)
    X = df[['A', 'B']]
    y = binary_target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    
    # Predictions
    y_pred = log_reg.predict(X_test)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    # Probability predictions
    y_prob = log_reg.predict_proba(X_test)
    
    # Decision Trees
    from sklearn.tree import DecisionTreeClassifier
    
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X_train, y_train)
    tree_pred = tree.predict(X_test)
    
    # Random Forest
    from sklearn.ensemble import RandomForestClassifier
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    
    # Feature importance
    feature_importance = pd.DataFrame(
        rf.feature_importances_,
        index=X.columns,
        columns=['importance']
    ).sort_values('importance', ascending=False)
    """)
    
    st.subheader("Model Evaluation & Cross-Validation")
    display_code("""
    # Cross-validation
    from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
    
    # Basic cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    
    # With KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf)
    
    # Grid search for hyperparameter tuning
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
        'max_iter': [1000, 2000]
    }
    
    grid = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error')
    grid.fit(X, y)
    
    # Best parameters
    best_params = grid.best_params_
    
    # Best estimator
    best_model = grid.best_estimator_
    
    # ROC curve for classification
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    
    # Get predicted probabilities for positive class
    y_prob = log_reg.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)
    
    # Learning curves
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_sizes, train_mean, label='Training score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    ax.plot(train_sizes, test_mean, label='Cross-validation score')
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2)
    ax.set_xlabel('Training size')
    ax.set_ylabel('Score')
    ax.set_title('Learning Curves')
    ax.legend(loc="best")
    st.pyplot(fig)
    """)

# Section: Input/Output Operations
elif section == "Input/Output Operations":
    st.header("Input/Output Operations")
    
    st.subheader("Reading Data")
    display_code("""
    # Reading from CSV
    df = pd.read_csv('file.csv')
    
    # With options
    df = pd.read_csv('file.csv', 
                     sep=',',  # Delimiter
                     header=0,  # Row to use as header
                     index_col=0,  # Column to use as index
                     skiprows=2,  # Skip the first 2 rows
                     nrows=100,  # Read only 100 rows
                     na_values=['NA', 'N/A', '-'],  # Custom NA values
                     parse_dates=['date_column'],  # Parse date columns
                     encoding='utf-8',  # File encoding
                     dtype={'A': 'float64', 'B': 'str'})  # Column dtypes
    
    # Reading from Excel
    df = pd.read_excel('file.xlsx')
    df = pd.read_excel('file.xlsx', 
                       sheet_name='Sheet1',
                       header=0, 
                       index_col=0)
    
    # Reading multiple sheets
    xlsx = pd.ExcelFile('file.xlsx')
    sheet_names = xlsx.sheet_names  # Get sheet names
    df_dict = pd.read_excel(xlsx, sheet_name=None)  # Dictionary of DataFrames
    
    # Reading from SQL
    import sqlite3
    conn = sqlite3.connect('database.db')
    df = pd.read_sql('SELECT * FROM table', conn)
    df = pd.read_sql_query('SELECT * FROM table WHERE column > 5', conn)
    df = pd.read_sql_table('table', conn)
    
    # Reading from JSON
    df = pd.read_json('file.json')
    df = pd.read_json('file.json', orient='records')
    
    # Reading from HTML tables
    tables = pd.read_html('https://en.wikipedia.org/wiki/Some_table')
    df = tables[0]  # Get first table
    
    # Reading from clipboard
    df = pd.read_clipboard()
    
    # Reading from fixed-width file
    df = pd.read_fwf('file.txt', widths=[10, 20, 15, 10])
    """)
    
    st.subheader("Writing Data")
    display_code("""
    # Writing to CSV
    df.to_csv('output.csv')
    
    # With options
    df.to_csv('output.csv',
              sep=',',
              index=False,  # Don't write index
              header=True,  # Write header
              columns=['A', 'B', 'C'],  # Only specific columns
              encoding='utf-8',
              date_format='%Y-%m-%d',  # Format for date columns
              float_format='%.2f')  # Format for float columns
    
  # Writing to Excel
    df.to_excel('output.xlsx')
    df.to_excel('output.xlsx', 
                sheet_name='Sheet1',
                index=False,
                freeze_panes=(1, 0))  # Freeze first row
    
    # Writing multiple DataFrames to different sheets
    with pd.ExcelWriter('output.xlsx') as writer:
        df1.to_excel(writer, sheet_name='Sheet1')
        df2.to_excel(writer, sheet_name='Sheet2')
        df3.to_excel(writer, sheet_name='Sheet3')
    
    # Writing to SQL
    import sqlite3
    conn = sqlite3.connect('database.db')
    df.to_sql('table_name', conn, if_exists='replace', index=False)
    # if_exists options: 'fail', 'replace', 'append'
    
    # Writing to JSON
    df.to_json('output.json')
    df.to_json('output.json', orient='records')  # List-like: [{col:val}, ...]
    df.to_json('output.json', orient='split')    # Split by index, columns and data
    df.to_json('output.json', orient='index')    # Index -> {column -> value} nested JSON
    
    # Writing to HTML
    df.to_html('output.html')
    df.to_html('output.html', 
               classes=['table', 'table-striped'],
               escape=False)  # Don't escape HTML
    
    # Writing to pickle (Python binary format)
    df.to_pickle('output.pkl')
    
    # Writing to clipboard
    df.to_clipboard()
    
    # Writing to string buffer
    from io import StringIO
    buffer = StringIO()
    df.to_csv(buffer)
    csv_string = buffer.getvalue()
    """)
    
    st.subheader("Other I/O Operations")
    display_code("""
    # Read/write parquet files (efficient columnar storage)
    df.to_parquet('output.parquet')
    df = pd.read_parquet('input.parquet')
    
    # Read/write HDF5 files (hierarchical data format)
    df.to_hdf('output.h5', key='df')
    df = pd.read_hdf('input.h5', key='df')
    
    # Read/write feather files (fast on-disk format)
    df.to_feather('output.feather')
    df = pd.read_feather('input.feather')
    
    # Read/write stata files
    df.to_stata('output.dta')
    df = pd.read_stata('input.dta')
    
    # Read from Google BigQuery
    # Requires: pip install pandas-gbq
    import pandas_gbq
    df = pandas_gbq.read_gbq('SELECT * FROM dataset.table', project_id='your-project-id')
    
    # Read from Amazon S3
    # Requires: pip install s3fs
    df = pd.read_csv('s3://bucket/file.csv')
    
    # Convert between pandas and other formats
    # To/from NumPy array
    array = df.to_numpy()
    df = pd.DataFrame(array)
    
    # To/from dictionary
    records_dict = df.to_dict('records')  # List of dictionaries
    df = pd.DataFrame.from_records(records_dict)
    
    dict_of_series = df.to_dict()  # Dictionary of Series
    df = pd.DataFrame(dict_of_series)
    """)

# Section: Advanced Operations
elif section == "Advanced Operations":
    st.header("Advanced Operations")
    
    st.subheader("Window Functions")
    display_code("""
    # Rolling window calculations
    df['rolling_mean'] = df['A'].rolling(window=3).mean()
    df['rolling_std'] = df['A'].rolling(window=3).std()
    
    # With min_periods
    df['rolling_mean'] = df['A'].rolling(window=3, min_periods=1).mean()
    
    # Centered rolling window
    df['rolling_mean_centered'] = df['A'].rolling(window=3, center=True).mean()
    
    # Exponentially weighted window
    df['ewm_mean'] = df['A'].ewm(span=3).mean()
    df['ewm_std'] = df['A'].ewm(span=3).std()
    
    # Expanding window (cumulative)
    df['expanding_mean'] = df['A'].expanding().mean()
    df['expanding_max'] = df['A'].expanding().max()
    
    # Rolling with custom function
    df['rolling_custom'] = df['A'].rolling(window=3).apply(lambda x: x.max() - x.min())
    """)
    
    st.subheader("Working with Text Data")
    display_code("""
    # String methods (works on Series with string data)
    df['E'].str.lower()
    df['E'].str.upper()
    df['E'].str.title()
    df['E'].str.strip()
    df['E'].str.replace('a', 'X')
    
    # Splitting strings
    df['E'].str.split('a', expand=True)  # Returns DataFrame
    df['E'].str.split('a').str[0]  # Get first element after split
    
    # Extracting substrings
    df['E'].str[0:2]  # First two characters
    
    # Pattern extraction with regex
    df['E'].str.extract(r'([a-z]+)([0-9]+)')  # Returns DataFrame
    df['E'].str.extractall(r'([a-z]+)')  # Returns all matches
    
    # Find and replace with regex
    df['E'].str.replace(r'[aeiou]', 'X', regex=True)
    
    # String checks
    df['E'].str.contains('a')
    df['E'].str.startswith('a')
    df['E'].str.endswith('a')
    df['E'].str.isalpha()
    df['E'].str.isdigit()
    df['E'].str.isnumeric()
    
    # String length
    df['E'].str.len()
    
    # Pad strings
    df['E'].str.pad(10, side='left', fillchar='0')
    df['E'].str.ljust(10, fillchar='-')
    df['E'].str.rjust(10, fillchar='-')
    
    # Count occurrences
    df['E'].str.count('a')
    """)
    
    st.subheader("Categories and Sparse Data")
    display_code("""
    # Convert to categorical data type
    df['E'] = df['E'].astype('category')
    
    # Get category information
    df['E'].cat.categories
    df['E'].cat.codes  # Integer codes for categories
    
    # Reorder categories
    df['E'] = df['E'].cat.reorder_categories(['alpha', 'beta', 'gamma', 'delta', 'epsilon'])
    
    # Set categories (adds missing, removes unlisted)
    df['E'] = df['E'].cat.set_categories(['alpha', 'beta', 'new_cat'])
    
    # Add categories
    df['E'] = df['E'].cat.add_categories(['new_cat1', 'new_cat2'])
    
    # Remove categories
    df['E'] = df['E'].cat.remove_categories(['alpha'])
    
    # Remove unused categories
    df['E'] = df['E'].cat.remove_unused_categories()
    
    # Sparse data (memory-efficient for mostly same values)
    df['sparse'] = pd.Series([0, 0, 1, 0, 0, 0, 0, 2, 0, 0]).astype('Sparse')
    df['sparse'] = pd.Series([0, 0, 1, 0, 0, 0, 0, 2, 0, 0]).astype('Sparse[int]')
    df['sparse'] = pd.Series([0, 0, 1, 0, 0, 0, 0, 2, 0, 0]).astype(pd.SparseDtype(int, fill_value=0))
    """)
    
    st.subheader("Memory Usage Optimization")
    display_code("""
    # Check memory usage
    df.memory_usage()
    df.memory_usage(deep=True)  # Include Python objects memory
    
    # Total memory usage
    df.memory_usage(deep=True).sum() / 1024**2  # In MB
    
    # Optimize numeric types
    df_int = df.select_dtypes(include=['int'])
    optimized_df = df.copy()
    
    for col in df_int.columns:
        col_min = df[col].min()
        col_max = df[col].max()
        
        # Select best integer type
        if col_min >= 0:
            if col_max < 2**8:
                optimized_df[col] = df[col].astype('uint8')
            elif col_max < 2**16:
                optimized_df[col] = df[col].astype('uint16')
            elif col_max < 2**32:
                optimized_df[col] = df[col].astype('uint32')
            else:
                optimized_df[col] = df[col].astype('uint64')
        else:
            if col_min > -2**7 and col_max < 2**7:
                optimized_df[col] = df[col].astype('int8')
            elif col_min > -2**15 and col_max < 2**15:
                optimized_df[col] = df[col].astype('int16')
            elif col_min > -2**31 and col_max < 2**31:
                optimized_df[col] = df[col].astype('int32')
            else:
                optimized_df[col] = df[col].astype('int64')
    
    # Convert float to smaller types
    df_float = df.select_dtypes(include=['float'])
    for col in df_float.columns:
        optimized_df[col] = df[col].astype('float32')  # From float64 to float32
    
    # Use categories for object columns with few unique values
    df_obj = df.select_dtypes(include=['object'])
    for col in df_obj.columns:
        num_unique = df[col].nunique()
        if num_unique / len(df) < 0.5:  # If less than 50% unique values
            optimized_df[col] = df[col].astype('category')
    """)
    
    st.subheader("Performance Tips")
    display_code("""
    # Faster iteration with itertuples()
    for row in df.itertuples():
        print(row.Index, row.A, row.B)
    
    # Use .loc/.iloc instead of chained indexing
    # Slower and can create copies:
    df['A'][df['B'] > 0] = 0
    
    # Faster and avoids copies:
    df.loc[df['B'] > 0, 'A'] = 0
    
    # Use vectorized operations instead of apply when possible
    # Slower:
    df['A'].apply(lambda x: x*2)
    
    # Faster:
    df['A'] * 2
    
    # Use query() for filtering when dealing with large DataFrames
    # query() can be faster than boolean indexing
    filtered_df = df.query('A > 0 and B < 2')
    
    # Use inplace=True carefully
    # Can be more memory-efficient but prevents method chaining
    df.fillna(0, inplace=True)
    
    # When loading large CSV files, consider specifying dtypes
    dtypes = {'A': 'float32', 'B': 'float32', 'C': 'int8', 'E': 'category'}
    df = pd.read_csv('large_file.csv', dtype=dtypes)
    
    # Use eval() for complex computations on large DataFrames
    df.eval('D = A + B + C')
    df.eval('D = (A + B) / 2 + C')
    """)
    
    st.subheader("Working with MultiIndex")
    display_code("""
    # Create a MultiIndex DataFrame
    arrays = [
        ['alpha', 'alpha', 'beta', 'beta'],
        ['A', 'B', 'A', 'B']
    ]
    multi_index = pd.MultiIndex.from_arrays(arrays, names=('first', 'second'))
    df_multi = pd.DataFrame(np.random.randn(4, 2), index=multi_index, columns=['value1', 'value2'])
    
    # Access with tuple
    df_multi.loc[('alpha', 'A')]
    
    # Access with partial indexing
    df_multi.loc['alpha']
    
    # Cross-section
    df_multi.xs('A', level='second')
    
    # Stacking and unstacking
    stacked = df.set_index(['D', 'E']).stack()
    unstacked = stacked.unstack()
    unstacked2 = stacked.unstack(level=0)  # Specific level
    
    # Adding a level to MultiIndex
    df_multi.index = pd.MultiIndex.from_tuples(
        [(a, b, 'new_level') for a, b in df_multi.index],
        names=('first', 'second', 'third')
    )
    
    # Swapping levels
    df_multi.swaplevel('first', 'second')
    
    # Sorting index
    df_multi.sort_index()
    df_multi.sort_index(level='second')
    
    # Reset specific levels
    df_multi.reset_index(level='first')
    """)
    
    st.subheader("Advanced Groupby Operations")
    display_code("""
    # Group by function
    df.groupby(lambda x: x % 3).sum()
    
    # Group by multiple columns with different aggregations
    df.groupby(['E', 'F']).agg({
        'A': ['mean', 'min', max'],
        'B': ['median', 'std'],
        'C': lambda x: x.max() - x.min()
    })
    
    # Named aggregation (pandas >= 0.25)
    df.groupby('E').agg(
        mean_A=('A', 'mean'),
        sum_B=('B', 'sum'),
        range_C=('C', lambda x: x.max() - x.min())
    )
    
    # Filter groups based on group properties
    df.groupby('E').filter(lambda x: x['A'].mean() > 0.5)
    
    # Apply function to each group
    df.groupby('E').apply(lambda x: x.sort_values('A', ascending=False))
    
    # Custom groupby operation
    def top_n(group, n=2, column='A'):
        return group.nlargest(n, column)
        
    df.groupby('E').apply(top_n, n=2, column='B')
    
    # Group by time periods
    df.groupby(pd.Grouper(key='D', freq='M')).sum()
    
    # Group by bins
    df.groupby(pd.cut(df['A'], bins=[-np.inf, 0, 0.5, np.inf])).count()
    """)
