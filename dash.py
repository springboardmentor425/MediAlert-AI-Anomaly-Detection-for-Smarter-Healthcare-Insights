import streamlit as st, pandas as pd, numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set the page configuration
st.set_page_config(
    page_title="Healthcare Providers Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and Description
st.title("📊 Healthcare Providers Dashboard")
st.markdown("""
Welcome to the **Healthcare Providers Dashboard**! Upload your CSV file to explore the data with interactive charts and insights.
""")

# Sidebar for Upload and Filters
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Check if a file is uploaded
if uploaded_file is not None:
    # Load the data
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")

    # Display Dataset Overview
    st.subheader("Dataset Overview")
    with st.expander("View Data"):
        st.write(data.head())

    # Data Summary
    st.subheader("Summary Statistics")
    with st.expander("View Summary"):
        st.write(data.describe())

    # Layout: Two columns for visualizations
    col1, col2 = st.columns(2)

    # Bar Chart (Categorical Analysis)
    with col1:
        st.subheader("Bar Chart (Categorical Data)")
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        if len(categorical_columns) > 0:
            selected_cat_col = st.selectbox("Select a Categorical Column", categorical_columns)
            bar_data = data[selected_cat_col].value_counts().reset_index()
            bar_data.columns = ['Category', 'Count']
            fig_bar = px.bar(
                bar_data,
                x="Category",
                y="Count",
                title=f"Distribution of {selected_cat_col}",
                labels={"Category": "Category", "Count": "Count"},
                color="Count",
                template="plotly_dark"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.write("No categorical columns available.")

    # Histogram (Numerical Analysis)
    with col2:
        st.subheader("Histogram (Numerical Data)")
        numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_columns) > 0:
            selected_num_col = st.selectbox("Select a Numerical Column", numeric_columns)
            fig_hist = px.histogram(
                data,
                x=selected_num_col,
                title=f"Distribution of {selected_num_col}",
                labels={selected_num_col: "Values"},
                nbins=30,
                template="plotly_dark"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.write("No numerical columns available.")

    # Scatter Plot (Relationships)
    st.subheader("Scatter Plot (Relationships)")
    if len(numeric_columns) > 1:
        col_x = st.selectbox("X-axis", numeric_columns, key="scatter_x")
        col_y = st.selectbox("Y-axis", numeric_columns, key="scatter_y")
        fig_scatter = px.scatter(
            data,
            x=col_x,
            y=col_y,
            color=data[categorical_columns[0]] if categorical_columns else None,
            title=f"Scatter Plot: {col_x} vs {col_y}",
            labels={col_x: col_x, col_y: col_y},
            template="plotly_dark"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.write("Not enough numerical columns for a scatter plot.")
else:
    st.sidebar.warning("Please upload a CSV file to proceed.")
    st.write("📂 Upload a CSV file using the sidebar to explore the data.")

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using [Streamlit](https://streamlit.io/) and [Plotly](https://plotly.com/).")
