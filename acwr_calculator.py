import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io

# Set page configuration
st.set_page_config(
    page_title="ACWR Calculator",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper Functions
def calculate_rolling_average_acwr(data, acute_days=7, chronic_days=28):
    """
    Calculate ACWR using Rolling Average method
    """
    if len(data) < chronic_days:
        return np.nan

    acute_load = np.sum(data[-acute_days:])
    chronic_load = np.mean([np.sum(data[i:i+acute_days]) for i in range(len(data)-chronic_days+1, len(data)-acute_days+1)])

    if chronic_load == 0:
        return np.nan

    return acute_load / chronic_load

def calculate_ewma_acwr(data, acute_lambda=0.25, chronic_lambda=0.069, chronic_days=28):
    """
    Calculate ACWR using Exponentially Weighted Moving Average
    Fixed version that properly updates both acute and chronic EWMA for all data points
    """
    if len(data) < chronic_days:
        return np.nan

    # Initialize EWMA values with first data point
    acute_ewma = data[0]
    chronic_ewma = data[0]
    
    # ✅ FIX: Calculate EWMA for ALL data points, not just first 7
    for i in range(1, len(data)):
        acute_ewma = acute_lambda * data[i] + (1 - acute_lambda) * acute_ewma
        chronic_ewma = chronic_lambda * data[i] + (1 - chronic_lambda) * chronic_ewma

    if chronic_ewma == 0:
        return np.nan

    return acute_ewma / chronic_ewma

def categorize_acwr(acwr_value):
    """
    Categorize ACWR value into risk zones
    """
    if pd.isna(acwr_value):
        return "Insufficient Data", "gray"
    elif acwr_value < 0.8:
        return "Undertraining", "blue"
    elif 0.8 <= acwr_value <= 1.3:
        return "Optimal", "green"
    elif 1.3 < acwr_value <= 1.5:
        return "Overreaching", "orange"
    else:
        return "High Risk", "red"

def create_sample_data():
    """
    Create sample training data
    """
    dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
    np.random.seed(42)

    # Create realistic training load pattern
    base_load = 100
    weekly_pattern = np.tile([80, 90, 100, 110, 120, 100, 60], 9)[:60]  # Weekly pattern
    noise = np.random.normal(0, 10, 60)
    training_load = base_load + weekly_pattern + noise
    training_load = np.maximum(training_load, 0)  # Ensure no negative values

    return pd.DataFrame({
        'Date': dates,
        'Training_Load': training_load.round(1)
    })

# Main App
def main():
    st.title("🏃‍♂️ Acute:Chronic Workload Ratio (ACWR) Calculator")
    st.markdown("""
    **ACWR** is a key metric in sports science for monitoring training load and injury risk.
    - **Acute Load**: Recent workload (typically 7 days)
    - **Chronic Load**: Long-term average workload (typically 28 days)
    - **ACWR = Acute Load / Chronic Load**
    """)

    # Sidebar
    st.sidebar.header("📊 Data Input Options")
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Upload CSV File", "Use Sample Data", "Manual Entry"]
    )

    df = None

    if data_source == "Upload CSV File":
        st.sidebar.markdown("### 📁 Upload Your Data")
        st.sidebar.markdown("CSV should contain columns: `Date` and `Training_Load`")

        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV or Excel file",
            type=["csv", "xlsx"],
            help="Upload a CSV or Excel file with Date and Training_Load columns"
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                else:
                    raise ValueError("Unsupported file type")

                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date')
                st.sidebar.success(f"✅ Loaded {len(df)} records")
            except Exception as e:
                st.sidebar.error(f"Error loading file: {e}")

    elif data_source == "Use Sample Data":
        df = create_sample_data()
        st.sidebar.success("✅ Sample data loaded")

    elif data_source == "Manual Entry":
        st.sidebar.markdown("### ✏️ Manual Data Entry")

        # Initialize session state for manual data
        if 'manual_data' not in st.session_state:
            st.session_state.manual_data = []

        col1, col2 = st.sidebar.columns(2)
        with col1:
            entry_date = st.date_input("Date", datetime.now())
        with col2:
            entry_load = st.number_input("Training Load", min_value=0.0, value=100.0)

        if st.sidebar.button("Add Entry"):
            st.session_state.manual_data.append({
                'Date': pd.to_datetime(entry_date),
                'Training_Load': entry_load
            })
            st.sidebar.success("Entry added!")

        if st.sidebar.button("Clear All Data"):
            st.session_state.manual_data = []
            st.sidebar.success("Data cleared!")

        if st.session_state.manual_data:
            df = pd.DataFrame(st.session_state.manual_data)
            df = df.sort_values('Date')
            st.sidebar.success(f"✅ {len(df)} manual entries")

    # Main content
    if df is not None and len(df) > 0:
        # Calculation settings
        st.sidebar.header("⚙️ Calculation Settings")
        acute_days = st.sidebar.slider("Acute Period (days)", 3, 14, 7)
        chronic_days = st.sidebar.slider("Chronic Period (days)", 14, 42, 28)
        method = st.sidebar.selectbox("ACWR Method", ["Rolling Average", "EWMA"])

        # Calculate ACWR
        acwr_values = []
        categories = []
        colors = []

        for i in range(len(df)):
            current_data = df['Training_Load'].iloc[:i+1].values

            if method == "Rolling Average":
                acwr = calculate_rolling_average_acwr(current_data, acute_days, chronic_days)
            else:
                acwr = calculate_ewma_acwr(current_data)

            acwr_values.append(acwr)
            category, color = categorize_acwr(acwr)
            categories.append(category)
            colors.append(color)

        df['ACWR'] = acwr_values
        df['Risk_Category'] = categories
        df['Color'] = colors

        # Display results
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📈 ACWR Timeline")

            # Create interactive plot
            fig = go.Figure()

            # Add training load
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['Training_Load'],
                mode='lines+markers',
                name='Training Load',
                yaxis='y2',
                line=dict(color='lightblue', width=2),
                marker=dict(size=4)
            ))

            # Add ACWR with color coding
            for category in df['Risk_Category'].unique():
                if category == "Insufficient Data":
                    continue
                category_data = df[df['Risk_Category'] == category]
                color_map = {
                    'Undertraining': 'blue',
                    'Optimal': 'green',
                    'Overreaching': 'orange',
                    'High Risk': 'red'
                }

                fig.add_trace(go.Scatter(
                    x=category_data['Date'],
                    y=category_data['ACWR'],
                    mode='markers',
                    name=f'ACWR - {category}',
                    marker=dict(
                        color=color_map.get(category, 'gray'),
                        size=8,
                        line=dict(width=1, color='white')
                    )
                ))

            # Add risk zone lines
            fig.add_hline(y=0.8, line_dash="dash", line_color="blue", 
                         annotation_text="Undertraining threshold")
            fig.add_hline(y=1.3, line_dash="dash", line_color="orange", 
                         annotation_text="Overreaching threshold")
            fig.add_hline(y=1.5, line_dash="dash", line_color="red", 
                         annotation_text="High risk threshold")

            fig.update_layout(
                title="Training Load and ACWR Over Time",
                xaxis_title="Date",
                yaxis_title="ACWR",
                yaxis2=dict(title="Training Load", overlaying='y', side='right'),
                height=500,
                showlegend=True,
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("📊 Current Status")

            # Current ACWR
            current_acwr = df['ACWR'].iloc[-1]
            current_category = df['Risk_Category'].iloc[-1]

            if not pd.isna(current_acwr):
                st.metric(
                    label="Current ACWR",
                    value=f"{current_acwr:.2f}",
                    delta=None
                )

                # Risk indicator
                if current_category == "Optimal":
                    st.success(f"✅ {current_category}")
                elif current_category in ["Undertraining", "Overreaching"]:
                    st.warning(f"⚠️ {current_category}")
                elif current_category == "High Risk":
                    st.error(f"🚨 {current_category}")
                else:
                    st.info(f"ℹ️ {current_category}")
            else:
                st.info("Insufficient data for ACWR calculation")

            # Statistics
            st.subheader("📈 Statistics")
            valid_acwr = df['ACWR'].dropna()
            if len(valid_acwr) > 0:
                st.write(f"**Mean ACWR**: {valid_acwr.mean():.2f}")
                st.write(f"**Max ACWR**: {valid_acwr.max():.2f}")
                st.write(f"**Min ACWR**: {valid_acwr.min():.2f}")

                # Risk distribution
                st.subheader("🎯 Risk Distribution")
                risk_counts = df['Risk_Category'].value_counts()
                for category, count in risk_counts.items():
                    if category != "Insufficient Data":
                        percentage = (count / len(df)) * 100
                        st.write(f"**{category}**: {count} days ({percentage:.1f}%)")

        # Data table
        st.subheader("📋 Data Table")
        display_df = df[['Date', 'Training_Load', 'ACWR', 'Risk_Category']].copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        display_df['ACWR'] = display_df['ACWR'].round(3)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # Download results
        st.subheader("💾 Download Results")
        csv_buffer = io.StringIO()
        display_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"acwr_results_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    else:
        st.info("👆 Please select a data source and upload/enter your training data to begin.")

        # Show information about ACWR
        st.subheader("ℹ️ About ACWR")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Risk Zones:**
            - 🔵 **< 0.8**: Undertraining risk
            - 🟢 **0.8 - 1.3**: Optimal workload
            - 🟠 **1.3 - 1.5**: Overreaching
            - 🔴 **> 1.5**: High injury risk
            """)

        with col2:
            st.markdown("""
            **Calculation Methods:**
            - **Rolling Average**: Traditional method using simple averages
            - **EWMA**: Exponentially weighted moving average (more sensitive to recent changes)
            """)

        st.markdown("""
        **Data Format:**
        Your CSV file should contain two columns:
        - `Date`: Date in YYYY-MM-DD format
        - `Training_Load`: Numerical training load value (e.g., RPE × duration, distance, etc.)
        """)

if __name__ == "__main__":
    main()
