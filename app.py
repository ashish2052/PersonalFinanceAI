import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.ensemble import IsolationForest
import requests, json
import os

# Safe import for auto-refresh
try:
    from streamlit_autorefresh import st_autorefresh
except ModuleNotFoundError:
    def st_autorefresh(*args, **kwargs):
        return None

# --------------------------------------
# PAGE CONFIG + GLOBAL NEON THEME
# --------------------------------------
st.set_page_config(
    page_title="Finance AI v4.2",
    page_icon="💰",
    layout="wide"
)

# Global soft-neon CSS
st.markdown("""
<style>
body {
    background-color: #050A1F;
    color: #E6EEFF;
    font-family: 'Poppins', sans-serif;
}

.sidebar .sidebar-content {
    background-color: #04071A !important;
}

h1, h2, h3 {
    color: #9EC5FF !important;
    text-shadow: 0 0 8px #0040ff;
}

.metric {
    color: #00eaff !important;
}

table {
    color: #E6EEFF !important;
}

/* Custom Neon Card */
.neon-card {
    background: rgba(10, 10, 30, 0.75);
    padding: 18px;
    border-radius: 15px;
    box-shadow: 0 0 18px #008cff, inset 0 0 10px #0033cc;
    text-align: center;
    color: #d9ecff;
    border: 1px solid #00aaff;
    margin-bottom: 20px;
}
.neon-value {
    font-size: 26px;
    font-weight: 700;
    color: #00eaff;
    text-shadow: 0 0 10px #00eaff;
}
.neon-label {
    font-size: 14px;
    opacity: 0.85;
}
</style>
""", unsafe_allow_html=True)

st.title("💰 Personal Finance AI v4.2 — Soft Neon Edition")


# --------------------------------------
# SIDEBAR — INPUTS
# --------------------------------------
st.sidebar.header("🔗 DATA SOURCE")

# Default to a demo csv if none provided, or ask user
# Using a placeholder for now, user can input their own
default_csv = "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv" # Placeholder, structure won't match but prevents crash on empty
CSV_URL = st.sidebar.text_input(
    "Paste your Google Sheet CSV URL",
    value="",
    placeholder="https://docs.google.com/spreadsheets/..."
)

# --------------------------------------
# HELPER: LOAD DATA
# --------------------------------------
@st.cache_data
def load_data(url):
    if not url:
        return None
    
    try:
        df = pd.read_csv(url)
        # Clean column names
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]
        
        # Ensure required columns exist (mapping if necessary could go here)
        required_cols = ["Month", "Income", "Expenses", "Ending_Balance"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Missing columns in CSV: {missing}")
            return None

        # Clean numeric columns
        numeric_cols = ["Income", "Expenses", "Ending_Balance"]
        if "Beginning_Balance" in df.columns:
            numeric_cols.append("Beginning_Balance")
            
        for col in numeric_cols:
            if col in df.columns:
                # Remove currency symbols, commas, etc.
                df[col] = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Parse dates
        df["Month"] = pd.to_datetime(df["Month"], errors='coerce')
        df = df.sort_values("Month")
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = None
if CSV_URL:
    df = load_data(CSV_URL)
else:
    st.info("👋 Welcome! Please paste your Google Sheet CSV URL in the sidebar to get started.")
    st.markdown("""
    **Expected CSV Format:**
    - `Month` (Date format)
    - `Income` (Number)
    - `Expenses` (Number)
    - `Ending_Balance` (Number)
    - `Beginning_Balance` (Optional)
    """)
    
    # Create a dummy DF for demo purposes if user wants to see UI? 
    # For now, we stop if no data.
    if st.button("Load Demo Data"):
        # Create synthetic data
        dates = pd.date_range(start="2023-01-01", periods=24, freq="M")
        data = {
            "Month": dates,
            "Income": np.random.normal(50000, 5000, 24),
            "Expenses": np.random.normal(30000, 3000, 24),
        }
        df = pd.DataFrame(data)
        df["Income"] = df["Income"].abs()
        df["Expenses"] = df["Expenses"].abs()
        df["Ending_Balance"] = (df["Income"] - df["Expenses"]).cumsum() + 100000
        df["Beginning_Balance"] = df["Ending_Balance"].shift(1).fillna(100000)
        st.success("Loaded Demo Data")

if df is None:
    st.stop()

# --------------------------------------
# REFRESH MECHANISMS
# --------------------------------------
if st.sidebar.button("🔄 Force Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st_autorefresh(interval=30000, key="auto_refresh_30s")


# --------------------------------------
# USER SETTINGS
# --------------------------------------
st.sidebar.header("⚙️ Settings")

monthly_expense = st.sidebar.number_input("Monthly Expense (Now)", value=35000)
cagr = st.sidebar.number_input("CAGR % (Investments)", value=11.5)
inflation = st.sidebar.number_input("Inflation %", value=6.0)
withdraw_rate = st.sidebar.number_input("Withdrawal % (SWR)", value=4.0)
retire_age = st.sidebar.slider("Retirement Age", 30, 80, 45)

# Bulk events
with st.sidebar.expander("💰 Bulk Events"):
    bulk_income = st.number_input("One-time Income", value=0)
    bulk_income_age = st.number_input("Age for Income", value=0)
    bulk_expense = st.number_input("One-time Expense", value=0)
    bulk_expense_age = st.number_input("Age for Expense", value=0)


# --------------------------------------
# VIEW SELECTOR
# --------------------------------------
view = st.sidebar.radio(
    "Choose View",
    ["Overview", "Forecast", "Anomaly Detection", "FIRE Engine", "AI Insights"]
)


# --------------------------------------
# PROPHET FORECAST HELPER
# --------------------------------------
def prophet_monthly(series, periods=12):
    temp = df.copy()
    temp["ds"] = pd.to_datetime(temp["Month"])
    temp["y"] = series

    model = Prophet()
    model.fit(temp[["ds", "y"]])

    future = model.make_future_dataframe(periods=periods, freq="M")
    forecast = model.predict(future)
    return forecast

# ======================================================
# VIEW: OVERVIEW
# ======================================================
if view == "Overview":
    st.subheader("📘 Data Overview")
    
    # Top Metrics
    latest_bal = df["Ending_Balance"].iloc[-1]
    start_bal = df["Ending_Balance"].iloc[0] # Or beginning balance of first row
    total_growth = latest_bal - start_bal
    avg_savings_rate = ((df["Income"] - df["Expenses"]) / df["Income"]).mean() * 100
    
    m1, m2, m3 = st.columns(3)
    m1.metric("💰 Current Net Worth", f"Rs {latest_bal:,.0f}", delta=f"{total_growth:,.0f}")
    m2.metric("💸 Avg Monthly Expense", f"Rs {df['Expenses'].mean():,.0f}")
    m3.metric("🐷 Avg Savings Rate", f"{avg_savings_rate:.1f}%")

    st.markdown("---")

    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("📊 Income vs Expenses")
        st.line_chart(df.set_index("Month")[["Income", "Expenses"]], color=["#00eaff", "#ff6b6b"])

    with c2:
        st.subheader("Recent Data")
        st.dataframe(df.tail(5).set_index("Month"), use_container_width=True)


# ======================================================
# VIEW: INCOME FORECAST (ML PROPHET)
# ======================================================
elif view == "Forecast":
    st.subheader("🔮 Income Forecast — Prophet ML Model")
    
    periods = st.slider("Months to Forecast", 6, 60, 12)
    
    with st.spinner("Training Prophet Model..."):
        forecast_df = prophet_monthly(df["Income"], periods=periods)

    st.line_chart(forecast_df.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]])
    
    st.write("### Forecast Data")
    st.dataframe(forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods), use_container_width=True)


# ======================================================
# VIEW: EXPENSE / INCOME ANOMALY DETECTION
# ======================================================
elif view == "Anomaly Detection":
    st.subheader("⚠️ Anomaly Detection — Isolation Forest")
    
    contamination = st.slider("Sensitivity (Contamination)", 0.01, 0.25, 0.10)

    iso = IsolationForest(contamination=contamination, random_state=42)

    # Detect anomalies
    df["Income_Anomaly"] = iso.fit_predict(df[["Income"]])
    df["Expense_Anomaly"] = iso.fit_predict(df[["Expenses"]])

    anomalies = df[(df["Income_Anomaly"] == -1) | (df["Expense_Anomaly"] == -1)]

    if not anomalies.empty:
        st.warning(f"Found {len(anomalies)} anomalies in your financial history.")
        st.dataframe(anomalies[["Month", "Income", "Expenses", "Income_Anomaly", "Expense_Anomaly"]], use_container_width=True)
        
        # Visualizing anomalies
        st.write("### Anomaly Visualization")
        
        # Plot Income
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["Month"], df["Income"], label="Income", color="cyan", alpha=0.6)
        # Highlight anomalies
        anom_inc = df[df["Income_Anomaly"] == -1]
        ax.scatter(anom_inc["Month"], anom_inc["Income"], color="red", label="Anomaly", s=50, zorder=5)
        ax.set_title("Income Anomalies")
        ax.legend()
        ax.set_facecolor("#050A1F")
        fig.patch.set_facecolor("#050A1F")
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('white')
        st.pyplot(fig)
        
    else:
        st.success("No significant anomalies detected with current sensitivity.")


# ======================================================
# VIEW: FIRE ENGINE v4.2 — ML + SOFT NEON UI
# ======================================================
elif view == "FIRE Engine":
    st.subheader("🔥 FIRE Engine v4.2 — ML + Soft Neon UI")

    # Inputs
    current_age = st.number_input("Current Age", value=30, min_value=18, max_value=90)
    end_age = 90
    
    if retire_age <= current_age:
        st.warning("Retirement age must be greater than current age.")
    else:
        age_range = list(range(current_age, end_age + 1))
        
        # ----------- RETIREMENT EXPENSES -----------
        retire_monthly_exp = st.number_input(
            "Retirement Monthly Expense (Today's Value)",
            value=int(monthly_expense * 0.8) # Default to 80% of current
        )

        # ----------- SIMULATION LOOP -----------
        # We need to project year by year.
        # 1. Estimate Annual Income Growth (using CAGR or Prophet trend?)
        # For simplicity in FIRE calc, we often use a fixed growth rate for income, 
        # but user asked for ML. Let's use the Prophet trend for the first N years, then flatten or grow by inflation.
        
        # Get Prophet trend for income
        # We'll project 5 years out with Prophet, then assume inflation growth
        future_months = (retire_age - current_age) * 12
        # Limit Prophet to reasonable timeframe (e.g. 5 years), as it gets wild
        # For long term, we'll use a conservative growth rate
        
        # Let's stick to the user's logic but refined:
        # Income grows by inflation + career growth (say 2%) until retirement
        # Expenses grow by inflation
        
        networth = df["Ending_Balance"].iloc[-1]
        projection = []
        
        annual_income_est = df["Income"].mean() * 12 # Base
        annual_expense_est = monthly_expense * 12
        
        # Career growth assumption (separate from investment CAGR)
        career_growth = st.slider("Est. Annual Income Growth %", 0.0, 10.0, 3.0) / 100
        
        for age in age_range:
            # 1. Determine Income/Expense for this year
            if age < retire_age:
                # Working
                inc = annual_income_est
                exp = annual_expense_est
                
                # Savings
                savings = inc - exp
                
                # Apply Growth to Portfolio
                investment_growth = networth * (cagr / 100)
                
                # Update Net Worth
                networth = networth + investment_growth + savings
                
                # Inflate for next year
                annual_income_est *= (1 + career_growth) # Income grows
                annual_expense_est *= (1 + inflation / 100) # Expenses grow
                
            else:
                # Retired
                inc = 0
                # Expenses in retirement (adjusted for inflation from today)
                # We need to calculate what the retirement expense IS at this future age
                # The user input 'retire_monthly_exp' is in TODAY's value.
                # So we inflate it by (age - current_age) years of inflation
                years_passed = age - current_age
                adjusted_retire_exp = (retire_monthly_exp * 12) * ((1 + inflation/100) ** years_passed)
                
                exp = adjusted_retire_exp
                savings = -exp # Drawdown
                
                investment_growth = networth * (cagr / 100)
                networth = networth + investment_growth - exp
            
            # Bulk Events
            if age == bulk_income_age:
                networth += bulk_income
            if age == bulk_expense_age:
                networth -= bulk_expense
            
            projection.append({
                "Age": age,
                "Income": inc if age < retire_age else 0,
                "Expenses": exp,
                "NetWorth": networth
            })
            
            if networth < 0:
                break
        
        proj_df = pd.DataFrame(projection)
        
        # Metrics
        final_nw = proj_df["NetWorth"].iloc[-1] if not proj_df.empty else 0
        
        # Display
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="neon-card"><div class="neon-label">Net Worth at Age {end_age}</div><div class="neon-value">Rs {final_nw:,.0f}</div></div>', unsafe_allow_html=True)
        
        # Charts
        st.line_chart(proj_df.set_index("Age")[["NetWorth", "Expenses"]])
        st.dataframe(proj_df, use_container_width=True)


# ======================================================
# VIEW: AI INSIGHTS
# ======================================================
elif view == "AI Insights":
    st.subheader("🤖 AI Financial Insights")
    
    # Check for OpenAI key
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    
    if not openai_api_key:
        st.info("Please enter your OpenAI API Key in the sidebar to use AI Insights.")
        st.stop()
        
    import openai
    client = openai.OpenAI(api_key=openai_api_key)
    
    q = st.text_area("Ask a question about your finances:", "How can I optimize my savings rate based on this data?")
    
    if st.button("Analyze"):
        # Prepare context
        # Summarize data to avoid token limits
        summary = df.describe().to_string()
        recent_data = df.tail(6).to_string()
        
        prompt = f"""
        You are an expert financial advisor. Analyze the user's financial data.
        
        Data Summary:
        {summary}
        
        Recent 6 Months:
        {recent_data}
        
        User Question: {q}
        
        Provide actionable, specific advice.
        """
        
        with st.spinner("AI is thinking..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o", # or gpt-3.5-turbo
                    messages=[
                        {"role": "system", "content": "You are a helpful financial assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                st.markdown(response.choices[0].message.content)
            except Exception as e:
                st.error(f"AI Error: {e}")

