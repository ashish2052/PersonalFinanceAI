# ======================================================
# Personal Finance AI v4.2 — Full App (Soft Neon Edition)
# ML Forecasting + FIRE Engine + AI Insights + Neon Cards
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.ensemble import IsolationForest
import requests, json
from streamlit_autorefresh import st_autorefresh

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

</style>
""", unsafe_allow_html=True)

st.title("💰 Personal Finance AI v4.2 — Soft Neon Edition")


# --------------------------------------
# SIDEBAR — INPUTS
# --------------------------------------
st.sidebar.header("🔗 DATA SOURCE")

CSV_URL = st.sidebar.text_input(
    "Paste your Google Sheet CSV URL",
    placeholder="https://docs.google.com/spreadsheets/..."
)

if CSV_URL.strip() == "":
    st.warning("Paste your CSV link to load dashboard.")
    st.stop()


# --------------------------------------
# LOAD DATA (with caching)
# --------------------------------------
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    numeric_cols = ["Beginning_Balance", "Income", "Expenses", "Ending_Balance"]
    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(",", "", regex=False).astype(float)

    return df

df = load_data(CSV_URL)


# --------------------------------------
# REFRESH MECHANISMS
# --------------------------------------
if st.sidebar.button("🔄 Force Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()

st_autorefresh(interval=30000, key="auto_refresh_30s")


# --------------------------------------
# USER SETTINGS
# --------------------------------------
st.sidebar.header("⚙️ Settings")

monthly_expense = st.sidebar.number_input("Monthly Expense (Now)", value=35000)
cagr = st.sidebar.number_input("CAGR %", value=11.5)
inflation = st.sidebar.number_input("Inflation %", value=6.0)
withdraw_rate = st.sidebar.number_input("Withdrawal %", value=5.5)
retire_age = st.sidebar.slider("Retirement Age", 35, 55, 40)

# Bulk events
bulk_income = st.sidebar.number_input("Bulk Income", value=0)
bulk_income_age = st.sidebar.number_input("Bulk Income Age", value=0)
bulk_expense = st.sidebar.number_input("Bulk Expense", value=0)
bulk_expense_age = st.sidebar.number_input("Bulk Expense Age", value=0)


# --------------------------------------
# VIEW SELECTOR
# --------------------------------------
view = st.sidebar.selectbox(
    "Choose View",
    ["Overview", "Forecast", "Anomaly Detection", "FIRE Engine", "AI Insights"]
)


# --------------------------------------
# PROPHET FORECAST HELPER
# --------------------------------------
def prophet_monthly(series):
    temp = df.copy()
    temp["ds"] = pd.to_datetime(temp["Month"])
    temp["y"] = series

    model = Prophet()
    model.fit(temp[["ds", "y"]])

    future = model.make_future_dataframe(periods=12, freq="M")
    return model.predict(future)
# ======================================================
# VIEW: OVERVIEW
# ======================================================
if view == "Overview":

    st.subheader("📘 Data Overview")
    st.dataframe(df, use_container_width=True)

    latest = df["Ending_Balance"].iloc[-1]
    start = df["Beginning_Balance"].iloc[0]
    growth = latest - start

    c1, c2 = st.columns(2)
    c1.metric("💰 Current Net Worth", f"Rs {latest:,.0f}")
    c2.metric("📈 Total Growth", f"Rs {growth:,.0f}")

    st.markdown("---")

    st.subheader("📊 Income vs Expenses")

    fig = plt.figure(figsize=(8,4))
    plt.plot(df["Month"], df["Income"], label="Income", color="#00eaff")
    plt.plot(df["Month"], df["Expenses"], label="Expenses", color="#ff6b6b")
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)




# ======================================================
# VIEW: INCOME FORECAST (ML PROPHET)
# ======================================================
if view == "Forecast":

    st.subheader("🔮 Income Forecast — Prophet ML Model (Next 12 Months)")

    forecast_df = prophet_monthly(df["Income"])

    fig = plt.figure(figsize=(10,4))
    plt.plot(forecast_df["ds"], forecast_df["yhat"], color="#00eaff")
    plt.title("Predicted Monthly Income")
    st.pyplot(fig)

    st.dataframe(forecast_df[["ds", "yhat"]].tail(12), use_container_width=True)




# ======================================================
# VIEW: EXPENSE / INCOME ANOMALY DETECTION
# ======================================================
if view == "Anomaly Detection":

    st.subheader("⚠️ Anomaly Detection — Isolation Forest")

    iso = IsolationForest(contamination=0.15, random_state=42)

    df["Income_Anomaly"] = iso.fit_predict(df[["Income"]])
    df["Expense_Anomaly"] = iso.fit_predict(df[["Expenses"]])

    anomalies = df[(df["Income_Anomaly"] == -1) | 
                   (df["Expense_Anomaly"] == -1)]

    st.warning("Showing months where Income or Expenses were significantly abnormal.")
    st.dataframe(anomalies, use_container_width=True)
# ======================================================
# VIEW: FIRE ENGINE v4.2 — ML + SOFT NEON UI
# ======================================================
if view == "FIRE Engine":

    # ----------- STYLING FOR NEON CARDS -----------
    st.markdown("""
        <style>
            .neon-card {
                background: rgba(10, 10, 30, 0.75);
                padding: 18px;
                border-radius: 15px;
                box-shadow: 0 0 18px #008cff, inset 0 0 10px #0033cc;
                text-align: center;
                color: #d9ecff;
                border: 1px solid #00aaff;
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

    st.subheader("🔥 FIRE Engine v4.2 — ML + Soft Neon UI")

    current_age = 30
    end_age = 90
    age_range = list(range(current_age, end_age + 1))

    # ----------- WORKING EXPENSES (INFLATING) -----------
    yearly_exp_working = []
    exp = monthly_expense * 12
    for _ in age_range:
        yearly_exp_working.append(exp)
        exp *= (1 + inflation/100)

    # ----------- RETIREMENT EXPENSES (SEPARATE) -----------
    retire_monthly_exp = st.sidebar.number_input(
        "Retirement Monthly Expense (Today's Value)",
        value=120000
    )

    yearly_exp_retire = []
    exp_r = retire_monthly_exp * 12
    for _ in age_range:
        yearly_exp_retire.append(exp_r)
        exp_r *= (1 + inflation/100)

    # ----------- ML INCOME PROJECTION -----------
    forecast_df = prophet_monthly(df["Income"])
    income_fc_list = list(forecast_df["yhat"])

    yearly_income = []
    idx = 0
    for age in age_range:
        if age <= retire_age:
            if idx + 12 <= len(income_fc_list):
                yr = sum(income_fc_list[idx:idx+12])
                yearly_income.append(max(0, yr))
                idx += 12
            else:
                yearly_income.append(income_fc_list[-1] * 12)
        else:
            yearly_income.append(0)

    # ----------- FIRE SIMULATION -----------
    networth = df["Ending_Balance"].iloc[-1]
    projection = []

    for i, age in enumerate(age_range):

        exp = yearly_exp_working[i] if age <= retire_age else yearly_exp_retire[i]
        inc = yearly_income[i]
        savings = max(0, inc - exp) if age <= retire_age else 0
        growth = networth * (cagr/100)

        # Bulk events
        if age == bulk_income_age:
            networth += bulk_income
        if age == bulk_expense_age:
            networth -= bulk_expense

        if age <= retire_age:
            networth = networth + growth + savings
        else:
            networth = networth + growth - exp

        projection.append([age, inc, exp, savings, growth, networth])

        if networth <= 0:
            break

    proj_df = pd.DataFrame(
        projection,
        columns=["Age","Income","Expenses","Savings","Growth","NetWorth"]
    )

    # ----------- KEY METRICS -----------
    earliest_fire_age = None
    for _, row in proj_df.iterrows():
        passive = row["NetWorth"] * (withdraw_rate/100)
        if passive >= row["Expenses"]:
            earliest_fire_age = int(row["Age"])
            break

    if earliest_fire_age is None:
        earliest_fire_age = "Not Achieved"

    # Net worth at retirement
    try:
        retirement_nw = proj_df.loc[proj_df["Age"] == retire_age, "NetWorth"].values[0]
    except:
        retirement_nw = proj_df.iloc[-1]["NetWorth"]

    # Net worth at Age 90
    final_nw = proj_df.iloc[-1]["NetWorth"]

    # Present Value of final corpus
    years_to_90 = proj_df.iloc[-1]["Age"] - current_age
    pv_final = final_nw / ((1 + inflation/100) ** years_to_90)

    # ----------- DISPLAY NEON CARDS -----------
    st.subheader("⭐ Key Retirement Metrics")

    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(f"""
        <div class="neon-card">
            <div class="neon-label">Earliest FIRE Age</div>
            <div class="neon-value">{earliest_fire_age}</div>
        </div>
    """, unsafe_allow_html=True)

    c2.markdown(f"""
        <div class="neon-card">
            <div class="neon-label">Portfolio at Retirement (Age {retire_age})</div>
            <div class="neon-value">Rs {retirement_nw:,.0f}</div>
        </div>
    """, unsafe_allow_html=True)

    c3.markdown(f"""
        <div class="neon-card">
            <div class="neon-label">Portfolio at Age 90</div>
            <div class="neon-value">Rs {final_nw:,.0f}</div>
        </div>
    """, unsafe_allow_html=True)

    c4.markdown(f"""
        <div class="neon-card">
            <div class="neon-label">Present Value of Age 90 Corpus</div>
            <div class="neon-value">Rs {pv_final:,.0f}</div>
        </div>
    """, unsafe_allow_html=True)

    # ----------- CHARTS -----------
    st.subheader("📈 Net Worth Projection")
    fig1 = plt.figure(figsize=(10,4))
    plt.plot(proj_df["Age"], proj_df["NetWorth"], color="#00eaff")
    st.pyplot(fig1)

    st.subheader("📉 Retirement & Depletion Curve")
    fig2 = plt.figure(figsize=(10,4))
    plt.plot(proj_df["Age"], proj_df["Expenses"], color="red", label="Expenses")
    plt.plot(proj_df["Age"], proj_df["NetWorth"], color="cyan", label="Net Worth")
    plt.legend()
    st.pyplot(fig2)

    st.subheader("📘 Full Projection Table")
    st.dataframe(proj_df, use_container_width=True)



# ======================================================
# VIEW: AI INSIGHTS (HUGGINGFACE • FREE)
# ======================================================
if view == "AI Insights":

    st.subheader("🤖 AI Insights (Free HuggingFace Model)")
    q = st.text_area("Ask a financial question:")

    if q:
        API_URL = "https://api-inference.huggingface.co/models/google/gemma-2-2b-it"
        headers = {"Content-Type": "application/json"}

        prompt = f"""
You are a financial analyst AI.

DATASET:
{df.to_string(index=False)}

QUESTION:
{q}

Give a short, clear, helpful insight based only on the data.
"""

        payload = {"inputs": prompt}

        with st.spinner("Analyzing..."):
            r = requests.post(API_URL, headers=headers, data=json.dumps(payload))

            try:
                ans = r.json()[0]["generated_text"]
            except:
                ans = "Model is busy. Try again."

        st.write("### 🔮 Insight:")
        st.write(ans)
