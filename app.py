import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.ensemble import IsolationForest
import requests
import json

# -------------------------------------
# PAGE CONFIG
# -------------------------------------
st.set_page_config(
    page_title="Personal Finance AI v3",
    page_icon="💰",
    layout="wide"
)

st.markdown("<h1 style='text-align:center;'>💰 Personal Finance AI Dashboard — v3</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#bbb;'>ML Forecasts • FIRE • Longevity • Bulk Events • Free AI</p>", unsafe_allow_html=True)


# -------------------------------------
# CSV DATA SOURCE FROM GOOGLE SHEET
# -------------------------------------
CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUAam8s3mtLyaecoGNCda-VJp6_AF4mvFdK8mexO2EUg509LwRttQcda1u6KgM9p5ThwkUv5zhCfn1/pub?gid=0&single=true&output=csv"


# -------------------------------------
# LOAD DATA
# -------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(CSV_URL)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Convert currency columns to float (remove commas)
    for col in ["Beginning_Balance", "Income", "Expenses", "Ending_Balance"]:
        df[col] = (
            df[col].astype(str)
            .str.replace(",", "", regex=False)
            .astype(float)
        )
    return df

df = load_data()


# -------------------------------------
# SIDEBAR INPUTS (Dynamic)
# -------------------------------------
st.sidebar.header("User Inputs 🔧")

monthly_expense = st.sidebar.number_input("Monthly Expense (Today)", value=35000, step=1000)
cagr = st.sidebar.number_input("Expected CAGR %", value=11.5, step=0.5)
inflation = st.sidebar.number_input("Inflation %", value=6.0, step=0.5)
withdraw_rate = st.sidebar.number_input("Withdrawal Rate %", value=5.5, step=0.5)

st.sidebar.markdown("---")

# Bulk event inputs
bulk_income_amount = st.sidebar.number_input("One-time Bulk Income (Rs)", value=0, step=50000)
bulk_income_age = st.sidebar.number_input("Bulk Income Age", value=0, step=1)

bulk_expense_amount = st.sidebar.number_input("One-time Bulk Expense (Rs)", value=0, step=50000)
bulk_expense_age = st.sidebar.number_input("Bulk Expense Age", value=0, step=1)

st.sidebar.markdown("---")

# View Selector
view = st.sidebar.selectbox(
    "Choose Dashboard View",
    [
        "Overview",
        "Income Forecast",
        "Expense Forecast",
        "Net Worth Forecast",
        "Anomaly Detection",
        "FIRE Simulation",
        "AI Insights"
    ]
)


# -------------------------------------
# HELPER: PREPARE PROPHET SERIES
# -------------------------------------
def prep_prophet(series):
    temp = df.copy()
    temp["ds"] = pd.to_datetime(temp["Month"])
    temp["y"] = series
    return temp[["ds", "y"]]


# -------------------------------------
# VIEW: OVERVIEW
# -------------------------------------
if view == "Overview":
    st.subheader("📌 Data Preview")
    st.dataframe(df, use_container_width=True)

    latest = df["Ending_Balance"].iloc[-1]
    start = df["Beginning_Balance"].iloc[0]
    growth = latest - start
    growth_pct = (growth / start) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Net Worth", f"Rs {latest:,.0f}")
    c2.metric("Total Growth", f"Rs {growth:,.0f}")
    c3.metric("Growth %", f"{growth_pct:.2f}%")

    st.markdown("<p style='color:#9f9;'>Steady wealth compounding 🚀</p>", unsafe_allow_html=True)


# -------------------------------------
# VIEW: INCOME FORECAST
# -------------------------------------
if view == "Income Forecast":
    st.subheader("📈 Income Forecast (Next 12 Months)")
    
    data = prep_prophet(df["Income"])
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=12, freq="M")
    forecast = model.predict(future)

    fig = model.plot(forecast)
    st.pyplot(fig)

    st.markdown("💡 Income trend is stable with occasional spikes.")


# -------------------------------------
# VIEW: EXPENSE FORECAST
# -------------------------------------
if view == "Expense Forecast":
    st.subheader("📉 Expense Forecast (Next 12 Months)")

    data = prep_prophet(df["Expenses"])
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=12, freq="M")
    forecast = model.predict(future)

    fig = model.plot(forecast)
    st.pyplot(fig)

    st.markdown("💡 Expenses show some seasonal patterns (travel, big purchases).")


# -------------------------------------
# VIEW: NET WORTH FORECAST
# -------------------------------------
if view == "Net Worth Forecast":
    st.subheader("💹 Net Worth Projection (Next 12 Months)")

    data = prep_prophet(df["Ending_Balance"])
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=12, freq="M")
    forecast = model.predict(future)

    fig = model.plot(forecast)
    st.pyplot(fig)

    st.markdown("💡 Net worth shows strong upward momentum even with dips.")


# -------------------------------------
# VIEW: ANOMALY DETECTION
# -------------------------------------
if view == "Anomaly Detection":
    st.subheader("⚠️ Income & Expense Anomalies")

    iso = IsolationForest(contamination=0.20)
    df["Income_Anomaly"] = iso.fit_predict(df[["Income"]])
    df["Expense_Anomaly"] = iso.fit_predict(df[["Expenses"]])

    anomalies = df[(df["Income_Anomaly"] == -1) | (df["Expense_Anomaly"] == -1)]

    st.write("### 🔍 Detected Anomalies:")
    st.dataframe(anomalies)

    st.markdown("💡 These months showed unusual spikes or drops.")


# -------------------------------------
# PAUSE — PART 2 WILL ADD:
# - Full FIRE Engine
# - Longevity Simulation
# - Bulk Event Application
# - Projection Tables
# - Depletion Charts
# - HuggingFace AI Insights
# -------------------------------------
# -------------------------------------
# FIRE SIMULATION ENGINE
# -------------------------------------
if view == "FIRE Simulation":
    st.subheader("🔥 FIRE Simulation & Longevity Engine")

    current_age = 30
    end_age = 90
    years = list(range(current_age, end_age + 1))

    # Expense inflation
    yearly_expenses = []
    exp = monthly_expense * 12
    for i, age in enumerate(years):
        yearly_expenses.append(exp)
        exp *= (1 + inflation / 100)

    # FIRE number (inflation-adjusted)
    fire_number = yearly_expenses[0] / (withdraw_rate / 100)

    # Prepare projection
    projection = []
    net_worth = df["Ending_Balance"].iloc[-1]

    for i, age in enumerate(years):
        year_expense = yearly_expenses[i]
        growth = net_worth * (cagr / 100)

        # Apply bulk events
        if age == bulk_income_age and bulk_income_amount > 0:
            net_worth += bulk_income_amount

        if age == bulk_expense_age and bulk_expense_amount > 0:
            net_worth -= bulk_expense_amount

        # FIRE check (withdrawals after FIRE age)
        if net_worth > year_expense:
            net_worth = net_worth + growth - year_expense
        else:
            net_worth = net_worth + growth - year_expense

        projection.append({
            "Age": age,
            "Start Balance": net_worth,
            "Growth": growth,
            "Expenses": year_expense,
            "End Balance": net_worth
        })

        if net_worth <= 0:
            break

    proj_df = pd.DataFrame(projection)

    # -------------------------------------
    # FIRE AGE CALCULATION
    # -------------------------------------
    fire_age = None
    for i, row in proj_df.iterrows():
        passive_income = row["Start Balance"] * (withdraw_rate / 100)
        if passive_income >= yearly_expenses[i]:
            fire_age = row["Age"]
            break

    if fire_age is None:
        fire_age = "Not reached by age 90"

    # -------------------------------------
    # LONGEVITY (How long portfolio lasts)
    # -------------------------------------
    portfolio_end_age = proj_df[proj_df["End Balance"] <= 0]
    longevity_age = portfolio_end_age["Age"].iloc[0] if len(portfolio_end_age) > 0 else 90

    # -------------------------------------
    # SUMMARY CARDS
    # -------------------------------------
    c1, c2, c3 = st.columns(3)
    c1.metric("🔥 FIRE Number", f"Rs {fire_number:,.0f}")
    c2.metric("🎯 FIRE Age", f"{fire_age}")
    c3.metric("⏳ Portfolio Lasts Until Age", f"{longevity_age}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # -------------------------------------
    # PROJECTION CHART
    # -------------------------------------
    st.subheader("📈 Wealth Projection")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(proj_df["Age"], proj_df["End Balance"], marker="o", color="#00ff88")
    ax.set_xlabel("Age")
    ax.set_ylabel("Net Worth")
    ax.set_title("Net Worth Projection Over Time")
    st.pyplot(fig)

    # -------------------------------------
    # DEPLETION CHART
    # -------------------------------------
    st.subheader("📉 Portfolio Depletion Curve")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(proj_df["Age"], proj_df["Expenses"], marker="o", color="red", label="Expenses")
    ax2.plot(proj_df["Age"], proj_df["Start Balance"], marker="o", color="green", label="Starting Balance")
    ax2.legend()
    ax2.set_title("Projected Balance vs Expenses")
    st.pyplot(fig2)

    # -------------------------------------
    # YEAR-BY-YEAR TABLE
    # -------------------------------------
    st.subheader("📘 Full Year-by-Year Projection Table")
    st.dataframe(proj_df, use_container_width=True)


# -------------------------------------
# AI INSIGHTS (FREE — HUGGINGFACE)
# -------------------------------------
if view == "AI Insights":
    st.subheader("🤖 Financial AI (Free — HuggingFace)")

    user_q = st.text_area("Ask a question about your finances:")

    if user_q:
        # lightweight HF model
        API_URL = "https://api-inference.huggingface.co/models/google/gemma-2b-it"
        headers = {"Content-Type": "application/json"}

        prompt = f"""
        You are an AI financial analyst. Use the following data and answer the user's question.

        DATA:
        {df.to_string(index=False)}

        QUESTION:
        {user_q}

        Give clear, simple insights.
        """

        payload = {"inputs": prompt}

        with st.spinner("Analyzing with free AI..."):
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
            try:
                ai_text = response.json()[0]["generated_text"]
            except:
                ai_text = "AI model is busy. Try again."

        st.write("### 🔮 AI Insight:")
        st.write(ai_text)

