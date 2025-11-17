import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="💰 Personal Finance AI",
    page_icon="💸",
    layout="wide"
)

# -------------------------------
# TITLE
# -------------------------------
st.markdown("<h1 style='text-align:center;'>💰 Personal Finance AI Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#aaa;'>v1.1 — running smooth ⚡</p>", unsafe_allow_html=True)

# -------------------------------
# GOOGLE SHEET CSV LINK
# -------------------------------
CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUAam8s3mtLyaecoGNCda-VJp6_AF4mvFdK8mexO2EUg509LwRttQcda1u6KgM9p5ThwkUv5zhCfn1/pub?gid=0&single=true&output=csv"

# -------------------------------
# LOAD DATA FUNCTION
# -------------------------------
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Convert currency columns to float
    num_cols = ["Beginning_Balance", "Income", "Expenses", "Ending_Balance"]
    for col in num_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .astype(float)
        )

    return df

df = load_data(CSV_URL)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("Inputs 🔧")
selected_view = st.sidebar.selectbox(
    "Dashboard View",
    ["Overview", "Income Trend", "Expenses Trend", "Net Worth Trend"]
)

st.sidebar.write("AI Mode: OFF 🤖")  # fun placeholder

# -------------------------------
# OVERVIEW SECTION
# -------------------------------
if selected_view == "Overview":
    st.subheader("📌 Data Preview")
    st.dataframe(df, use_container_width=True)

    # Basic Stats
    latest_balance = df["Ending_Balance"].iloc[-1]
    first_balance = df["Beginning_Balance"].iloc[0]
    growth = latest_balance - first_balance
    growth_percent = (growth / first_balance) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Net Worth 💸", f"Rs {latest_balance:,.0f}", "+ steady")
    col2.metric("Total Growth 📈", f"Rs {growth:,.0f}")
    col3.metric("Growth % 💹", f"{growth_percent:.2f}%")

    st.markdown("<p style='color:#888;'>Smooth ride so far 🚀</p>", unsafe_allow_html=True)


# -------------------------------
# INCOME TREND
# -------------------------------
elif selected_view == "Income Trend":
    st.subheader("📈 Income Trend (Yo!)")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["Month"], df["Income"], marker="o", color="#00c4ff")
    plt.xticks(rotation=45)
    plt.title("Monthly Income Flow")
    plt.xlabel("Month")
    plt.ylabel("Income")
    st.pyplot(fig)


# -------------------------------
# EXPENSE TREND
# -------------------------------
elif selected_view == "Expenses Trend":
    st.subheader("📉 Expenses Trend (Uh-oh)")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["Month"], df["Expenses"], marker="o", color="#ff4b4b")
    plt.xticks(rotation=45)
    plt.title("Monthly Expense Pattern")
    plt.xlabel("Month")
    plt.ylabel("Expenses")
    st.pyplot(fig)


# -------------------------------
# NET WORTH TREND
# -------------------------------
elif selected_view == "Net Worth Trend":
    st.subheader("💹 Net Worth Trend (King Mode)")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["Month"], df["Ending_Balance"], marker="o", color="#00ff88")
    plt.xticks(rotation=45)
    plt.title("Ending Balance Over Time")
    plt.xlabel("Month")
    plt.ylabel("Net Worth")
    st.pyplot(fig)

    st.markdown("<p style='color:#44ff99;'>Big moves only 💚</p>", unsafe_allow_html=True)
