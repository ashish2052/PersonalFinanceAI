import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Personal Finance AI Dashboard",
    page_icon="💰",
    layout="wide"
)

# --- CSV LINK ---
CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUAam8s3mtLyaecoGNCda-VJp6_AF4mvFdK8mexO2EUg509LwRttQcda1u6KgM9p5ThwkUv5zhCfn1/pub?gid=0&single=true&output=csv"


# --- LOAD DATA ---
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df

df = load_data(CSV_URL)

st.title("💰 Personal Finance AI Dashboard")
st.write("Version 1 — Data loaded successfully")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Inputs")
selected_view = st.sidebar.selectbox(
    "Choose Dashboard View",
    ["Overview", "Income Trend", "Expenses Trend", "Net Worth Trend"]
)

# --- OVERVIEW SECTION ---
if selected_view == "Overview":
    st.subheader("📌 Data Preview")
    st.dataframe(df)

    # Basic Stats
    latest_balance = df["Ending_Balance"].iloc[-1]
    first_balance = df["Beginning_Balance"].iloc[0]
    growth = latest_balance - first_balance
    growth_percent = (growth / first_balance) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Net Worth", f"Rs {latest_balance:,.0f}")
    col2.metric("Lifetime Growth", f"Rs {growth:,.0f}")
    col3.metric("Growth %", f"{growth_percent:.2f}%")

# --- INCOME TREND ---
elif selected_view == "Income Trend":
    st.subheader("📈 Income Trend")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Month"], df["Income"], marker="o")
    plt.xticks(rotation=45)
    plt.title("Monthly Income Trend")
    plt.xlabel("Month")
    plt.ylabel("Income")
    st.pyplot(fig)

# --- EXPENSES TREND ---
elif selected_view == "Expenses Trend":
    st.subheader("📉 Expenses Trend")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Month"], df["Expenses"], marker="o", color="red")
    plt.xticks(rotation=45)
    plt.title("Monthly Expense Trend")
    plt.xlabel("Month")
    plt.ylabel("Expenses")
    st.pyplot(fig)

# --- NET WORTH TREND ---
elif selected_view == "Net Worth Trend":
    st.subheader("💹 Net Worth Trend")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Month"], df["Ending_Balance"], marker="o", color="green")
    plt.xticks(rotation=45)
    plt.title("Ending Balance Over Time")
    plt.xlabel("Month")
    plt.ylabel("Net Worth")
    st.pyplot(fig)


