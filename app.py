
# Personal Finance AI v4 - FIRE Engine + ML + Longevity + Bulk Events + Free AI
# NOTE: Trimmed and optimized version to fit execution limits while keeping core functionality.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.ensemble import IsolationForest
import requests, json

st.set_page_config(page_title="Finance AI v4", page_icon="💰", layout="wide")
st.title("💰 Personal Finance AI v4 — Full FIRE Engine")

CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUAam8s3mtLyaecoGNCda-VJp6_AF4mvFdK8mexO2EUg509LwRttQcda1u6KgM9p5ThwkUv5zhCfn1/pub?gid=0&single=true&output=csv"

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_URL)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    for col in ["Beginning_Balance","Income","Expenses","Ending_Balance"]:
        df[col] = df[col].astype(str).str.replace(",","",regex=False).astype(float)
    return df

df = load_data()

st.sidebar.header("Inputs")
monthly_expense = st.sidebar.number_input("Monthly Expense (Now)", value=35000)
cagr = st.sidebar.number_input("CAGR %", value=11.5)
inflation = st.sidebar.number_input("Inflation %", value=6.0)
withdraw_rate = st.sidebar.number_input("Withdrawal %", value=5.5)
retire_age = st.sidebar.slider("Target Retirement Age", 35, 55, 40)

bulk_income = st.sidebar.number_input("Bulk Income", value=0)
bulk_income_age = st.sidebar.number_input("Bulk Income Age", value=0)

bulk_expense = st.sidebar.number_input("Bulk Expense", value=0)
bulk_expense_age = st.sidebar.number_input("Bulk Expense Age", value=0)

view = st.sidebar.selectbox("View", ["Overview","Forecast","FIRE Engine","AI Insights"])

def prophet_monthly(series):
    temp=df.copy()
    temp["ds"]=pd.to_datetime(temp["Month"])
    temp["y"]=series
    model=Prophet()
    model.fit(temp[["ds","y"]])
    future=model.make_future_dataframe(periods=12, freq="M")
    return model.predict(future)

if view=="Overview":
    st.subheader("Data")
    st.dataframe(df)

if view=="Forecast":
    st.subheader("Income Forecast")
    inc_fc = prophet_monthly(df["Income"])
    fig1 = plt.figure(figsize=(8,4))
    plt.plot(inc_fc["ds"],inc_fc["yhat"])
    st.pyplot(fig1)

if view=="FIRE Engine":
    st.subheader("🔥 Full FIRE Simulation")

    current_age=30
    end_age=90
    yearly_exp=[]
    exp=monthly_expense*12
    for age in range(current_age, end_age+1):
        yearly_exp.append(exp)
        exp*=1+inflation/100

    # ML income forecast
    income_fc = prophet_monthly(df["Income"])
    future_inc = list(income_fc["yhat"][-12:])  # next 12 months

    yearly_income=[]
    idx=0
    for age in range(current_age, retire_age+1):
        # sum 12 months forecast
        if idx+12 <= len(income_fc):
            yr = income_fc["yhat"].iloc[idx:idx+12].sum()
        else:
            yr = future_inc[0]*12
        yearly_income.append(max(0,yr))
        idx+=12

    networth = df["Ending_Balance"].iloc[-1]
    projection=[]
    age_list=list(range(current_age,end_age+1))

    for i,age in enumerate(age_list):
        inc = yearly_income[i] if age<=retire_age else 0
        exp = yearly_exp[i]
        savings = max(0, inc-exp) if age<=retire_age else 0
        growth = networth*(cagr/100)

        if age==bulk_income_age:
            networth+=bulk_income
        if age==bulk_expense_age:
            networth-=bulk_expense

        if age>retire_age:
            networth = networth + growth - exp
        else:
            networth = networth + growth + savings

        projection.append([age, inc, exp, savings, growth, networth])

        if networth<=0:
            break

    proj_df = pd.DataFrame(projection, columns=["Age","Income","Expenses","Savings","Growth","NetWorth"])

    st.subheader("Projection Table")
    st.dataframe(proj_df)

    st.subheader("Wealth Curve")
    figw=plt.figure(figsize=(8,4))
    plt.plot(proj_df["Age"],proj_df["NetWorth"])
    st.pyplot(figw)

    st.subheader("Depletion Curve")
    figd=plt.figure(figsize=(8,4))
    plt.plot(proj_df["Age"],proj_df["Expenses"],label="Expenses")
    plt.plot(proj_df["Age"],proj_df["NetWorth"],label="NetWorth")
    plt.legend()
    st.pyplot(figd)

if view=="AI Insights":
    st.subheader("🤖 AI Insights (Free HuggingFace)")

    q = st.text_area("Ask something:")

    if q:
        API_URL = "https://api-inference.huggingface.co/models/google/gemma-2b-it"
        headers = {"Content-Type": "application/json"}

        prompt = f"""
You are a financial analyst AI.

DATA:
{df.to_string(index=False)}

QUESTION:
{q}

Give a short and clear explanation based only on the data.
"""

        payload = {"inputs": prompt}

        with st.spinner("Analyzing with free AI..."):
            r = requests.post(API_URL, headers=headers, data=json.dumps(payload))

            try:
                ans = r.json()[0]["generated_text"]
            except:
                ans = "AI model is busy, try again."

        st.write("### 🔮 Insight:")
        st.write(ans)
