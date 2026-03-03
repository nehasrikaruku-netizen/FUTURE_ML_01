import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

st.set_page_config(page_title="Sales Forecasting", page_icon="📊", layout="wide")
st.title("📊 Sales & Demand Forecasting Dashboard")
st.markdown("Business Intelligence & Time-Series Forecasting")

st.sidebar.header("Configuration")
file = st.sidebar.file_uploader("Upload CSV", type="csv")

if file:
    df = pd.read_csv(file)
    st.sidebar.success("Data loaded!")
    
    with st.expander("Data Overview"):
        st.metric("Records", len(df))
        st.metric("Columns", len(df.columns))
        st.dataframe(df.head(), use_container_width=True)
    
    cols = df.columns.tolist()
    if len(cols) < 2:
        st.error("Need at least 2 columns")
        st.stop()
    
    date_col = st.sidebar.selectbox("Date Column", cols)
    sales_col = st.sidebar.selectbox("Sales Column", [c for c in cols if c != date_col])
    
    try:
        data = df[[date_col, sales_col]].copy()
        data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
        data[sales_col] = pd.to_numeric(data[sales_col], errors='coerce')
        data = data.dropna().sort_values(date_col).reset_index(drop=True)
        
        if len(data) < 2:
            st.error("Not enough valid data")
            st.stop()
        
        pdata = data[[date_col, sales_col]].copy()
        pdata.columns = ['ds', 'y']
        
        st.sidebar.success(f"Cleaned: {len(pdata)} records")
        
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
    
    st.header("Sales Analysis")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(pdata['ds'], pdata['y'], linewidth=2.5, color='#1f77b4')
        ax.fill_between(pdata['ds'], pdata['y'], alpha=0.25, color='#1f77b4')
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales ($)")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        st.metric("Total", f"${float(pdata['y'].sum()):,.0f}")
        st.metric("Avg", f"${float(pdata['y'].mean()):,.0f}")
        st.metric("Max", f"${float(pdata['y'].max()):,.0f}")
        st.metric("Min", f"${float(pdata['y'].min()):,.0f}")
    
    st.subheader("Statistics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Std Dev", f"${float(pdata['y'].std()):,.0f}")
    c2.metric("Median", f"${float(pdata['y'].median()):,.0f}")
    c3.metric("Q1", f"${float(pdata['y'].quantile(0.25)):,.0f}")
    c4.metric("Q3", f"${float(pdata['y'].quantile(0.75)):,.0f}")
    c5.metric("IQR", f"${float(pdata['y'].quantile(0.75) - pdata['y'].quantile(0.25)):,.0f}")
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(pdata['y'], bins=50, color='#2ca02c', edgecolor='black', alpha=0.7)
        ax.set_xlabel("Sales ($)")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.boxplot(pdata['y'], vert=True, patch_artist=True)
        ax.set_ylabel("Sales ($)")
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    
    st.header("Forecasting")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        periods = st.slider("Months", 1, 24, 12)
    with c2:
        mode = st.selectbox("Seasonality", ["additive", "multiplicative"])
    with c3:
        conf = st.slider("Confidence", 0.80, 0.99, 0.95, 0.01)
    
    st.write("Training model...")
    
    try:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode=mode,
            interval_width=conf
        )
        
        with st.spinner("Fitting..."):
            model.fit(pdata)
        
        future = model.make_future_dataframe(periods=periods, freq='MS')
        fcst = model.predict(future)
        
        st.success("Model trained!")
        
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
    
    st.subheader("Forecast")
    
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(pdata['ds'], pdata['y'], 'b-', label='Historical', linewidth=2)
    
    fcst_future = fcst[fcst['ds'] > pdata['ds'].max()].copy()
    ax.plot(fcst_future['ds'], fcst_future['yhat'], 'r-', label='Forecast', linewidth=2)
    ax.fill_between(fcst_future['ds'], fcst_future['yhat_lower'], fcst_future['yhat_upper'], alpha=0.2, color='red')
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    
    st.subheader("Forecast Table")
    
    ftable = fcst[fcst['ds'] > pdata['ds'].max()][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    ftable.columns = ['Date', 'Forecast', 'Lower', 'Upper']
    ftable['Date'] = ftable['Date'].dt.strftime('%Y-%m-%d')
    for col in ['Forecast', 'Lower', 'Upper']:
        ftable[col] = ftable[col].apply(lambda x: f"${x:,.0f}")
    st.dataframe(ftable, use_container_width=True, hide_index=True)
    
    st.subheader("Components")
    
    try:
        fig = model.plot_components(fcst)
        st.pyplot(fig, use_container_width=True)
    except:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(fcst['ds'], fcst['trend'], color='#1f77b4', linewidth=2)
        ax.set_ylabel('Trend')
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    
    st.header("Diagnostics")
    
    try:
        ftrain = fcst[fcst['ds'] <= pdata['ds'].max()].copy()
        merged = pdata.merge(ftrain[['ds', 'yhat']], on='ds', how='inner')
        
        if len(merged) > 0:
            act = merged['y'].values
            pred = merged['yhat'].values
            
            mae = mean_absolute_error(act, pred)
            rmse = np.sqrt(mean_squared_error(act, pred))
            mape = np.mean(np.abs((act - pred) / act)) * 100 if np.all(act != 0) else 0
            ss_res = np.sum((act - pred) ** 2)
            ss_tot = np.sum((act - np.mean(act)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MAE", f"${mae:,.0f}")
            c2.metric("RMSE", f"${rmse:,.0f}")
            c3.metric("MAPE", f"{mape:.2f}%")
            c4.metric("R² Score", f"{r2:.4f}")
            
            st.subheader("Residuals")
            res = act - pred
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(merged['ds'], res, marker='o', color='#d62728', markersize=4)
                ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
                ax.set_xlabel("Date")
                ax.set_ylabel("Residuals ($)")
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(res, bins=40, color='#9467bd', edgecolor='black', alpha=0.7)
                ax.set_xlabel("Residuals ($)")
                ax.set_ylabel("Frequency")
                ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
                ax.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error: {e}")
    
    st.header("Summary")
    
    try:
        fcst_data = fcst[fcst['ds'] > pdata['ds'].max()].copy()
        if len(fcst_data) > 0:
            avg_fcst = float(fcst_data['yhat'].mean())
            total_fcst = float(fcst_data['yhat'].sum())
            hist_avg = float(pdata['y'].mean())
            growth = ((avg_fcst / hist_avg) - 1) * 100
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg Forecast", f"${avg_fcst:,.0f}")
            c2.metric("Total Forecast", f"${total_fcst:,.0f}")
            c3.metric("Growth", f"{growth:+.2f}%")
            c4.metric("Period", f"{periods} months")
    except:
        pass
    
    st.header("Insights")
    
    try:
        days = (pdata['ds'].max() - pdata['ds'].min()).days
        first30 = pdata['y'].iloc[:min(30, len(pdata))].mean()
        last30 = pdata['y'].iloc[max(0, len(pdata)-30):].mean()
        trend = "Up" if last30 > first30 else "Down"
        hist_avg = float(pdata['y'].mean())
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"Period: {days} days | Avg: ${hist_avg:,.0f}")
        with c2:
            st.markdown(f"Algorithm: Prophet | Seasonality: {mode}")
    except:
        pass
    
    st.header("Download")
    
    c1, c2 = st.columns(2)
    try:
        with c1:
            exp = fcst[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            exp.columns = ['Date', 'Forecast', 'Lower', 'Upper']
            csv = exp.to_csv(index=False)
            st.download_button("Forecast", csv, "forecast.csv", "text/csv")
        
        with c2:
            if 'merged' in locals():
                exp = merged[['ds', 'y', 'yhat']].copy()
                exp.columns = ['Date', 'Actual', 'Predicted']
                csv = exp.to_csv(index=False)
                st.download_button("Predictions", csv, "predictions.csv", "text/csv")
    except:
        pass

else:
    st.info("Upload a CSV with sales data")
    sample = pd.DataFrame({'Date': pd.date_range('2023-01-01', periods=5), 'Sales': [10000, 12000, 11500, 13000, 14200]})
    st.dataframe(sample, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("<div style='text-align: center; font-size: 11px;'>Streamlit | Prophet | ML</div>", unsafe_allow_html=True)