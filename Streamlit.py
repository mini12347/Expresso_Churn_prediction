import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Expresso Churn Prediction", layout="wide")

with open('LogisticRegression_model.pkl', 'rb') as f:
    model, encoders = pickle.load(f)

df = pd.read_csv('Expresso_churn_dataset.csv')

l = LabelEncoder()

st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to:", ["Churn Prediction", "Dashboard"])

if page == "Churn Prediction":
    st.title('📈 Expresso Churn Prediction')
    st.image('OQKLgVy - Imgur.png', use_container_width=True)
    st.markdown("### Enter customer information below to predict the likelihood of churn:")

    with st.form(key='prediction_form'):
        var = {}
        for i in df.columns:
            if i in ['CHURN','user_id','MRG','ZONE1','ZONE2'] :
                continue
            if df[i].dtype in ['int64', 'float64']:
                var[i] = st.number_input(
                    label=i,
                    min_value=float(df[i].min()),
                    max_value=float(df[i].max()),
                    value=float(round(df[i].mean(), 2))
                )
            else:
                var[i] = st.selectbox(i, df[i].dropna().unique())

        submit_button = st.form_submit_button(label='Predict 🔮')

    if submit_button:
        input_df = pd.DataFrame([var])
        for col, le in encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col])

        prediction_proba = model.predict_proba(input_df)[0, 1]
        prediction = "Likely to Churn" if prediction_proba >= 0.5 else "Not Likely to Churn"

        st.success(f"### 🧾 Prediction: **{prediction}**")
        st.progress(int(prediction_proba * 100))
        st.write(f"**Probability of Churn:** {prediction_proba:.2f}")


if page == "Dashboard":
    st.title("📊 Expresso Churn Analytics Dashboard")
    st.markdown("Explore interactive insights from the Expresso customer churn dataset.")

    # Region filter
    regions = ['All'] + list(df['REGION'].dropna().unique())
    selected_region = st.selectbox('🌍 Select Region', regions)

    if selected_region != "All":
        data = df[df['REGION'] == selected_region].copy()
    else:
        data = df.copy()

    data['CHURN_ENCODED'] = l.fit_transform(data['CHURN'])
    data['CHURN_LABEL'] = data['CHURN_ENCODED'].map({1: 'Churned', 0: 'Not Churned'})

    total = len(data)
    churned = data['CHURN_ENCODED'].sum()
    pct_churn = (churned / total) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("👥 Total Customers", f"{total:,}")
    col2.metric("💔 Churned", f"{churned:,}")
    col3.metric("📉 Churn Rate", f"{pct_churn:.2f}%")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn Distribution")
        pie_data = data['CHURN_LABEL'].value_counts().reset_index()
        pie_data.columns = ['CHURN', 'Count']
        fig_pie = px.pie(
            pie_data,
            names='CHURN',
            values='Count',
            color='CHURN',
            color_discrete_map={'Not Churned': '#2ecc71', 'Churned': '#e74c3c'},
            hole=0.4,
            template='plotly_dark'
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Churn by Region")
        churn_by_region = data.groupby('REGION')['CHURN_ENCODED'].mean().reset_index()
        churn_by_region['CHURN_ENCODED'] *= 100
        fig_region = px.bar(
            churn_by_region,
            x='REGION', y='CHURN_ENCODED',
            labels={'CHURN_ENCODED': '% Churn Rate'},
            color='REGION',
            template='plotly_dark'
        )
        fig_region.update_xaxes(tickangle=45)
        st.plotly_chart(fig_region, use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn by Customer Tenure")
        if 'TENURE' in data.columns:
            fig_tenure = px.histogram(
                data, x='TENURE', color='CHURN_LABEL',
                nbins=20, template='plotly_dark', barmode='overlay'
            )
            st.plotly_chart(fig_tenure, use_container_width=True)

    with col2:
        st.subheader("Churn by Revenue Segment")
        if 'ARPU_SEGMENT' in data.columns:
            fig_arpu = px.bar(
                data.groupby('CHURN_ENCODED')['ARPU_SEGMENT'].mean().reset_index(),
                x='ARPU_SEGMENT', y='ARPU_SEGMENT',
                labels={'ARPU_SEGMENT': 'ARPU Segment', 'CHURN_ENCODED': '% Churn Rate'},
                color='ARPU_SEGMENT', template='plotly_dark'
            )
            st.plotly_chart(fig_arpu, use_container_width=True)

    st.divider()

    st.subheader("Data Volume vs. Churn Probability")
    if 'DATA_VOLUME' in data.columns:
        fig_data = px.scatter(
            data, x='DATA_VOLUME', y='CHURN_ENCODED',
            color='CHURN_LABEL', opacity=0.7,
            labels={'CHURN_ENCODED': 'Churn (1=Yes, 0=No)', 'DATA_VOLUME': 'Data Volume'},
            template='plotly_dark'
        )
        st.plotly_chart(fig_data, use_container_width=True)

    st.caption("📊 Built with Streamlit & Plotly — Expresso Churn Analytics")

