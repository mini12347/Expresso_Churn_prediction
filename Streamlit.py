import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Expresso Churn Prediction", layout="wide")

with open('LogisticRegression_model.pkl', 'rb') as f:
    model, encoders = pickle.load(f)

df = pd.read_csv('Expresso_churn_dataset.csv.bz2', compression='bz2')

st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Aller à :", ["Prédiction du Churn", "Tableau de Bord"])

if page == "Prédiction du Churn":
    st.title('📈 Prédiction du Churn - Expresso')
    st.image('OQKLgVy - Imgur.png', use_container_width=True)
    st.markdown("### Entrez les informations du client ci-dessous pour prédire la probabilité de churn :")

    with st.form(key='prediction_form'):
        var = {}
        for i in df.columns:
            if i in ['CHURN', 'user_id', 'MRG', 'ZONE1', 'ZONE2']:
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

        submit_button = st.form_submit_button(label='🔮 Prédire')

    if submit_button:
        input_df = pd.DataFrame([var])
        for col, le in encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col])
        prediction_proba = model.predict_proba(input_df)[0, 1]
        prediction = "🔴 Susceptible de résilier" if prediction_proba >= 0.5 else "🟢 Client fidèle"
        st.success(f"### Résultat : **{prediction}**")
        st.progress(int(prediction_proba * 100))
        st.write(f"**Probabilité de churn : {prediction_proba:.2f}**")

if page == "Tableau de Bord":
    st.title("📊 Tableau de Bord - Expresso Churn Analytics")
    st.markdown("Explorez les tendances et indicateurs clés liés au churn des clients.")

    if 'REGION' in df.columns:
        regions = ['Toutes'] + list(df['REGION'].dropna().unique())
        selected_region = st.selectbox('🌍 Sélectionnez une région', regions)
        data = df.copy() if selected_region == "Toutes" else df[df['REGION'] == selected_region]
    else:
        st.warning("La colonne 'REGION' est absente du dataset.")
        data = df.copy()

    if 'CHURN' in data.columns:
        le = LabelEncoder()
        data['CHURN_ENCODED'] = le.fit_transform(data['CHURN'])
        data['CHURN_LABEL'] = data['CHURN_ENCODED'].map({1: 'Churned', 0: 'Not Churned'})
    else:
        st.error("La colonne 'CHURN' est absente du dataset.")
        st.stop()

    total = len(data)
    churned = data['CHURN_ENCODED'].sum()
    pct_churn = (churned / total) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("👥 Total Clients", f"{total:,}")
    col2.metric("💔 Clients perdus", f"{churned:,}")
    col3.metric("📉 Taux de churn", f"{pct_churn:.2f}%")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Répartition du churn")
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
        if 'REGION' in data.columns:
            st.subheader("Churn par région")
            churn_by_region = data.groupby('REGION')['CHURN_ENCODED'].mean().reset_index()
            churn_by_region['CHURN_ENCODED'] *= 100
            fig_region = px.bar(
                churn_by_region,
                x='REGION', y='CHURN_ENCODED',
                labels={'CHURN_ENCODED': '% Churn'},
                color='REGION',
                template='plotly_dark'
            )
            fig_region.update_xaxes(tickangle=45)
            st.plotly_chart(fig_region, use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if 'TENURE' in data.columns:
            st.subheader("Churn selon la durée d'abonnement")
            fig_tenure = px.histogram(
                data, x='TENURE', color='CHURN_LABEL',
                nbins=20, template='plotly_dark', barmode='overlay'
            )
            st.plotly_chart(fig_tenure, use_container_width=True)

    with col2:
        if 'ARPU_SEGMENT' in data.columns:
            st.subheader("Churn par segment de revenu (ARPU)")
            churn_arpu = data.groupby('ARPU_SEGMENT')['CHURN_ENCODED'].mean().reset_index()
            churn_arpu['CHURN_ENCODED'] *= 100
            fig_arpu = px.bar(
                churn_arpu,
                x='ARPU_SEGMENT', y='CHURN_ENCODED',
                labels={'CHURN_ENCODED': '% Churn'},
                color='ARPU_SEGMENT', template='plotly_dark'
            )
            st.plotly_chart(fig_arpu, use_container_width=True)

    st.divider()

    if 'DATA_VOLUME' in data.columns:
        st.subheader("Volume de données vs probabilité de churn")
        fig_data = px.scatter(
            data, x='DATA_VOLUME', y='CHURN_ENCODED',
            color='CHURN_LABEL', opacity=0.7,
            labels={'CHURN_ENCODED': 'Churn (1=Oui, 0=Non)', 'DATA_VOLUME': 'Volume de données'},
            template='plotly_dark'
        )
        st.plotly_chart(fig_data, use_container_width=True)

    st.caption("📊 Réalisé avec Streamlit & Plotly — Expresso Churn Analytics")
