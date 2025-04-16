
# app_streamlit_predicao.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

# Título da aplicação
st.title("Predição de Bioacessibilidade com IA")
st.markdown("Insira os dados abaixo para prever a **bioacessibilidade de compostos fenólicos** em ingredientes vegetais.")

# Entradas do usuário
proteina = st.number_input("Proteína (%)", min_value=0.0, max_value=100.0, value=3.0)
fibras = st.number_input("Fibras (%)", min_value=0.0, max_value=100.0, value=5.0)
fenolicos = st.number_input("Fenólicos Totais (mg GAE/g)", min_value=0.0, max_value=100.0, value=10.0)
ph = st.number_input("pH", min_value=1.0, max_value=14.0, value=6.5)
cor_L = st.number_input("Cor L*", min_value=0.0, max_value=100.0, value=70.0)
cor_a = st.number_input("Cor a*", min_value=-100.0, max_value=100.0, value=5.0)
cor_b = st.number_input("Cor b*", min_value=-100.0, max_value=100.0, value=15.0)
tratamento = st.selectbox("Tipo de Pré-Tratamento", ["Nenhum", "Micro-ondas", "Ultrassom", "Fermentação"])

# Mapear tratamento para valores numéricos (simples)
tratamento_dict = {"Nenhum": 0, "Micro-ondas": 1, "Ultrassom": 2, "Fermentação": 3}
tratamento_cod = tratamento_dict[tratamento]

# Botão de predição
if st.button("Prever Bioacessibilidade"):
    entrada = pd.DataFrame({
        'proteina': [proteina],
        'fibras': [fibras],
        'fenolicos': [fenolicos],
        'pH': [ph],
        'cor_L': [cor_L],
        'cor_a': [cor_a],
        'cor_b': [cor_b],
        'tratamento': [tratamento_cod]
    })

    if not os.path.exists("modelo_bioacessibilidade.pkl"):
        st.error("❌ O arquivo 'modelo_bioacessibilidade.pkl' **não foi encontrado** na pasta do projeto.")
        st.info("💡 Verifique se o arquivo foi enviado corretamente para o GitHub ou está na mesma pasta do script.")
    else:
        try:
            modelo = joblib.load("modelo_bioacessibilidade.pkl")
            pred = modelo.predict(entrada)[0]
            st.success(f"✅ Bioacessibilidade prevista: **{pred:.2f}%**")
        except Exception as e:
            st.error(f"Erro ao carregar o modelo ou realizar a predição: {e}")
