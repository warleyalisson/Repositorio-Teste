
# app_streamlit_predicao_embutido.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Título da aplicação
st.title("Predição de Bioacessibilidade com IA (Modelo Integrado)")
st.markdown("Versão de teste com modelo embutido para prever a **bioacessibilidade de compostos fenólicos**.")

# Entradas do usuário
proteina = st.number_input("Proteína (%)", min_value=0.0, max_value=100.0, value=3.0)
fibras = st.number_input("Fibras (%)", min_value=0.0, max_value=100.0, value=5.0)
fenolicos = st.number_input("Fenólicos Totais (mg GAE/g)", min_value=0.0, max_value=100.0, value=10.0)
ph = st.number_input("pH", min_value=1.0, max_value=14.0, value=6.5)
cor_L = st.number_input("Cor L*", min_value=0.0, max_value=100.0, value=70.0)
cor_a = st.number_input("Cor a*", min_value=-100.0, max_value=100.0, value=5.0)
cor_b = st.number_input("Cor b*", min_value=-100.0, max_value=100.0, value=15.0)
tratamento = st.selectbox("Tipo de Pré-Tratamento", ["Nenhum", "Micro-ondas", "Ultrassom", "Fermentação"])
tratamento_dict = {"Nenhum": 0, "Micro-ondas": 1, "Ultrassom": 2, "Fermentação": 3}
tratamento_cod = tratamento_dict[tratamento]

# Criar entrada
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

# Treinar modelo interno simples (dados simulados)
def gerar_modelo_teste():
    np.random.seed(42)
    n_samples = 100
    X = pd.DataFrame({
        'proteina': np.random.uniform(1, 10, n_samples),
        'fibras': np.random.uniform(2, 15, n_samples),
        'fenolicos': np.random.uniform(5, 40, n_samples),
        'pH': np.random.uniform(3.5, 7.5, n_samples),
        'cor_L': np.random.uniform(30, 90, n_samples),
        'cor_a': np.random.uniform(-5, 10, n_samples),
        'cor_b': np.random.uniform(0, 30, n_samples),
        'tratamento': np.random.choice([0, 1, 2, 3], n_samples)
    })
    y = (
        0.8 * X["fenolicos"]
        + 0.3 * X["proteina"]
        - 0.2 * X["fibras"]
        + np.random.normal(0, 3, n_samples)
    )
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X, y)
    return modelo

# Criar modelo embutido
modelo_embutido = gerar_modelo_teste()

# Botão de previsão
if st.button("Prever Bioacessibilidade"):
    pred = modelo_embutido.predict(entrada)[0]
    st.success(f"✅ Bioacessibilidade prevista (modelo de teste): **{pred:.2f}%**")
