#!/usr/bin/env python3
"""
Análise de Interpretabilidade com SHAP e LIME
TCC - Bacharelado em Física
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import warnings
import os
from pathlib import Path

# --- Configurações Iniciais ---
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

# Paths agnósticos - encontra raiz do projeto
def get_project_root():
    path = Path(__file__).resolve().parent
    for _ in range(5):
        if (path / 'dados' / 'data_treino.csv').exists():
            return path
        if (path / 'dados' / 'processados').is_dir():
            return path
        path = path.parent
    return Path(__file__).resolve().parent.parent

PROJECT_ROOT = get_project_root()
GRAFICOS_DIR = PROJECT_ROOT / 'graficos'
os.makedirs(GRAFICOS_DIR, exist_ok=True)

# --- Funções Auxiliares ---

def carregar_e_preparar_dados():
    """Carrega e prepara os dados, retornando os conjuntos de treino/teste."""
    df = pd.read_csv(PROJECT_ROOT / "dados" / "data_treino.csv")
    
    # Réplica exata do pré-processamento do script principal
    colunas_remover = ["group_id", "id"]
    colunas_atomos = [col for col in df.columns if col.startswith("atoms_") and not col.endswith("size")]
    colunas_remover.extend(colunas_atomos)
    
    y = (df["Tc"] > 0).astype(int)
    
    X = df.drop(columns=["Tc"] + [c for c in colunas_remover if c in df.columns])
    
    threshold = 0.5
    X = X.loc[:, X.isnull().mean() < threshold]
    
    X = X.fillna(X.median())
    
    X = X.select_dtypes(include=[np.number])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, X.columns

def treinar_melhor_modelo(X_train, y_train):
    """Treina o modelo com melhor desempenho (Random Forest)."""
    model = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

# --- Análise SHAP ---

def analise_shap(model, X_train, X_test, y_test):
    """Realiza a análise de interpretabilidade usando SHAP."""
    print("\n--- Iniciando Análise SHAP ---")
    
    # Usar arrays numpy para evitar mismatch de shape (índices do DataFrame)
    X_test_values = np.asarray(X_test)
    feature_names = X_test.columns.tolist()
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_values)
    
    # TreeExplainer retorna lista [class0, class1] para binário; às vezes array (n, feat, 2)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    elif shap_values.ndim == 3 and shap_values.shape[-1] == 2:
        shap_vals = shap_values[:, :, 1]
    else:
        shap_vals = shap_values
    
    # Garantir shape (n_samples, n_features) para alinhar com X_test
    n_samples, n_features = X_test_values.shape
    shap_vals = np.asarray(shap_vals)
    if shap_vals.ndim == 3 and shap_vals.shape[-1] == 2:
        shap_vals = shap_vals[:, :, 1]
    if shap_vals.shape != (n_samples, n_features):
        print(f"  [AVISO] SHAP shape {shap_vals.shape} != X shape ({n_samples},{n_features}); dependência limitada.")
    
    # 1. SHAP Summary Plot (Bar)
    plt.figure()
    shap.summary_plot(shap_vals, X_test_values, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("Importância Global das Features (SHAP)")
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / "12_shap_summary_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Gráfico salvo: 12_shap_summary_bar.png")

    # 2. SHAP Summary Plot (Dot)
    plt.figure()
    shap.summary_plot(shap_vals, X_test_values, feature_names=feature_names, show=False)
    plt.title("Distribuição dos Impactos das Features (SHAP)")
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / "13_shap_summary_dot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Gráfico salvo: 13_shap_summary_dot.png")

    # 3. SHAP Dependence Plots
    importances = np.abs(shap_vals).mean(0)
    if importances.ndim > 1:
        importances = importances.mean(-1)
    top_indices = np.argsort(importances).flatten()[-4:]
    n_top = min(4, len(top_indices))

    for i in range(n_top):
        feature_idx = int(top_indices.flat[i])
        feature_name = feature_names[feature_idx]
        plt.figure()
        if shap_vals.shape == (n_samples, n_features):
            try:
                shap.dependence_plot(feature_idx, shap_vals, X_test_values, feature_names=feature_names, show=False)
            except (ValueError, TypeError) as e:
                print(f"  [AVISO] Dependence plot para '{feature_name}' pulado: {e}")
        else:
            print(f"  [AVISO] Dependence plot pulado (shape SHAP {shap_vals.shape} != X {n_samples}x{n_features})")
        plt.title(f"Gráfico de Dependância SHAP para ‘{feature_name}’")
        plt.tight_layout()
        plt.savefig(GRAFICOS_DIR / f"14_shap_dependence_{i}_{feature_name}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Gráfico salvo: 14_shap_dependence_{i}_{feature_name}.png")

    # 4. SHAP Force Plot
    y_pred = model.predict(X_test_values)
    y_test_arr = np.asarray(y_test).ravel()
    vp_mask = (y_pred == 1) & (y_test_arr == 1)
    fn_mask = (y_pred == 0) & (y_test_arr == 1)
    idx_vp_pos = int(np.where(vp_mask)[0][0]) if vp_mask.any() else 0
    idx_fn_pos = int(np.where(fn_mask)[0][0]) if fn_mask.any() else 0
    
    expected_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    
    plt.figure()
    shap.force_plot(expected_val, shap_vals[idx_vp_pos, :], X_test_values[idx_vp_pos], feature_names=feature_names, matplotlib=True, show=False)
    plt.title(f"Análise SHAP para uma Predição (Verdadeiro Positivo, Amostra {idx_vp_pos})")
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / "15_shap_force_plot_vp.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Gráfico salvo: 15_shap_force_plot_vp.png")

    plt.figure()
    shap.force_plot(expected_val, shap_vals[idx_fn_pos, :], X_test_values[idx_fn_pos], feature_names=feature_names, matplotlib=True, show=False)
    plt.title(f"Análise SHAP para uma Predição (Falso Negativo, Amostra {idx_fn_pos})")
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / "16_shap_force_plot_fn.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Gráfico salvo: 16_shap_force_plot_fn.png")

# --- Análise LIME ---

def analise_lime(model, X_train, X_test, y_train, y_test, feature_names):
    """Realiza a análise de interpretabilidade usando LIME."""
    print("\n--- Iniciando Análise LIME ---")

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=["Não-SC", "SC"],
        mode="classification"
    )

    fn_mask = (model.predict(X_test) == 0) & (np.asarray(y_test).ravel() == 1)
    if not fn_mask.any():
        print("  [AVISO] Nenhum falso negativo encontrado, usando primeira amostra para LIME.")
        idx_fn = X_test.index[0]
    else:
        idx_fn = X_test.index[np.where(fn_mask)[0][0]]
    instance = X_test.loc[idx_fn]

    explanation = explainer.explain_instance(
        data_row=instance.values,
        predict_fn=model.predict_proba,
        num_features=10
    )

    fig = explanation.as_pyplot_figure()
    fig.suptitle(f"Explicação LIME para uma Predição (Falso Negativo, Amostra {idx_fn})")
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / "17_lime_explanation_fn.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Gráfico salvo: 17_lime_explanation_fn.png")

# --- Função Principal ---

def main():
    """Função principal para executar as análises."""
    print("=" * 80)
    print("INICIANDO ANÁLISE DE INTERPRETABILIDADE (SHAP & LIME)")
    print("=" * 80)

    X_train, X_test, y_train, y_test, feature_names = carregar_e_preparar_dados()
    model = treinar_melhor_modelo(X_train, y_train)

    analise_shap(model, X_train, X_test, y_test)
    analise_lime(model, X_train, X_test, y_train, y_test, feature_names)

    print("\n" + "=" * 80)
    print("ANÁLISE DE INTERPRETABILIDADE FINALIZADA!")
    print("=" * 80)

if __name__ == "__main__":
    main()
