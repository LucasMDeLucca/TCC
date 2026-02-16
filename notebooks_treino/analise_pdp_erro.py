#!/usr/bin/env python3
"""
Análise de Partial Dependence Plots (PDP) e Análise de Erro
TCC - Bacharelado em Física
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import confusion_matrix
import warnings
import os
from pathlib import Path

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

def carregar_e_preparar_dados():
    """Carrega e prepara os dados."""
    df = pd.read_csv(PROJECT_ROOT / "dados" / "data_treino.csv")
    colunas_remover = ["group_id", "id"]
    colunas_atomos = [col for col in df.columns if col.startswith("atoms_") and not col.endswith("size")]
    colunas_remover.extend(colunas_atomos)
    y = (df["Tc"] > 0).astype(int)
    X = df.drop(columns=["Tc"] + [c for c in colunas_remover if c in df.columns])
    X = X.loc[:, X.isnull().mean() < 0.5]
    X = X.fillna(X.median())
    X = X.select_dtypes(include=[np.number])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, df

def treinar_modelo(X_train, y_train):
    """Treina o modelo Random Forest."""
    model = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def gerar_pdp(model, X_train):
    """Gera Partial Dependence Plots para as features mais importantes."""
    print("\n--- Gerando Partial Dependence Plots ---")
    
    importances = model.feature_importances_
    top_features_indices = np.argsort(importances)[-4:][::-1]
    top_features_names = X_train.columns[top_features_indices].tolist()
    
    fig, ax = plt.subplots(figsize=(14, 10))
    PartialDependenceDisplay.from_estimator(
        model, X_train, features=top_features_names, 
        kind="average", ax=ax, grid_resolution=50
    )
    fig.suptitle("Partial Dependence Plots para as 4 Features Mais Importantes", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(GRAFICOS_DIR / "18_partial_dependence_plots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Gráfico salvo: 18_partial_dependence_plots.png")

def analise_de_erro(model, X_test, y_test, df_original):
    """Realiza uma análise detalhada dos erros do modelo."""
    print("\n--- Realizando Análise de Erro ---")
    
    y_pred = model.predict(X_test)
    
    # Identificar Falsos Positivos e Falsos Negativos
    fp_indices = X_test[(y_pred == 1) & (y_test == 0)].index
    fn_indices = X_test[(y_pred == 0) & (y_test == 1)].index
    
    print(f"Número de Falsos Positivos (FP): {len(fp_indices)}")
    print(f"Número de Falsos Negativos (FN): {len(fn_indices)}")
    
    # Analisar a distribuição de Tc para os Falsos Negativos (supercondutores que o modelo errou)
    tc_fn = df_original.loc[fn_indices, "Tc"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histograma de Tc dos Falsos Negativos
    axes[0].hist(tc_fn, bins=30, color='coral', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel("Temperatura Crítica (K)")
    axes[0].set_ylabel("Frequência")
    axes[0].set_title("Distribuição de $T_c$ para Falsos Negativos")
    axes[0].axvline(tc_fn.median(), color='red', linestyle='--', label=f'Mediana: {tc_fn.median():.2f} K')
    axes[0].legend()
    
    # Comparar a distribuição de uma feature importante entre acertos e erros
    # Feature: sc_DOSs_mean
    feature_name = 'sc_DOSs_mean'
    if feature_name in X_test.columns:
        correct_indices = X_test[(y_pred == y_test)].index
        incorrect_indices = X_test[(y_pred != y_test)].index
        
        data_correct = X_test.loc[correct_indices, feature_name]
        data_incorrect = X_test.loc[incorrect_indices, feature_name]
        
        axes[1].hist(data_correct, bins=30, alpha=0.6, label='Predições Corretas', color='green', edgecolor='black')
        axes[1].hist(data_incorrect, bins=30, alpha=0.6, label='Predições Incorretas', color='red', edgecolor='black')
        axes[1].set_xlabel(f'{feature_name}')
        axes[1].set_ylabel("Frequência")
        axes[1].set_title(f"Distribuição de '{feature_name}' por Tipo de Predição")
        axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / "19_analise_erro.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Gráfico salvo: 19_analise_erro.png")
    
    # Salvar resumo da análise de erro
    with open(GRAFICOS_DIR / "resumo_analise_erro.txt", "w") as f:
        f.write("=== RESUMO DA ANÁLISE DE ERRO ===\n\n")
        f.write(f"Total de amostras no conjunto de teste: {len(y_test)}\n")
        f.write(f"Número de Falsos Positivos (FP): {len(fp_indices)}\n")
        f.write(f"Número de Falsos Negativos (FN): {len(fn_indices)}\n\n")
        f.write("--- Análise dos Falsos Negativos ---\n")
        f.write(f"Mediana de Tc: {tc_fn.median():.2f} K\n")
        f.write(f"Média de Tc: {tc_fn.mean():.2f} K\n")
        f.write(f"Desvio Padrão de Tc: {tc_fn.std():.2f} K\n")
        f.write(f"Tc Mínimo: {tc_fn.min():.2f} K\n")
        f.write(f"Tc Máximo: {tc_fn.max():.2f} K\n\n")
        f.write("Interpretação: Os falsos negativos tendem a ser supercondutores com Tc mais baixo, \n")
        f.write("o que pode indicar que os descritores não capturam bem as características de \n")
        f.write("supercondutores convencionais de baixa temperatura.\n")
    print("Resumo salvo: resumo_analise_erro.txt")

def main():
    """Função principal."""
    print("=" * 80)
    print("ANÁLISE DE PDP E ERRO")
    print("=" * 80)
    
    X_train, X_test, y_train, y_test, df_original = carregar_e_preparar_dados()
    model = treinar_modelo(X_train, y_train)
    
    gerar_pdp(model, X_train)
    analise_de_erro(model, X_test, y_test, df_original)
    
    print("\n" + "=" * 80)
    print("ANÁLISE FINALIZADA!")
    print("=" * 80)

if __name__ == "__main__":
    main()
