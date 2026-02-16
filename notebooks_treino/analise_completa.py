#!/usr/bin/env python3
"""
Análise Completa de Modelos de Machine Learning para Classificação de Supercondutores
TCC - Bacharelado em Física
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve, average_precision_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

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

# Configuração de estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

def carregar_dados():
    """Carrega e prepara os dados para análise."""
    print("=" * 80)
    print("CARREGANDO DADOS")
    print("=" * 80)
    
    df = pd.read_csv(PROJECT_ROOT / 'dados' / 'data_treino.csv')
    print(f"Dataset carregado: {df.shape[0]} amostras, {df.shape[1]} features")
    
    # Remover colunas não numéricas e identificadores
    colunas_remover = ['group_id', 'id']
    colunas_atomos = [col for col in df.columns if col.startswith('atoms_') and not col.endswith('size')]
    colunas_remover.extend(colunas_atomos)
    
    # Separar target
    y = df['Tc'].copy()
    
    # Criar variável binária: supercondutor (Tc > 0) vs não-supercondutor (Tc = 0 ou NaN)
    y_binary = (y > 0).astype(int)
    
    # Remover colunas não necessárias
    X = df.drop(columns=['Tc'] + [c for c in colunas_remover if c in df.columns])
    
    # Remover colunas com muitos valores faltantes
    threshold = 0.5
    X = X.loc[:, X.isnull().mean() < threshold]
    
    # Preencher valores faltantes com a mediana
    X = X.fillna(X.median())
    
    # Manter apenas colunas numéricas
    X = X.select_dtypes(include=[np.number])
    
    print(f"Features após pré-processamento: {X.shape[1]}")
    print(f"Distribuição das classes:")
    print(f"  - Supercondutores (Tc > 0): {y_binary.sum()} ({100*y_binary.mean():.1f}%)")
    print(f"  - Não-supercondutores: {len(y_binary) - y_binary.sum()} ({100*(1-y_binary.mean()):.1f}%)")
    
    return X, y_binary, y

def analise_exploratoria(X, y_binary, y):
    """Realiza análise exploratória dos dados."""
    print("\n" + "=" * 80)
    print("ANÁLISE EXPLORATÓRIA")
    print("=" * 80)
    
    # 1. Distribuição de Tc
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histograma de Tc (apenas valores > 0)
    y_positivo = y[y > 0]
    if len(y_positivo) == 0:
        axes[0].text(0.5, 0.5, 'Sem dados com Tc > 0', ha='center', va='center', transform=axes[0].transAxes)
    else:
        axes[0].hist(y_positivo, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(y_positivo.median(), color='red', linestyle='--', label=f'Mediana: {y_positivo.median():.2f} K')
        axes[0].legend()
    axes[0].set_xlabel('Temperatura Crítica (K)')
    axes[0].set_ylabel('Frequência')
    axes[0].set_title('Distribuição da Temperatura Crítica (Tc > 0)')
    
    # Distribuição das classes
    classes = ['Não-Supercondutor', 'Supercondutor']
    contagens = [len(y_binary) - y_binary.sum(), y_binary.sum()]
    colors = ['#ff6b6b', '#4ecdc4']
    axes[1].bar(classes, contagens, color=colors, edgecolor='black')
    axes[1].set_ylabel('Número de Amostras')
    axes[1].set_title('Distribuição das Classes')
    for i, v in enumerate(contagens):
        axes[1].text(i, v + 20, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / '01_distribuicao_tc.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Gráfico salvo: 01_distribuicao_tc.png")
    
    # 2. Correlação entre features principais
    # Selecionar features mais importantes
    features_importantes = ['fermi_lens_size', 'fermi_line_size', 'fermi_line_mean', 'fermi_line_std',
                           'sc_DOSs_mean', 'sc_DOSs_std', 'sc_bands_mean', 'sc_bands_std']
    features_disponiveis = [f for f in features_importantes if f in X.columns]
    
    if len(features_disponiveis) > 3:
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = X[features_disponiveis].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                    square=True, linewidths=0.5, ax=ax, fmt='.2f')
        ax.set_title('Matriz de Correlação - Features Físicas Principais')
        plt.tight_layout()
        plt.savefig(GRAFICOS_DIR / '02_correlacao_features.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Gráfico salvo: 02_correlacao_features.png")
    
    # 3. Boxplot de features por classe
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    features_plot = ['fermi_line_mean', 'sc_DOSs_mean', 'sc_bands_std', 'fermi_lens_size']
    features_plot = [f for f in features_plot if f in X.columns]
    
    for i, feat in enumerate(features_plot[:4]):
        ax = axes[i // 2, i % 2]
        data_plot = pd.DataFrame({'Feature': X[feat], 'Classe': y_binary.map({0: 'Não-SC', 1: 'SC'})})
        sns.boxplot(x='Classe', y='Feature', data=data_plot, ax=ax, palette=['#ff6b6b', '#4ecdc4'])
        ax.set_title(f'Distribuição de {feat}')
        ax.set_ylabel(feat)
    
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / '03_boxplot_features.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Gráfico salvo: 03_boxplot_features.png")

def treinar_modelos(X, y):
    """Treina todos os modelos e retorna métricas."""
    print("\n" + "=" * 80)
    print("TREINAMENTO DOS MODELOS")
    print("=" * 80)
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                         random_state=42, stratify=y)
    
    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Definir modelos
    modelos = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, 
                                                 min_samples_split=5, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                      random_state=42, use_label_encoder=False, 
                                      eval_metric='logloss', n_jobs=-1),
        'LightGBM': lgb.LGBMClassifier(n_estimators=200, max_depth=10, learning_rate=0.1,
                                        random_state=42, verbose=-1, n_jobs=-1),
        'SVM (RBF)': SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
        'MLP': MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu',
                             max_iter=500, random_state=42, early_stopping=True),
        'LASSO (L1)': LogisticRegression(penalty='l1', solver='saga', C=1.0, 
                                          max_iter=1000, random_state=42)
    }
    
    resultados = {}
    predicoes = {}
    probabilidades = {}
    
    for nome, modelo in modelos.items():
        print(f"\nTreinando {nome}...")
        
        # Usar dados escalados para SVM, MLP e LASSO
        if nome in ['SVM (RBF)', 'MLP', 'LASSO (L1)']:
            modelo.fit(X_train_scaled, y_train)
            y_pred = modelo.predict(X_test_scaled)
            y_prob = modelo.predict_proba(X_test_scaled)[:, 1]
        else:
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            y_prob = modelo.predict_proba(X_test)[:, 1]
        
        # Calcular métricas
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        resultados[nome] = {
            'Acurácia': acc,
            'Precisão': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        }
        
        predicoes[nome] = y_pred
        probabilidades[nome] = y_prob
        
        print(f"  Acurácia: {acc:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {auc:.4f}")
    
    return resultados, predicoes, probabilidades, y_test, modelos, X_train, X_test, scaler

def gerar_graficos_resultados(resultados, predicoes, probabilidades, y_test):
    """Gera gráficos comparativos dos resultados."""
    print("\n" + "=" * 80)
    print("GERANDO GRÁFICOS DE RESULTADOS")
    print("=" * 80)
    
    # 1. Comparação de métricas
    df_resultados = pd.DataFrame(resultados).T
    
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(df_resultados))
    width = 0.15
    
    metricas = ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'ROC-AUC']
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
    
    for i, (metrica, cor) in enumerate(zip(metricas, colors)):
        ax.bar(x + i*width, df_resultados[metrica], width, label=metrica, color=cor, edgecolor='black')
    
    ax.set_xlabel('Modelo')
    ax.set_ylabel('Score')
    ax.set_title('Comparação de Métricas por Modelo')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df_resultados.index, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Baseline 80%')
    
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / '04_comparacao_metricas.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Gráfico salvo: 04_comparacao_metricas.png")
    
    # 2. Curvas ROC
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(probabilidades)))
    
    for (nome, prob), cor in zip(probabilidades.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        ax.plot(fpr, tpr, color=cor, lw=2, label=f'{nome} (AUC = {auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Classificador Aleatório')
    ax.set_xlabel('Taxa de Falsos Positivos (FPR)')
    ax.set_ylabel('Taxa de Verdadeiros Positivos (TPR)')
    ax.set_title('Curvas ROC - Comparação de Modelos')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / '05_curvas_roc.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Gráfico salvo: 05_curvas_roc.png")
    
    # 3. Curvas Precision-Recall
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for (nome, prob), cor in zip(probabilidades.items(), colors):
        precision, recall, _ = precision_recall_curve(y_test, prob)
        ap = average_precision_score(y_test, prob)
        ax.plot(recall, precision, color=cor, lw=2, label=f'{nome} (AP = {ap:.3f})')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Curvas Precision-Recall - Comparação de Modelos')
    ax.legend(loc='lower left')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / '06_curvas_precision_recall.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Gráfico salvo: 06_curvas_precision_recall.png")
    
    # 4. Matrizes de Confusão
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (nome, pred) in enumerate(predicoes.items()):
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Não-SC', 'SC'], yticklabels=['Não-SC', 'SC'])
        axes[i].set_title(f'{nome}')
        axes[i].set_xlabel('Predito')
        axes[i].set_ylabel('Real')
    
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / '07_matrizes_confusao.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Gráfico salvo: 07_matrizes_confusao.png")
    
    # 5. Ranking de modelos
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_ranking = df_resultados.sort_values('F1-Score', ascending=True)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(df_ranking)))
    
    bars = ax.barh(df_ranking.index, df_ranking['F1-Score'], color=colors, edgecolor='black')
    ax.set_xlabel('F1-Score')
    ax.set_title('Ranking de Modelos por F1-Score')
    ax.set_xlim(0, 1)
    
    for bar, val in zip(bars, df_ranking['F1-Score']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.4f}', 
                va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / '08_ranking_modelos.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Gráfico salvo: 08_ranking_modelos.png")

def analise_importancia_features(modelos, X_train, feature_names):
    """Analisa a importância das features para modelos baseados em árvore."""
    print("\n" + "=" * 80)
    print("ANÁLISE DE IMPORTÂNCIA DE FEATURES")
    print("=" * 80)
    
    # Random Forest
    rf = modelos['Random Forest']
    importancias_rf = pd.DataFrame({
        'Feature': feature_names,
        'Importância': rf.feature_importances_
    }).sort_values('Importância', ascending=False).head(20)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Random Forest
    axes[0].barh(importancias_rf['Feature'][::-1], importancias_rf['Importância'][::-1], 
                 color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Importância')
    axes[0].set_title('Top 20 Features - Random Forest')
    
    # XGBoost
    xgb_model = modelos['XGBoost']
    importancias_xgb = pd.DataFrame({
        'Feature': feature_names,
        'Importância': xgb_model.feature_importances_
    }).sort_values('Importância', ascending=False).head(20)
    
    axes[1].barh(importancias_xgb['Feature'][::-1], importancias_xgb['Importância'][::-1], 
                 color='forestgreen', edgecolor='black')
    axes[1].set_xlabel('Importância')
    axes[1].set_title('Top 20 Features - XGBoost')
    
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / '09_importancia_features.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Gráfico salvo: 09_importancia_features.png")
    
    # LightGBM
    lgb_model = modelos['LightGBM']
    importancias_lgb = pd.DataFrame({
        'Feature': feature_names,
        'Importância': lgb_model.feature_importances_
    }).sort_values('Importância', ascending=False).head(20)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(importancias_lgb['Feature'][::-1], importancias_lgb['Importância'][::-1], 
            color='darkorange', edgecolor='black')
    ax.set_xlabel('Importância')
    ax.set_title('Top 20 Features - LightGBM')
    
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / '10_importancia_lightgbm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Gráfico salvo: 10_importancia_lightgbm.png")
    
    return importancias_rf, importancias_xgb, importancias_lgb

def validacao_cruzada(X, y, modelos):
    """Realiza validação cruzada para todos os modelos."""
    print("\n" + "=" * 80)
    print("VALIDAÇÃO CRUZADA (5-Fold)")
    print("=" * 80)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    resultados_cv = {}
    
    for nome in ['Random Forest', 'XGBoost', 'LightGBM', 'SVM (RBF)', 'MLP', 'LASSO (L1)']:
        modelo = modelos[nome]
        
        if nome in ['SVM (RBF)', 'MLP', 'LASSO (L1)']:
            scores = cross_val_score(modelo, X_scaled, y, cv=cv, scoring='f1')
        else:
            scores = cross_val_score(modelo, X, y, cv=cv, scoring='f1')
        
        resultados_cv[nome] = {
            'Média': scores.mean(),
            'Desvio Padrão': scores.std(),
            'Scores': scores
        }
        
        print(f"{nome}: F1 = {scores.mean():.4f} ± {scores.std():.4f}")
    
    # Gráfico de validação cruzada
    fig, ax = plt.subplots(figsize=(12, 6))
    
    nomes = list(resultados_cv.keys())
    medias = [resultados_cv[n]['Média'] for n in nomes]
    stds = [resultados_cv[n]['Desvio Padrão'] for n in nomes]
    
    x = np.arange(len(nomes))
    colors = plt.cm.Set2(np.linspace(0, 1, len(nomes)))
    
    bars = ax.bar(x, medias, yerr=stds, capsize=5, color=colors, edgecolor='black')
    ax.set_xlabel('Modelo')
    ax.set_ylabel('F1-Score')
    ax.set_title('Validação Cruzada (5-Fold) - F1-Score')
    ax.set_xticks(x)
    ax.set_xticklabels(nomes, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    
    for bar, media, std in zip(bars, medias, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f'{media:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / '11_validacao_cruzada.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Gráfico salvo: 11_validacao_cruzada.png")
    
    return resultados_cv

def salvar_resultados(resultados, resultados_cv, importancias_rf):
    """Salva os resultados em arquivos."""
    print("\n" + "=" * 80)
    print("SALVANDO RESULTADOS")
    print("=" * 80)
    
    # DataFrame de resultados
    df_resultados = pd.DataFrame(resultados).T
    df_resultados.to_csv(GRAFICOS_DIR / 'resultados_modelos.csv')
    print("Resultados salvos: resultados_modelos.csv")
    
    # Importância de features
    importancias_rf.to_csv(GRAFICOS_DIR / 'importancia_features_rf.csv', index=False)
    print("Importância de features salva: importancia_features_rf.csv")
    
    # Resumo em texto
    with open(GRAFICOS_DIR / 'resumo_resultados.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RESUMO DOS RESULTADOS - CLASSIFICAÇÃO DE SUPERCONDUTORES\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("MÉTRICAS NO CONJUNTO DE TESTE:\n")
        f.write("-" * 40 + "\n")
        for nome, metricas in resultados.items():
            f.write(f"\n{nome}:\n")
            for metrica, valor in metricas.items():
                f.write(f"  {metrica}: {valor:.4f}\n")
        
        f.write("\n\nVALIDAÇÃO CRUZADA (5-Fold):\n")
        f.write("-" * 40 + "\n")
        for nome, res in resultados_cv.items():
            f.write(f"{nome}: F1 = {res['Média']:.4f} ± {res['Desvio Padrão']:.4f}\n")
        
        f.write("\n\nTOP 10 FEATURES MAIS IMPORTANTES (Random Forest):\n")
        f.write("-" * 40 + "\n")
        for i, row in importancias_rf.head(10).iterrows():
            f.write(f"  {row['Feature']}: {row['Importância']:.4f}\n")
    
    print("Resumo salvo: resumo_resultados.txt")

def main():
    """Função principal."""
    print("\n" + "=" * 80)
    print("ANÁLISE COMPLETA DE MODELOS DE ML PARA SUPERCONDUTORES")
    print("=" * 80 + "\n")
    
    # 1. Carregar dados
    X, y_binary, y = carregar_dados()
    
    # 2. Análise exploratória
    analise_exploratoria(X, y_binary, y)
    
    # 3. Treinar modelos
    resultados, predicoes, probabilidades, y_test, modelos, X_train, X_test, scaler = treinar_modelos(X, y_binary)
    
    # 4. Gerar gráficos de resultados
    gerar_graficos_resultados(resultados, predicoes, probabilidades, y_test)
    
    # 5. Análise de importância de features
    feature_names = X.columns.tolist()
    importancias_rf, importancias_xgb, importancias_lgb = analise_importancia_features(modelos, X_train, feature_names)
    
    # 6. Validação cruzada
    resultados_cv = validacao_cruzada(X, y_binary, modelos)
    
    # 7. Salvar resultados
    salvar_resultados(resultados, resultados_cv, importancias_rf)
    
    print("\n" + "=" * 80)
    print("ANÁLISE COMPLETA FINALIZADA!")
    print("=" * 80)
    print(f"\nGráficos salvos em: {GRAFICOS_DIR}")
    
    # Retornar resultados para uso posterior
    return resultados, resultados_cv, modelos

if __name__ == "__main__":
    resultados, resultados_cv, modelos = main()
