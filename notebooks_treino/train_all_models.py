"""
train_all_models.py
====================
Re-treinamento de todos os 6 modelos com hiperparâmetros otimizados.
Gera métricas, gráficos (confusão, ROC, PR, importância) e salva em resultados_v2.json.
"""

import numpy as np
import pandas as pd
import json
import os
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             confusion_matrix, classification_report, average_precision_score)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
warnings.filterwarnings('ignore')

# --- Paths ---
DATA_PATH = '../dados_preprocessados/dados_tcc_v2.npz'
PARAMS_PATH = '../dados_preprocessados/best_params.json'
GRAFICOS_PATH = '../graficos/'
MODELOS_PATH = '../modelos/'
RESULTS_PATH = '../dados_preprocessados/resultados_v2.json'

# Por padrão, NÃO salva figuras individuais por modelo (matriz de confusão, ROC, PR,
# feature importance) porque elas não são referenciadas no TCC e poluíam o repositório
# (ver revisão crítica P4). As figuras consolidadas (07_*) sempre são salvas.
# Para gerar as individuais para inspeção (em graficos/individuais/), defina como True.
SAVE_INDIVIDUAL_FIGS = False
INDIVIDUAL_FIGS_PATH = os.path.join(GRAFICOS_PATH, 'individuais')

os.makedirs(GRAFICOS_PATH, exist_ok=True)
os.makedirs(MODELOS_PATH, exist_ok=True)
if SAVE_INDIVIDUAL_FIGS:
    os.makedirs(INDIVIDUAL_FIGS_PATH, exist_ok=True)

print("="*70)
print("RE-TREINAMENTO DOS MODELOS COM HIPERPARÂMETROS OTIMIZADOS")
print("="*70)

# --- Carregar dados ---
data = np.load(DATA_PATH, allow_pickle=True)
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
feature_names = data['feature_names']

print(f"\nDados: X_train={X_train.shape}, X_test={X_test.shape}")

# --- Carregar melhores parâmetros ---
with open(PARAMS_PATH, 'r') as f:
    params_data = json.load(f)
best_params = params_data['best_params']

# --- Definir modelos ---
RANDOM_STATE = 42

def get_model(name, params):
    if name == 'random_forest':
        return RandomForestClassifier(random_state=RANDOM_STATE, **params)
    elif name == 'xgboost':
        return XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0, **params)
    elif name == 'lightgbm':
        return LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1, n_jobs=1, **params)
    elif name == 'svm':
        return SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE, **params)
    elif name == 'mlp':
        p = params.copy()
        if 'hidden_layer_sizes' in p and isinstance(p['hidden_layer_sizes'], list):
            p['hidden_layer_sizes'] = tuple(p['hidden_layer_sizes'])
        return MLPClassifier(random_state=RANDOM_STATE, max_iter=1000, early_stopping=True, **p)
    elif name == 'lasso':
        return LogisticRegression(penalty='l1', solver='saga', random_state=RANDOM_STATE, max_iter=5000, **params)

# =============================================================================
# TREINAMENTO E AVALIAÇÃO
# =============================================================================

results = {}
models = {}
predictions = {}

model_names = ['random_forest', 'xgboost', 'lightgbm', 'svm', 'mlp', 'lasso']
model_display = {
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost',
    'lightgbm': 'LightGBM',
    'svm': 'SVM (RBF)',
    'mlp': 'MLP',
    'lasso': 'LASSO (L1)'
}

for name in model_names:
    print(f"\n{'='*50}")
    print(f"Treinando: {model_display[name]}")
    print(f"{'='*50}")
    
    params = best_params[name]
    print(f"  Parâmetros: {params}")
    
    model = get_model(name, params)
    model.fit(X_train, y_train)
    
    # Predições
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Métricas
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    
    results[name] = {
        'accuracy': float(acc),
        'f1_score': float(f1),
        'precision': float(prec),
        'recall': float(rec),
        'roc_auc': float(roc_auc),
        'average_precision': float(ap),
        'params': params
    }
    
    models[name] = model
    predictions[name] = {'y_pred': y_pred, 'y_proba': y_proba}
    
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  AP:        {ap:.4f}")
    
    # --- Gráficos individuais (opcionais, controlados por SAVE_INDIVIDUAL_FIGS) ---
    prefix = f"{model_names.index(name)+1:02d}"

    if SAVE_INDIVIDUAL_FIGS:
        # Matriz de Confusão
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Não-SC', 'SC'], yticklabels=['Não-SC', 'SC'])
        ax.set_title(f'Matriz de Confusão — {model_display[name]}')
        ax.set_xlabel('Predito')
        ax.set_ylabel('Real')
        plt.tight_layout()
        plt.savefig(os.path.join(INDIVIDUAL_FIGS_PATH, f'{prefix}_{name}_confusion_matrix.png'), dpi=300)
        plt.close()

        # Curva ROC
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax.plot(fpr, tpr, 'b-', lw=2, label=f'ROC-AUC = {roc_auc:.4f}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('Taxa de Falso Positivo')
        ax.set_ylabel('Taxa de Verdadeiro Positivo')
        ax.set_title(f'Curva ROC — {model_display[name]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(INDIVIDUAL_FIGS_PATH, f'{prefix}_{name}_roc_curve.png'), dpi=300)
        plt.close()

        # Curva Precision-Recall
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
        ax.plot(rec_curve, prec_curve, 'r-', lw=2, label=f'AP = {ap:.4f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Curva Precision-Recall — {model_display[name]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(INDIVIDUAL_FIGS_PATH, f'{prefix}_{name}_pr_curve.png'), dpi=300)
        plt.close()

        # Feature Importance (para modelos que suportam)
        if hasattr(model, 'feature_importances_'):
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            importances = model.feature_importances_
            idx_sorted = np.argsort(importances)[-20:]
            ax.barh(range(len(idx_sorted)), importances[idx_sorted], color='steelblue')
            ax.set_yticks(range(len(idx_sorted)))
            ax.set_yticklabels([feature_names[i] for i in idx_sorted])
            ax.set_xlabel('Importância')
            ax.set_title(f'Top 20 Features — {model_display[name]}')
            plt.tight_layout()
            plt.savefig(os.path.join(INDIVIDUAL_FIGS_PATH, f'{prefix}_{name}_feature_importance.png'), dpi=300)
            plt.close()
        elif name == 'lasso':
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            coefs = model.coef_[0]
            idx_sorted = np.argsort(np.abs(coefs))[-20:]
            colors = ['green' if c > 0 else 'red' for c in coefs[idx_sorted]]
            ax.barh(range(len(idx_sorted)), coefs[idx_sorted], color=colors)
            ax.set_yticks(range(len(idx_sorted)))
            ax.set_yticklabels([feature_names[i] for i in idx_sorted])
            ax.set_xlabel('Coeficiente')
            ax.set_title(f'Top 20 Coeficientes — {model_display[name]}')
            plt.tight_layout()
            plt.savefig(os.path.join(INDIVIDUAL_FIGS_PATH, f'{prefix}_{name}_feature_importance.png'), dpi=300)
            plt.close()
    
    # Salvar modelo
    with open(os.path.join(MODELOS_PATH, f'{name}_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    # Salvar resultados individuais
    with open(os.path.join(MODELOS_PATH, f'{name}_results.pkl'), 'wb') as f:
        pickle.dump({
            'y_pred': y_pred,
            'y_proba': y_proba,
            'metrics': results[name]
        }, f)

# =============================================================================
# GRÁFICOS COMPARATIVOS
# =============================================================================
print(f"\n{'='*70}")
print("GERANDO GRÁFICOS COMPARATIVOS")
print(f"{'='*70}")

# Comparação de métricas
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
metrics_df = pd.DataFrame(results).T
metrics_plot = metrics_df[['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc']]
metrics_plot.index = [model_display[n] for n in metrics_plot.index]
metrics_plot.plot(kind='bar', ax=ax, width=0.8)
ax.set_title('Comparação de Métricas entre Modelos')
ax.set_ylabel('Score')
ax.set_xlabel('')
ax.set_ylim(0, 1)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_PATH, '07_comparacao_metricas.png'), dpi=300)
plt.close()

# Curvas ROC sobrepostas
fig, ax = plt.subplots(1, 1, figsize=(8, 7))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
for i, name in enumerate(model_names):
    fpr, tpr, _ = roc_curve(y_test, predictions[name]['y_proba'])
    ax.plot(fpr, tpr, color=colors[i], lw=2,
            label=f"{model_display[name]} (AUC={results[name]['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_xlabel('Taxa de Falso Positivo')
ax.set_ylabel('Taxa de Verdadeiro Positivo')
ax.set_title('Curvas ROC — Comparação entre Modelos')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_PATH, '07_curvas_roc_comparacao.png'), dpi=300)
plt.close()

# Curvas PR sobrepostas
fig, ax = plt.subplots(1, 1, figsize=(8, 7))
for i, name in enumerate(model_names):
    prec_c, rec_c, _ = precision_recall_curve(y_test, predictions[name]['y_proba'])
    ax.plot(rec_c, prec_c, color=colors[i], lw=2,
            label=f"{model_display[name]} (AP={results[name]['average_precision']:.3f})")
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Curvas Precision-Recall — Comparação entre Modelos')
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_PATH, '07_curvas_pr_comparacao.png'), dpi=300)
plt.close()

# Matrizes de confusão lado a lado
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, name in enumerate(model_names):
    ax = axes[i // 3, i % 3]
    cm = confusion_matrix(y_test, predictions[name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Não-SC', 'SC'], yticklabels=['Não-SC', 'SC'])
    ax.set_title(f'{model_display[name]}')
    ax.set_xlabel('Predito')
    ax.set_ylabel('Real')
plt.suptitle('Matrizes de Confusão — Todos os Modelos', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_PATH, '07_matrizes_confusao.png'), dpi=300)
plt.close()

# =============================================================================
# SALVAR RESULTADOS FINAIS
# =============================================================================

with open(RESULTS_PATH, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResultados salvos em: {RESULTS_PATH}")
print(f"\n{'='*70}")
print("RESUMO FINAL")
print(f"{'='*70}")
print(f"\n{'Modelo':<15} {'Accuracy':>10} {'F1':>8} {'Precision':>10} {'Recall':>8} {'ROC-AUC':>9}")
print("-"*60)
for name in model_names:
    r = results[name]
    print(f"{model_display[name]:<15} {r['accuracy']:>10.4f} {r['f1_score']:>8.4f} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['roc_auc']:>9.4f}")

# Best model
best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
print(f"\n*** Melhor modelo (F1): {model_display[best_model[0]]} com F1 = {best_model[1]['f1_score']:.4f} ***")
