"""
07_analise_comparativa.py
==========================
Análise comparativa estatística entre modelos:
- Validação cruzada estratificada 5-fold
- Teste de McNemar pareado
- Teste de Friedman + Nemenyi post-hoc
- Intervalos de confiança bootstrap 95%
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (f1_score, accuracy_score, roc_auc_score)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy import stats
from itertools import combinations
warnings.filterwarnings('ignore')

# --- Paths ---
DATA_PATH = '../dados_preprocessados/dados_tcc_v2.npz'
PARAMS_PATH = '../dados_preprocessados/best_params.json'
GRAFICOS_PATH = '../graficos/'
RESULTS_PATH = '../dados_preprocessados/comparacao_estatistica.json'

os.makedirs(GRAFICOS_PATH, exist_ok=True)

print("="*70)
print("ANÁLISE COMPARATIVA ESTATÍSTICA")
print("="*70)

# --- Carregar dados ---
data = np.load(DATA_PATH, allow_pickle=True)
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
feature_names = data['feature_names']

with open(PARAMS_PATH, 'r') as f:
    params_data = json.load(f)
best_params = params_data['best_params']

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

model_names = ['random_forest', 'xgboost', 'lightgbm', 'svm', 'mlp', 'lasso']
model_display = {
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost',
    'lightgbm': 'LightGBM',
    'svm': 'SVM (RBF)',
    'mlp': 'MLP',
    'lasso': 'LASSO (L1)'
}

# =============================================================================
# 1. VALIDAÇÃO CRUZADA ESTRATIFICADA 5-FOLD
# =============================================================================
print("\n[1] Validação Cruzada Estratificada (5-fold)...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = {name: [] for name in model_names}

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    for name in model_names:
        model = get_model(name, best_params[name])
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        cv_results[name].append(f1)
    
    print(f"  Fold {fold_idx+1} concluído")

# Print CV results
print(f"\n  {'Modelo':<15} {'Média F1':>10} {'Std':>8}")
print("  " + "-"*35)
for name in model_names:
    scores = cv_results[name]
    print(f"  {model_display[name]:<15} {np.mean(scores):>10.4f} {np.std(scores):>8.4f}")

# Boxplot da CV
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
cv_df = pd.DataFrame({model_display[n]: cv_results[n] for n in model_names})
cv_df.boxplot(ax=ax, grid=True)
ax.set_title('Validação Cruzada (5-fold) — F1-Score por Modelo')
ax.set_ylabel('F1-Score')
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_PATH, '07_cv_boxplot.png'), dpi=300)
plt.close()
print("  Gráfico salvo: 07_cv_boxplot.png")

# =============================================================================
# 2. TESTE DE McNEMAR PAREADO
# =============================================================================
print("\n[2] Teste de McNemar pareado...")

# Train all models on full training set and get predictions on test set
test_predictions = {}
for name in model_names:
    model = get_model(name, best_params[name])
    model.fit(X_train, y_train)
    test_predictions[name] = model.predict(X_test)

# McNemar test between all pairs
mcnemar_pvalues = np.ones((len(model_names), len(model_names)))

for i, j in combinations(range(len(model_names)), 2):
    name_i, name_j = model_names[i], model_names[j]
    pred_i = test_predictions[name_i]
    pred_j = test_predictions[name_j]
    
    # Contingency table
    # n00: both wrong, n01: i wrong j right, n10: i right j wrong, n11: both right
    correct_i = (pred_i == y_test)
    correct_j = (pred_j == y_test)
    
    n01 = np.sum(~correct_i & correct_j)  # i wrong, j right
    n10 = np.sum(correct_i & ~correct_j)  # i right, j wrong
    
    # McNemar test (with continuity correction)
    if n01 + n10 > 0:
        chi2 = (abs(n01 - n10) - 1)**2 / (n01 + n10)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
    else:
        p_value = 1.0
    
    mcnemar_pvalues[i, j] = p_value
    mcnemar_pvalues[j, i] = p_value

# Plot McNemar p-values matrix
fig, ax = plt.subplots(1, 1, figsize=(8, 7))
labels = [model_display[n] for n in model_names]
mask = np.triu(np.ones_like(mcnemar_pvalues, dtype=bool), k=1)
sns.heatmap(mcnemar_pvalues, annot=True, fmt='.3f', cmap='RdYlGn_r',
            xticklabels=labels, yticklabels=labels, ax=ax,
            mask=np.tril(np.ones_like(mcnemar_pvalues, dtype=bool), k=-1),
            vmin=0, vmax=1)
ax.set_title('Teste de McNemar — p-valores\n(p < 0.05 indica diferença significativa)')
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_PATH, '07_mcnemar_pvalues.png'), dpi=300)
plt.close()
print("  Gráfico salvo: 07_mcnemar_pvalues.png")

# =============================================================================
# 3. TESTE DE FRIEDMAN + NEMENYI POST-HOC
# =============================================================================
print("\n[3] Teste de Friedman + Nemenyi post-hoc...")

# Friedman test on CV results
cv_matrix = np.array([cv_results[name] for name in model_names]).T  # (n_folds, n_models)
friedman_stat, friedman_p = stats.friedmanchisquare(*[cv_matrix[:, i] for i in range(len(model_names))])
print(f"  Friedman chi²: {friedman_stat:.4f}, p-value: {friedman_p:.4f}")

if friedman_p < 0.05:
    print("  → Diferença significativa entre modelos (p < 0.05)")
else:
    print("  → Sem diferença significativa entre modelos (p >= 0.05)")

# Nemenyi post-hoc (using ranks)
try:
    import scikit_posthocs as sp
    nemenyi_result = sp.posthoc_nemenyi_friedman(cv_matrix)
    nemenyi_result.columns = [model_display[n] for n in model_names]
    nemenyi_result.index = [model_display[n] for n in model_names]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    sns.heatmap(nemenyi_result, annot=True, fmt='.3f', cmap='RdYlGn',
                ax=ax, vmin=0, vmax=1)
    ax.set_title('Teste de Nemenyi — p-valores\n(p < 0.05 indica diferença significativa)')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAFICOS_PATH, '23_nemenyi_diagram.png'), dpi=300)
    plt.close()
    print("  Gráfico salvo: 23_nemenyi_diagram.png")
except ImportError:
    print("  AVISO: scikit-posthocs não disponível. Nemenyi não gerado.")
    nemenyi_result = None

# Critical Difference Diagram
# Calculate average ranks
ranks = np.zeros((cv_matrix.shape[0], cv_matrix.shape[1]))
for i in range(cv_matrix.shape[0]):
    ranks[i] = stats.rankdata(-cv_matrix[i])  # Higher F1 = lower rank
avg_ranks = ranks.mean(axis=0)

fig, ax = plt.subplots(1, 1, figsize=(10, 4))
sorted_idx = np.argsort(avg_ranks)
y_positions = range(len(model_names))
ax.barh(y_positions, avg_ranks[sorted_idx], color='steelblue', edgecolor='black')
ax.set_yticks(y_positions)
ax.set_yticklabels([model_display[model_names[i]] for i in sorted_idx])
ax.set_xlabel('Rank Médio (menor = melhor)')
ax.set_title(f'Ranking dos Modelos (Friedman p={friedman_p:.4f})')
ax.grid(True, alpha=0.3, axis='x')
for i, idx in enumerate(sorted_idx):
    ax.text(avg_ranks[idx] + 0.05, i, f'{avg_ranks[idx]:.2f}', va='center')
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_PATH, '07_ranking_modelos.png'), dpi=300)
plt.close()
print("  Gráfico salvo: 07_ranking_modelos.png")

# =============================================================================
# 4. INTERVALOS DE CONFIANÇA BOOTSTRAP 95%
# =============================================================================
print("\n[4] Intervalos de confiança bootstrap 95% (1000 reamostragens)...")

N_BOOTSTRAP = 1000
bootstrap_results = {name: {'f1': [], 'roc_auc': [], 'accuracy': []} for name in model_names}

# Load probabilities from saved models
for name in model_names:
    with open(f'../modelos/{name}_results.pkl', 'rb') as f:
        model_res = pickle.load(f)
    y_pred = model_res['y_pred']
    y_proba = model_res['y_proba']
    
    rng = np.random.RandomState(RANDOM_STATE)
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(len(y_test), size=len(y_test), replace=True)
        y_t = y_test[idx]
        y_p = y_pred[idx]
        y_prob = y_proba[idx]
        
        # Avoid edge cases
        if len(np.unique(y_t)) < 2:
            continue
        
        bootstrap_results[name]['f1'].append(f1_score(y_t, y_p))
        bootstrap_results[name]['accuracy'].append(accuracy_score(y_t, y_p))
        try:
            bootstrap_results[name]['roc_auc'].append(roc_auc_score(y_t, y_prob))
        except:
            pass

# Print bootstrap CIs
print(f"\n  {'Modelo':<15} {'F1 (95% CI)':>25} {'ROC-AUC (95% CI)':>25} {'Accuracy (95% CI)':>25}")
print("  " + "-"*95)

bootstrap_ci = {}
for name in model_names:
    f1_arr = np.array(bootstrap_results[name]['f1'])
    roc_arr = np.array(bootstrap_results[name]['roc_auc'])
    acc_arr = np.array(bootstrap_results[name]['accuracy'])
    
    f1_ci = (np.percentile(f1_arr, 2.5), np.percentile(f1_arr, 97.5))
    roc_ci = (np.percentile(roc_arr, 2.5), np.percentile(roc_arr, 97.5)) if len(roc_arr) > 0 else (0, 0)
    acc_ci = (np.percentile(acc_arr, 2.5), np.percentile(acc_arr, 97.5))
    
    bootstrap_ci[name] = {
        'f1_mean': float(np.mean(f1_arr)),
        'f1_ci_low': float(f1_ci[0]),
        'f1_ci_high': float(f1_ci[1]),
        'roc_auc_mean': float(np.mean(roc_arr)) if len(roc_arr) > 0 else 0,
        'roc_auc_ci_low': float(roc_ci[0]),
        'roc_auc_ci_high': float(roc_ci[1]),
        'accuracy_mean': float(np.mean(acc_arr)),
        'accuracy_ci_low': float(acc_ci[0]),
        'accuracy_ci_high': float(acc_ci[1])
    }
    
    print(f"  {model_display[name]:<15} {np.mean(f1_arr):.4f} [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]  "
          f"{np.mean(roc_arr):.4f} [{roc_ci[0]:.4f}, {roc_ci[1]:.4f}]  "
          f"{np.mean(acc_arr):.4f} [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]")

# Plot bootstrap CIs
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics_labels = ['F1-Score', 'ROC-AUC', 'Accuracy']
metrics_keys = ['f1', 'roc_auc', 'accuracy']

for ax, metric, label in zip(axes, metrics_keys, metrics_labels):
    means = [bootstrap_ci[n][f'{metric}_mean'] for n in model_names]
    ci_low = [bootstrap_ci[n][f'{metric}_ci_low'] for n in model_names]
    ci_high = [bootstrap_ci[n][f'{metric}_ci_high'] for n in model_names]
    errors = [[m - l for m, l in zip(means, ci_low)],
              [h - m for m, h in zip(means, ci_high)]]
    
    ax.barh(range(len(model_names)), means, xerr=errors, 
            color='steelblue', edgecolor='black', alpha=0.7, capsize=3)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels([model_display[n] for n in model_names])
    ax.set_xlabel(label)
    ax.set_title(f'{label} com IC 95% (Bootstrap)')
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_PATH, '07_bootstrap_ci.png'), dpi=300)
plt.close()
print("  Gráfico salvo: 07_bootstrap_ci.png")

# =============================================================================
# 5. SALVAR RESULTADOS
# =============================================================================
print("\n[5] Salvando resultados...")

output = {
    'cv_results': {name: cv_results[name] for name in model_names},
    'cv_means': {name: float(np.mean(cv_results[name])) for name in model_names},
    'cv_stds': {name: float(np.std(cv_results[name])) for name in model_names},
    'friedman': {
        'statistic': float(friedman_stat),
        'p_value': float(friedman_p),
        'significant': bool(friedman_p < 0.05)
    },
    'mcnemar_pvalues': mcnemar_pvalues.tolist(),
    'bootstrap_ci': bootstrap_ci,
    'avg_ranks': {model_names[i]: float(avg_ranks[i]) for i in range(len(model_names))}
}

with open(RESULTS_PATH, 'w') as f:
    json.dump(output, f, indent=2)

print(f"  Resultados salvos em: {RESULTS_PATH}")
print(f"\n{'='*70}")
print("ANÁLISE COMPARATIVA CONCLUÍDA")
print(f"{'='*70}")
