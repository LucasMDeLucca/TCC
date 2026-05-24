"""
08_interpretabilidade.py
=========================
Interpretabilidade expandida:
- SHAP (summary, waterfall, dependence)
- Calibração de probabilidades (reliability diagram, Brier score)
- Curva de aprendizado (F1 vs tamanho do treino)
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
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, f1_score
from sklearn.model_selection import learning_curve, StratifiedKFold
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

# --- Paths ---
DATA_PATH = '../dados_preprocessados/dados_tcc_v2.npz'
PARAMS_PATH = '../dados_preprocessados/best_params.json'
GRAFICOS_PATH = '../graficos/'

os.makedirs(GRAFICOS_PATH, exist_ok=True)

print("="*70)
print("ANÁLISE DE INTERPRETABILIDADE EXPANDIDA")
print("="*70)

# --- Carregar dados ---
data = np.load(DATA_PATH, allow_pickle=True)
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
feature_names = list(data['feature_names'])

with open(PARAMS_PATH, 'r') as f:
    params_data = json.load(f)
best_params = params_data['best_params']

RANDOM_STATE = 42

# =============================================================================
# 1. TREINAR MODELOS PRINCIPAIS (RF e XGBoost)
# =============================================================================
print("\n[1] Treinando modelos principais...")

rf_params = best_params['random_forest']
rf = RandomForestClassifier(random_state=RANDOM_STATE, **rf_params)
rf.fit(X_train, y_train)

xgb_params = best_params['xgboost']
xgb = XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0, **xgb_params)
xgb.fit(X_train, y_train)

rf_proba = rf.predict_proba(X_test)[:, 1]
xgb_proba = xgb.predict_proba(X_test)[:, 1]

print(f"  RF treinado. Test F1: {f1_score(y_test, rf.predict(X_test)):.4f}")
print(f"  XGB treinado. Test F1: {f1_score(y_test, xgb.predict(X_test)):.4f}")

# =============================================================================
# 2. ANÁLISE SHAP (Random Forest)
# =============================================================================
print("\n[2] Análise SHAP...")

try:
    import shap
    
    # Use TreeExplainer for RF
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    
    # For binary classification, shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]  # Class 1 (supercondutor)
    else:
        shap_vals = shap_values
    
    # 2a. Summary plot (beeswarm) - top 20 features
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    shap.summary_plot(shap_vals, X_test, feature_names=feature_names,
                      max_display=20, show=False)
    plt.title('SHAP Summary Plot — Random Forest (Top 20 Features)')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAFICOS_PATH, '24_shap_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Gráfico salvo: 24_shap_summary.png")
    
    # 2b. Waterfall plots for typical cases
    # Find a true positive with high confidence
    rf_pred = rf.predict(X_test)
    tp_mask = (rf_pred == 1) & (y_test == 1)
    fn_mask = (rf_pred == 0) & (y_test == 1)
    
    if tp_mask.sum() > 0:
        tp_idx = np.where(tp_mask)[0]
        # Pick the one with highest probability
        tp_probs = rf_proba[tp_idx]
        best_tp = tp_idx[np.argmax(tp_probs)]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        shap.waterfall_plot(shap.Explanation(
            values=shap_vals[best_tp],
            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            data=X_test[best_tp],
            feature_names=feature_names
        ), max_display=15, show=False)
        plt.title('SHAP Waterfall — Verdadeiro Positivo (alta confiança)')
        plt.tight_layout()
        plt.savefig(os.path.join(GRAFICOS_PATH, '25_shap_waterfall_tp.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("  Gráfico salvo: 25_shap_waterfall_tp.png")
    
    if fn_mask.sum() > 0:
        fn_idx = np.where(fn_mask)[0][0]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        shap.waterfall_plot(shap.Explanation(
            values=shap_vals[fn_idx],
            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            data=X_test[fn_idx],
            feature_names=feature_names
        ), max_display=15, show=False)
        plt.title('SHAP Waterfall — Falso Negativo')
        plt.tight_layout()
        plt.savefig(os.path.join(GRAFICOS_PATH, '25_shap_waterfall_fn.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("  Gráfico salvo: 25_shap_waterfall_fn.png")
    
    # 2c. Dependence plots for top 3 features
    # Find top 3 by mean absolute SHAP value
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    top3_idx = np.argsort(mean_abs_shap)[-3:][::-1]
    
    for rank, feat_idx in enumerate(top3_idx):
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        shap.dependence_plot(feat_idx, shap_vals, X_test,
                            feature_names=feature_names, show=False, ax=ax)
        plt.title(f'SHAP Dependence — {feature_names[feat_idx]}')
        plt.tight_layout()
        plt.savefig(os.path.join(GRAFICOS_PATH, f'26_shap_dependence_{rank}_{feature_names[feat_idx][:20]}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
    print("  Gráficos de dependência salvos: 26_shap_dependence_*.png")

except Exception as e:
    print(f"  AVISO: Erro no SHAP: {e}")
    print("  Gerando análise de importância alternativa...")
    
    # Fallback: Feature importance comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # RF importance
    rf_imp = rf.feature_importances_
    idx_sorted = np.argsort(rf_imp)[-20:]
    axes[0].barh(range(len(idx_sorted)), rf_imp[idx_sorted], color='steelblue')
    axes[0].set_yticks(range(len(idx_sorted)))
    axes[0].set_yticklabels([feature_names[i] for i in idx_sorted])
    axes[0].set_title('Random Forest — Feature Importance')
    
    # XGB importance
    xgb_imp = xgb.feature_importances_
    idx_sorted = np.argsort(xgb_imp)[-20:]
    axes[1].barh(range(len(idx_sorted)), xgb_imp[idx_sorted], color='darkorange')
    axes[1].set_yticks(range(len(idx_sorted)))
    axes[1].set_yticklabels([feature_names[i] for i in idx_sorted])
    axes[1].set_title('XGBoost — Feature Importance')
    
    plt.suptitle('Comparação de Importância de Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAFICOS_PATH, '24_shap_summary.png'), dpi=300)
    plt.close()
    print("  Gráfico alternativo salvo: 24_shap_summary.png")

# =============================================================================
# 3. CALIBRAÇÃO DE PROBABILIDADES
# =============================================================================
print("\n[3] Calibração de probabilidades...")

fig, ax = plt.subplots(1, 1, figsize=(8, 7))

# Perfect calibration line
ax.plot([0, 1], [0, 1], 'k--', label='Calibração perfeita')

# RF calibration
prob_true_rf, prob_pred_rf = calibration_curve(y_test, rf_proba, n_bins=10, strategy='uniform')
brier_rf = brier_score_loss(y_test, rf_proba)
ax.plot(prob_pred_rf, prob_true_rf, 'o-', color='steelblue', lw=2,
        label=f'Random Forest (Brier={brier_rf:.4f})')

# XGBoost calibration
prob_true_xgb, prob_pred_xgb = calibration_curve(y_test, xgb_proba, n_bins=10, strategy='uniform')
brier_xgb = brier_score_loss(y_test, xgb_proba)
ax.plot(prob_pred_xgb, prob_true_xgb, 's-', color='darkorange', lw=2,
        label=f'XGBoost (Brier={brier_xgb:.4f})')

ax.set_xlabel('Probabilidade Predita')
ax.set_ylabel('Fração de Positivos')
ax.set_title('Diagrama de Confiabilidade (Calibração)')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_PATH, '27_calibration.png'), dpi=300)
plt.close()
print(f"  Brier Score RF: {brier_rf:.4f}")
print(f"  Brier Score XGB: {brier_xgb:.4f}")
print("  Gráfico salvo: 27_calibration.png")

# =============================================================================
# 4. CURVA DE APRENDIZADO
# =============================================================================
print("\n[4] Curva de aprendizado...")

train_sizes_frac = [0.1, 0.25, 0.5, 0.75, 1.0]
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# RF learning curve
train_sizes_rf, train_scores_rf, val_scores_rf = learning_curve(
    RandomForestClassifier(random_state=RANDOM_STATE, **rf_params),
    X_train, y_train, train_sizes=train_sizes_frac,
    cv=cv, scoring='f1', n_jobs=2
)

ax.plot(train_sizes_rf, train_scores_rf.mean(axis=1), 'o-', color='steelblue',
        label='RF — Treino')
ax.plot(train_sizes_rf, val_scores_rf.mean(axis=1), 's--', color='steelblue',
        label='RF — Validação')
ax.fill_between(train_sizes_rf,
                val_scores_rf.mean(axis=1) - val_scores_rf.std(axis=1),
                val_scores_rf.mean(axis=1) + val_scores_rf.std(axis=1),
                alpha=0.1, color='steelblue')

# XGBoost learning curve
train_sizes_xgb, train_scores_xgb, val_scores_xgb = learning_curve(
    XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0, **xgb_params),
    X_train, y_train, train_sizes=train_sizes_frac,
    cv=cv, scoring='f1', n_jobs=2
)

ax.plot(train_sizes_xgb, train_scores_xgb.mean(axis=1), 'o-', color='darkorange',
        label='XGBoost — Treino')
ax.plot(train_sizes_xgb, val_scores_xgb.mean(axis=1), 's--', color='darkorange',
        label='XGBoost — Validação')
ax.fill_between(train_sizes_xgb,
                val_scores_xgb.mean(axis=1) - val_scores_xgb.std(axis=1),
                val_scores_xgb.mean(axis=1) + val_scores_xgb.std(axis=1),
                alpha=0.1, color='darkorange')

ax.set_xlabel('Tamanho do Conjunto de Treino')
ax.set_ylabel('F1-Score')
ax.set_title('Curva de Aprendizado — Random Forest vs XGBoost')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_PATH, '28_learning_curve.png'), dpi=300)
plt.close()
print("  Gráfico salvo: 28_learning_curve.png")

# Analyze if more data would help
val_last_rf = val_scores_rf.mean(axis=1)[-1]
val_prev_rf = val_scores_rf.mean(axis=1)[-2]
improvement_rf = val_last_rf - val_prev_rf
print(f"  RF: Melhoria de 75% → 100% dos dados: {improvement_rf:+.4f}")
print(f"  → {'Mais dados provavelmente ajudariam' if improvement_rf > 0.005 else 'Saturação atingida'}")

# =============================================================================
# 5. SALVAR RESULTADOS DE INTERPRETABILIDADE
# =============================================================================
print("\n[5] Salvando resultados...")

interp_results = {
    'brier_score_rf': float(brier_rf),
    'brier_score_xgb': float(brier_xgb),
    'learning_curve_improvement_rf': float(improvement_rf),
    'top_features_rf': [feature_names[i] for i in np.argsort(rf.feature_importances_)[-10:][::-1]],
    'top_features_xgb': [feature_names[i] for i in np.argsort(xgb.feature_importances_)[-10:][::-1]]
}

with open('../dados_preprocessados/interpretabilidade_results.json', 'w') as f:
    json.dump(interp_results, f, indent=2)

print(f"\n{'='*70}")
print("ANÁLISE DE INTERPRETABILIDADE CONCLUÍDA")
print(f"{'='*70}")
print(f"\nTop 5 features (RF): {interp_results['top_features_rf'][:5]}")
print(f"Top 5 features (XGB): {interp_results['top_features_xgb'][:5]}")
