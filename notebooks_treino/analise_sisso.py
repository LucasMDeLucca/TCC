"""
analise_sisso.py
=================
Análise SISSO refinada — Regressão Simbólica para Tc de Supercondutores.
Usa features fisicamente motivadas e múltiplas dimensionalidades.
"""

import numpy as np
import pandas as pd
import json
import os
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
warnings.filterwarnings('ignore')

# --- Paths ---
DATA_PATH = '../dados/data_treino.csv'
GRAFICOS_PATH = '../graficos/'
OUTPUT_PATH = '../dados_preprocessados/sisso_results.json'

os.makedirs(GRAFICOS_PATH, exist_ok=True)

print("="*70)
print("SISSO — REGRESSÃO SIMBÓLICA PARA Tc")
print("="*70)

# 1. Carregar dados
df = pd.read_csv(DATA_PATH)
df_sc = df[df['Tc'] > 0].copy()
print(f"\nSupercondutores (Tc > 0): {len(df_sc)}")

# Selecionar features numéricas
X = df_sc.select_dtypes(include=[np.number]).drop(columns=['Tc'], errors='ignore')
X = X.dropna(axis=1, thresh=int(0.5*len(X)))
X = X.fillna(X.median())
y = df_sc['Tc'].values

# Top 10 features mais correlacionadas com Tc
corr = X.corrwith(pd.Series(y, index=X.index)).abs().sort_values(ascending=False)
top_features = corr.head(10).index.tolist()
print(f"Top 10 features: {top_features}")

X_sel = X[top_features]

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.2, random_state=42)
print(f"Treino: {len(X_train)}, Teste: {len(X_test)}")

# 3. Escalar
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 4. Múltiplas dimensionalidades
cv = KFold(n_splits=5, shuffle=True, random_state=42)
results_by_dim = {}

for degree in range(1, 4):
    poly = PolynomialFeatures(degree=degree, interaction_only=(degree > 2), include_bias=False)
    Xp_train = poly.fit_transform(X_train_s)
    Xp_test = poly.transform(X_test_s)
    
    lasso_cv = LassoCV(cv=cv, max_iter=10000, random_state=42)
    lasso_cv.fit(Xp_train, y_train)
    
    lasso = Lasso(alpha=lasso_cv.alpha_, max_iter=10000, random_state=42)
    lasso.fit(Xp_train, y_train)
    
    y_pred_train = lasso.predict(Xp_train)
    y_pred_test = lasso.predict(Xp_test)
    
    cv_scores = cross_val_score(lasso, Xp_train, y_train, cv=cv, scoring='r2')
    
    r2_tr = r2_score(y_train, y_pred_train)
    r2_te = r2_score(y_test, y_pred_test)
    rmse_te = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_te = mean_absolute_error(y_test, y_pred_test)
    n_nz = np.sum(np.abs(lasso.coef_) > 1e-10)
    
    results_by_dim[degree] = {
        'r2_cv': float(np.mean(cv_scores)),
        'r2_cv_std': float(np.std(cv_scores)),
        'r2_train': float(r2_tr),
        'r2_test': float(r2_te),
        'rmse_test': float(rmse_te),
        'mae_test': float(mae_te),
        'n_features_selected': int(n_nz),
        'n_features_total': int(Xp_train.shape[1]),
        'alpha': float(lasso_cv.alpha_)
    }
    
    print(f"\n  {degree}D: R²(CV)={np.mean(cv_scores):.4f}±{np.std(cv_scores):.4f}, "
          f"R²(test)={r2_te:.4f}, RMSE={rmse_te:.2f}K, Features={n_nz}/{Xp_train.shape[1]}")

# 5. Melhor modelo (2D) - extrair equação
best_degree = 2
poly = PolynomialFeatures(degree=best_degree, include_bias=False)
Xp_train = poly.fit_transform(X_train_s)
Xp_test = poly.transform(X_test_s)
feat_names = poly.get_feature_names_out(top_features)

lasso_cv = LassoCV(cv=cv, max_iter=10000, random_state=42)
lasso_cv.fit(Xp_train, y_train)
lasso = Lasso(alpha=lasso_cv.alpha_, max_iter=10000, random_state=42)
lasso.fit(Xp_train, y_train)

y_pred_train = lasso.predict(Xp_train)
y_pred_test = lasso.predict(Xp_test)

# Equação
nz_mask = np.abs(lasso.coef_) > 1e-10
nz_coefs = lasso.coef_[nz_mask]
nz_names = feat_names[nz_mask]
sort_idx = np.argsort(np.abs(nz_coefs))[::-1]
nz_coefs = nz_coefs[sort_idx]
nz_names = nz_names[sort_idx]

eq_parts = [f"{lasso.intercept_:.3f}"]
for c, n in zip(nz_coefs[:8], nz_names[:8]):
    sign = "+" if c > 0 else "-"
    eq_parts.append(f"{sign} {abs(c):.3f}*{n}")
equation = "Tc = " + " ".join(eq_parts)
print(f"\nEquação (2D, top 8 termos): {equation}")

# 6. Gráficos
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# R² vs dimensionalidade
dims = list(results_by_dim.keys())
r2_cv = [results_by_dim[d]['r2_cv'] for d in dims]
r2_cv_std = [results_by_dim[d]['r2_cv_std'] for d in dims]
r2_test = [results_by_dim[d]['r2_test'] for d in dims]

axes[0].errorbar(dims, r2_cv, yerr=r2_cv_std, fmt='o-', color='steelblue', lw=2, capsize=5, label='R² (CV)')
axes[0].plot(dims, r2_test, 's--', color='darkorange', lw=2, label='R² (Teste)')
axes[0].set_xlabel('Dimensionalidade')
axes[0].set_ylabel('R²')
axes[0].set_title('SISSO: R² vs Dimensionalidade')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(dims)

# Predição vs Real
axes[1].scatter(y_test, y_pred_test, alpha=0.4, s=20, color='steelblue')
axes[1].plot([0, y_test.max()], [0, y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('$T_c$ Real (K)')
axes[1].set_ylabel('$T_c$ Predito (K)')
r2_te = r2_score(y_test, y_pred_test)
axes[1].set_title(f'SISSO 2D: Predição vs Real (R²={r2_te:.3f})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_PATH, '20_sisso_predicao_tc.png'), dpi=300)
plt.close()
print("Gráfico salvo: 20_sisso_predicao_tc.png")

# Coeficientes
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
n_show = min(15, len(nz_coefs))
colors = ['green' if c > 0 else 'red' for c in nz_coefs[:n_show]]
ax.barh(range(n_show), nz_coefs[:n_show], color=colors, edgecolor='black')
ax.set_yticks(range(n_show))
ax.set_yticklabels([str(n)[:35] for n in nz_names[:n_show]])
ax.set_xlabel('Coeficiente')
ax.set_title('SISSO 2D: Termos da Equação')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_PATH, '22_sisso_coeficientes.png'), dpi=300)
plt.close()
print("Gráfico salvo: 22_sisso_coeficientes.png")

# 7. Salvar resultados
output = {
    'results_by_dimensionality': results_by_dim,
    'best_degree': best_degree,
    'equation': equation,
    'top_features': top_features,
    'intercept': float(lasso.intercept_),
    'n_terms': int(len(nz_coefs))
}
with open(OUTPUT_PATH, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResultados salvos em: {OUTPUT_PATH}")
print("\n" + "="*70)
print("ANÁLISE SISSO CONCLUÍDA")
print("="*70)
