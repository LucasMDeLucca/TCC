"""
00_pipeline_preprocessamento.py
================================
Pipeline de pré-processamento corrigido para o TCC.
Correções implementadas:
  - Separação explícita de Tc>0, Tc=0 e Tc=NaN com documentação
  - Abordagem padrão: Tc>0 = supercondutor, Tc=0 ou NaN = não-supercondutor
  - Alternativa PU Learning documentada (comentada)
  - Split treino/teste com controle de data leakage via GroupShuffleSplit
  - Análise descritiva completa do dataset
  - Salva data_treino_v2.csv e dados processados
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
import json
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# --- Configurações ---
DATA_PATH = '../dados/data_treino.csv'
OUTPUT_DATA_PATH = '../dados/data_treino_v2.csv'
OUTPUT_NPZ_PATH = '../dados_preprocessados/dados_tcc_v2.npz'
OUTPUT_SCALER_PATH = '../dados_preprocessados/scaler_v2.pkl'
OUTPUT_FEATURES_PATH = '../dados_preprocessados/selected_features_v2.pkl'
GRAFICOS_PATH = '../graficos/'
RANDOM_STATE = 42

os.makedirs(GRAFICOS_PATH, exist_ok=True)
os.makedirs('../dados_preprocessados/', exist_ok=True)

print("="*70)
print("PIPELINE DE PRÉ-PROCESSAMENTO — VERSÃO 2 (CORRIGIDA)")
print("="*70)

# =============================================================================
# 1. CARREGAMENTO E ANÁLISE INICIAL
# =============================================================================
print("\n[1] Carregando dados...")
df = pd.read_csv(DATA_PATH)
print(f"   Dataset original: {df.shape[0]} materiais, {df.shape[1]} colunas")

# =============================================================================
# 2. SEPARAÇÃO EXPLÍCITA DAS CATEGORIAS DE Tc
# =============================================================================
print("\n[2] Separação explícita de categorias de Tc...")

tc_positive = (df['Tc'] > 0).sum()       # SC confirmado
tc_zero = (df['Tc'] == 0).sum()           # Não-SC confirmado (Tc medido = 0)
tc_nan = df['Tc'].isna().sum()            # Não testado / Tc não reportado

print(f"   Tc > 0  (supercondutor confirmado):       {tc_positive}")
print(f"   Tc = 0  (não-SC confirmado, Tc medido):   {tc_zero}")
print(f"   Tc NaN  (não testado / não reportado):    {tc_nan}")
print(f"   Total:                                    {len(df)}")
print(f"\n   NOTA: Materiais com Tc=NaN são tratados como 'não-supercondutores'")
print(f"   neste trabalho. Esta é uma aproximação comum na literatura (Stanev")
print(f"   et al., 2018), mas introduz ruído: alguns desses materiais podem ser")
print(f"   supercondutores não testados. Uma abordagem alternativa seria PU Learning.")

# =============================================================================
# 3. CRIAÇÃO DO LABEL BINÁRIO
# =============================================================================
print("\n[3] Criando label binário...")

# ABORDAGEM PADRÃO: Tc > 0 → supercondutor; Tc = 0 ou NaN → não-supercondutor
df['is_superconductor'] = ((df['Tc'] > 0) & (df['Tc'].notna())).astype(int)

n_sc = df['is_superconductor'].sum()
n_nsc = len(df) - n_sc
print(f"   Supercondutores (Tc > 0):     {n_sc} ({100*n_sc/len(df):.1f}%)")
print(f"   Não-supercondutores:          {n_nsc} ({100*n_nsc/len(df):.1f}%)")
print(f"   Razão SC/não-SC:              {n_sc/n_nsc:.2f}")

# --- ALTERNATIVA (comentada): PU Learning ---
# """
# Positive-Unlabeled (PU) Learning:
# - Positivos: Tc > 0 (supercondutores confirmados)
# - Unlabeled: Tc = NaN (podem ser SC ou não-SC)
# - Negativos confiáveis: Tc = 0 (não-SC confirmado, apenas 1 material)
#
# from pulearn import ElkanotoPuClassifier
# from sklearn.ensemble import RandomForestClassifier
#
# # Marcar: 1 = positivo confirmado, 0 = unlabeled
# y_pu = df['is_superconductor'].copy()  # 1 para Tc>0
# # Treinar PU classifier
# base_estimator = RandomForestClassifier(n_estimators=100, random_state=42)
# pu_clf = ElkanotoPuClassifier(estimator=base_estimator, hold_out_ratio=0.1)
# pu_clf.fit(X_scaled, y_pu)
# # Usar predições do PU como labels refinados
# """

# =============================================================================
# 4. SELEÇÃO DE FEATURES NUMÉRICAS
# =============================================================================
print("\n[4] Selecionando features numéricas...")

# Remover colunas não-numéricas e identificadores
cols_to_drop = ['Tc', 'is_superconductor', 'id']
# Manter group_id separado para GroupShuffleSplit
text_cols = [c for c in df.columns if df[c].dtype == 'object']
cols_to_drop.extend(text_cols)

feature_cols = [c for c in df.columns if c not in cols_to_drop and c != 'group_id']

# Remover features com mais de 50% de valores ausentes
missing_pct = df[feature_cols].isna().mean()
high_missing = missing_pct[missing_pct > 0.5].index.tolist()
print(f"   Features com >50% ausentes (removidas): {len(high_missing)}")
feature_cols = [c for c in feature_cols if c not in high_missing]

# Converter para numérico (forçar erros para NaN)
X_all = df[feature_cols].apply(pd.to_numeric, errors='coerce')

# Remover features com variância zero
X_temp = X_all.fillna(X_all.median(numeric_only=True))
vt = VarianceThreshold(threshold=0.0)
vt.fit(X_temp)
zero_var = [feature_cols[i] for i in range(len(feature_cols)) if not vt.get_support()[i]]
print(f"   Features com variância zero (removidas): {len(zero_var)}")
feature_cols = [c for c in feature_cols if c not in zero_var]

print(f"   Features selecionadas: {len(feature_cols)}")

# =============================================================================
# 5. IMPUTAÇÃO DE VALORES AUSENTES
# =============================================================================
print("\n[5] Imputação de valores ausentes...")

X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
y = df['is_superconductor'].values

imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols, index=X.index)

missing_before = X.isna().sum().sum()
missing_after = X_imputed.isna().sum().sum()
print(f"   Valores ausentes antes: {missing_before}")
print(f"   Valores ausentes depois: {missing_after}")

# =============================================================================
# 6. SPLIT TREINO/TESTE COM CONTROLE DE DATA LEAKAGE
# =============================================================================
print("\n[6] Split treino/teste com controle de data leakage...")

# Usar group_id para agrupar materiais da mesma família estrutural
groups = df['group_id'].values

# GroupShuffleSplit: materiais do mesmo group_id ficam no mesmo split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X_imputed, y, groups))

X_train = X_imputed.iloc[train_idx]
X_test = X_imputed.iloc[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

print(f"   Treino: {len(X_train)} amostras ({100*y_train.mean():.1f}% SC)")
print(f"   Teste:  {len(X_test)} amostras ({100*y_test.mean():.1f}% SC)")
print(f"   Grupos únicos no treino: {len(np.unique(groups[train_idx]))}")
print(f"   Grupos únicos no teste:  {len(np.unique(groups[test_idx]))}")

# Verificar que não há leakage de grupos
train_groups = set(groups[train_idx])
test_groups = set(groups[test_idx])
overlap = train_groups.intersection(test_groups)
print(f"   Overlap de grupos (deve ser 0): {len(overlap)}")
assert len(overlap) == 0, "DATA LEAKAGE DETECTADO!"

# =============================================================================
# 7. ESCALONAMENTO
# =============================================================================
print("\n[7] Escalonamento (StandardScaler)...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   Média do treino (deve ser ~0): {X_train_scaled.mean():.6f}")
print(f"   Std do treino (deve ser ~1): {X_train_scaled.std():.6f}")

# =============================================================================
# 8. SALVAR DADOS PROCESSADOS
# =============================================================================
print("\n[8] Salvando dados processados...")

# Salvar dataset com labels
df_save = df.copy()
df_save.to_csv(OUTPUT_DATA_PATH, index=False)
print(f"   Salvo: {OUTPUT_DATA_PATH}")

# Salvar dados processados em NPZ
np.savez(OUTPUT_NPZ_PATH,
         X_train=X_train_scaled,
         X_test=X_test_scaled,
         y_train=y_train,
         y_test=y_test,
         feature_names=np.array(feature_cols),
         train_idx=train_idx,
         test_idx=test_idx,
         groups_train=groups[train_idx],
         groups_test=groups[test_idx])
print(f"   Salvo: {OUTPUT_NPZ_PATH}")

# Salvar scaler
with open(OUTPUT_SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)
print(f"   Salvo: {OUTPUT_SCALER_PATH}")

# Salvar lista de features
with open(OUTPUT_FEATURES_PATH, 'wb') as f:
    pickle.dump(feature_cols, f)
print(f"   Salvo: {OUTPUT_FEATURES_PATH}")

# =============================================================================
# 9. ANÁLISE DESCRITIVA E GRÁFICO RESUMO
# =============================================================================
print("\n[9] Gerando análise descritiva e gráfico resumo...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Resumo do Dataset (data_treino_v2.csv)', fontsize=14, fontweight='bold')

# 9a. Distribuição de classes (3 categorias)
ax = axes[0, 0]
categories = ['SC\n(Tc>0)', 'Não-SC\n(Tc=0)', 'Não testado\n(Tc=NaN)']
cat_counts = [tc_positive, tc_zero, tc_nan]
colors = ['#2ecc71', '#e74c3c', '#95a5a6']
bars = ax.bar(categories, cat_counts, color=colors, edgecolor='black')
ax.set_title('Categorias de Tc no Dataset')
ax.set_ylabel('Número de Materiais')
for bar, v in zip(bars, cat_counts):
    ax.text(bar.get_x() + bar.get_width()/2, v + 10, str(v), ha='center', fontweight='bold')

# 9b. Distribuição de Tc (apenas SC)
ax = axes[0, 1]
tc_sc = df[df['Tc'] > 0]['Tc']
ax.hist(tc_sc, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax.set_title('Distribuição de $T_c$ (Supercondutores)')
ax.set_xlabel('$T_c$ (K)')
ax.set_ylabel('Frequência')
ax.axvline(tc_sc.median(), color='red', linestyle='--', label=f'Mediana: {tc_sc.median():.1f} K')
ax.axvline(tc_sc.mean(), color='orange', linestyle=':', label=f'Média: {tc_sc.mean():.1f} K')
ax.legend()

# 9c. Completude por feature (top 20 com mais missing)
ax = axes[0, 2]
completeness = (1 - X[feature_cols].isna().mean()).sort_values()[:20]
completeness.plot(kind='barh', ax=ax, color='orange', edgecolor='black')
ax.set_title('Completude (20 piores features)')
ax.set_xlabel('Fração de dados presentes')
ax.axvline(0.5, color='red', linestyle='--', alpha=0.5)

# 9d. Distribuição por classe das top features
ax = axes[1, 0]
# Find top features by variance
feature_var = X_imputed.var().sort_values(ascending=False)
top5 = feature_var.index[:5].tolist()
data_plot = X_imputed[top5].copy()
data_plot['Classe'] = ['SC' if yi == 1 else 'Não-SC' for yi in y]
data_melted = data_plot.melt(id_vars='Classe', var_name='Feature', value_name='Valor')
# Truncate feature names for display
data_melted['Feature'] = data_melted['Feature'].str[:15]
sns.boxplot(data=data_melted, x='Feature', y='Valor', hue='Classe', ax=ax)
ax.set_title('Top 5 Features por Classe')
ax.tick_params(axis='x', rotation=45)
ax.legend(loc='upper right', fontsize=8)

# 9e. Distribuição do split
ax = axes[1, 1]
split_data = pd.DataFrame({
    'Split': ['Treino\nSC', 'Treino\nNão-SC', 'Teste\nSC', 'Teste\nNão-SC'],
    'Count': [y_train.sum(), len(y_train)-y_train.sum(), y_test.sum(), len(y_test)-y_test.sum()]
})
colors_split = ['#2ecc71', '#e74c3c', '#27ae60', '#c0392b']
ax.bar(split_data['Split'], split_data['Count'], color=colors_split, edgecolor='black')
ax.set_title('Distribuição do Split Treino/Teste')
ax.set_ylabel('Número de Amostras')
for i, v in enumerate(split_data['Count']):
    ax.text(i, v + 5, str(v), ha='center', fontsize=9)

# 9f. Resumo textual
ax = axes[1, 2]
ax.axis('off')
info_text = f"""RESUMO DO PIPELINE v2

Dataset: {len(df)} materiais
  SC confirmados (Tc>0): {tc_positive}
  Não-SC confirmados (Tc=0): {tc_zero}
  Não testados (Tc=NaN): {tc_nan}

Label binário:
  SC (Tc>0): {n_sc} ({100*n_sc/len(df):.1f}%)
  Não-SC (Tc=0 ou NaN): {n_nsc} ({100*n_nsc/len(df):.1f}%)

Features: {len(feature_cols)}
  Removidas (>50% missing): {len(high_missing)}
  Removidas (var=0): {len(zero_var)}

Split (GroupShuffleSplit):
  Treino: {len(X_train)} ({100*y_train.mean():.1f}% SC)
  Teste: {len(X_test)} ({100*y_test.mean():.1f}% SC)
  Data leakage: CONTROLADO
  Overlap de grupos: {len(overlap)}
"""
ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_PATH, '00_dataset_summary.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Gráfico salvo: graficos/00_dataset_summary.png")

# =============================================================================
# 10. RESUMO FINAL
# =============================================================================
print("\n" + "="*70)
print("PIPELINE CONCLUÍDO COM SUCESSO")
print("="*70)

summary = {
    'dataset_original': int(len(df)),
    'tc_positive': int(tc_positive),
    'tc_zero': int(tc_zero),
    'tc_nan': int(tc_nan),
    'n_supercondutores': int(n_sc),
    'n_nao_supercondutores': int(n_nsc),
    'n_features': int(len(feature_cols)),
    'features_removed_missing': int(len(high_missing)),
    'features_removed_zero_var': int(len(zero_var)),
    'n_train': int(len(X_train)),
    'n_test': int(len(X_test)),
    'pct_sc_train': float(y_train.mean()),
    'pct_sc_test': float(y_test.mean()),
    'data_leakage_overlap': int(len(overlap)),
    'random_state': RANDOM_STATE,
    'feature_names': feature_cols
}

with open('../dados_preprocessados/pipeline_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nResumo salvo em: dados_preprocessados/pipeline_summary.json")

# Print key info for LaTeX
print(f"\n--- VALORES PARA O TEXTO ---")
print(f"N total: {len(df)}")
print(f"N features: {len(feature_cols)}")
print(f"N treino: {len(X_train)}")
print(f"N teste: {len(X_test)}")
print(f"% SC treino: {100*y_train.mean():.1f}")
print(f"% SC teste: {100*y_test.mean():.1f}")
