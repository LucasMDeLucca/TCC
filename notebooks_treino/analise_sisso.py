#!/usr/bin/env python3
"""
SISSO (Sure Independence Screening and Sparsifying Operator) Analysis
======================================================================
Este script implementa o modelo SISSO para descobrir expressões analíticas
que relacionam os descritores físicos à temperatura crítica de supercondutores.

Autor: TCC Supercondutores
Data: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
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
    return Path(__file__).resolve().parent.parent  # fallback: notebooks_treino/../

PROJECT_ROOT = get_project_root()
DIR_DADOS = PROJECT_ROOT / 'dados'
GRAFICOS_DIR = PROJECT_ROOT / 'graficos'
GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

# Configuração de estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("=" * 70)
print("SISSO - Sure Independence Screening and Sparsifying Operator")
print("Análise de Supercondutores")
print("=" * 70)

# =============================================================================
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS
# =============================================================================
print("\n[1] Carregando e pré-processando dados...")

# Carregar dados
df = pd.read_csv(DIR_DADOS / 'data_treino.csv')
print(f"    Dataset carregado: {df.shape[0]} amostras, {df.shape[1]} colunas")

# Identificar a coluna target (Tc)
# Procurar por colunas que podem ser Tc
tc_candidates = [col for col in df.columns if 'tc' in col.lower() or 'critical' in col.lower() or 'temperature' in col.lower()]
print(f"    Candidatos a Tc: {tc_candidates}")

# Se não encontrar, usar a primeira coluna numérica ou a coluna 'critical_temp'
if 'critical_temp' in df.columns:
    target_col = 'critical_temp'
elif tc_candidates:
    target_col = tc_candidates[0]
else:
    # Verificar as primeiras colunas
    print(f"    Primeiras colunas: {df.columns[:5].tolist()}")
    target_col = df.columns[0]

print(f"    Coluna target identificada: {target_col}")

# Remover colunas não numéricas e identificadores
cols_to_drop = ['group_id', 'id', 'material', 'formula', 'name']
cols_to_drop = [col for col in cols_to_drop if col in df.columns]
df_clean = df.drop(columns=cols_to_drop, errors='ignore')

# Separar features e target
if target_col in df_clean.columns:
    y = df_clean[target_col].values
    X = df_clean.drop(columns=[target_col])
else:
    # Assumir que a primeira coluna é o target
    y = df_clean.iloc[:, 0].values
    X = df_clean.iloc[:, 1:]

# Remover linhas com valores ausentes no target
mask = ~np.isnan(y)
y = y[mask]
X = X.loc[mask]

# Para SISSO, precisamos de materiais com Tc > 0 (supercondutores)
# Vamos fazer regressão apenas para supercondutores
mask_sc = y > 0
y_sc = y[mask_sc]
X_sc = X.loc[mask_sc].reset_index(drop=True)

print(f"    Supercondutores (Tc > 0): {len(y_sc)} amostras")
print(f"    Features disponíveis: {X_sc.shape[1]}")

# Selecionar as features mais importantes (baseado na análise anterior)
# Usar apenas features numéricas e remover colunas com muitos NaN
X_sc = X_sc.select_dtypes(include=[np.number])
X_sc = X_sc.dropna(axis=1, thresh=int(0.5 * len(X_sc)))  # Manter colunas com pelo menos 50% de dados
X_sc = X_sc.fillna(X_sc.median())

print(f"    Features após limpeza: {X_sc.shape[1]}")

# Selecionar as top 10 features mais correlacionadas com Tc
correlations = X_sc.corrwith(pd.Series(y_sc)).abs().sort_values(ascending=False)
top_features = correlations.head(10).index.tolist()
print(f"    Top 10 features mais correlacionadas com Tc:")
for i, feat in enumerate(top_features):
    print(f"        {i+1}. {feat}: {correlations[feat]:.4f}")

# Usar apenas as top features para SISSO (para evitar explosão combinatória)
X_sisso = X_sc[top_features]

# =============================================================================
# 2. DIVISÃO DOS DADOS
# =============================================================================
print("\n[2] Dividindo dados em treino e teste...")

X_train, X_test, y_train, y_test = train_test_split(
    X_sisso, y_sc, test_size=0.2, random_state=42
)
print(f"    Treino: {len(X_train)} amostras")
print(f"    Teste: {len(X_test)} amostras")

# =============================================================================
# 3. IMPLEMENTAÇÃO DO SISSO
# =============================================================================
print("\n[3] Executando SISSO...")

try:
    from TorchSisso import SissoModel
    
    # Preparar DataFrame para SISSO (target na primeira coluna)
    df_sisso_train = pd.DataFrame(y_train, columns=['Tc'])
    for col in X_train.columns:
        df_sisso_train[col] = X_train[col].values
    
    # Definir operadores (mantendo simples para evitar explosão)
    operators = ['+', '-', '*', '/', 'pow(2)', 'sqrt', 'ln']
    
    print("    Configurando modelo SISSO...")
    print(f"    Operadores: {operators}")
    print(f"    Features: {list(X_train.columns)}")
    
    # Criar modelo SISSO com parâmetros conservadores
    sm = SissoModel(
        df=df_sisso_train,
        operators=operators,
        n_expansion=2,  # Número de expansões de features
        n_term=2,       # Número de termos na equação final
        k=15,           # Número de features SIS para regularização L0
        use_gpu=False
    )
    
    # Treinar modelo
    print("    Treinando modelo SISSO (isso pode levar alguns minutos)...")
    rmse_train, equation, r2_train, _ = sm.fit()
    
    print(f"\n    === RESULTADOS DO SISSO ===")
    print(f"    Equação descoberta: {equation}")
    print(f"    RMSE (treino): {rmse_train:.4f}")
    print(f"    R² (treino): {r2_train:.4f}")
    
    sisso_success = True
    sisso_equation = str(equation)
    
except Exception as e:
    print(f"    ERRO ao executar SISSO: {e}")
    print("    Tentando abordagem alternativa com regressão simbólica simplificada...")
    sisso_success = False
    sisso_equation = None

# =============================================================================
# 4. ABORDAGEM ALTERNATIVA: REGRESSÃO SIMBÓLICA SIMPLIFICADA
# =============================================================================
if not sisso_success:
    print("\n[4] Implementando regressão simbólica simplificada...")
    
    from sklearn.linear_model import LassoCV, RidgeCV
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    
    # Normalizar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Criar features polinomiais (grau 2)
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    
    print(f"    Features originais: {X_train_scaled.shape[1]}")
    print(f"    Features polinomiais: {X_train_poly.shape[1]}")
    
    # LASSO para seleção esparsa de features
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_train_poly, y_train)
    
    # Predições
    y_pred_train = lasso.predict(X_train_poly)
    y_pred_test = lasso.predict(X_test_poly)
    
    # Métricas
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    print(f"\n    === RESULTADOS DA REGRESSÃO SIMBÓLICA SIMPLIFICADA ===")
    print(f"    RMSE (treino): {rmse_train:.4f} K")
    print(f"    RMSE (teste): {rmse_test:.4f} K")
    print(f"    R² (treino): {r2_train:.4f}")
    print(f"    R² (teste): {r2_test:.4f}")
    print(f"    MAE (teste): {mae_test:.4f} K")
    
    # Identificar features mais importantes (coeficientes não-zero)
    feature_names = poly.get_feature_names_out(X_train.columns)
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coeficiente': lasso.coef_
    })
    coef_df = coef_df[coef_df['Coeficiente'] != 0].sort_values('Coeficiente', key=abs, ascending=False)
    
    print(f"\n    Features selecionadas pelo LASSO ({len(coef_df)} de {len(feature_names)}):")
    for i, row in coef_df.head(10).iterrows():
        print(f"        {row['Feature']}: {row['Coeficiente']:.4f}")
    
    # Construir equação aproximada
    equation_terms = []
    for i, row in coef_df.head(5).iterrows():
        if row['Coeficiente'] > 0:
            equation_terms.append(f"+ {row['Coeficiente']:.3f}*{row['Feature']}")
        else:
            equation_terms.append(f"{row['Coeficiente']:.3f}*{row['Feature']}")
    
    sisso_equation = f"Tc ≈ {lasso.intercept_:.3f} " + " ".join(equation_terms)
    print(f"\n    Equação aproximada:")
    print(f"    {sisso_equation}")

# =============================================================================
# 5. VISUALIZAÇÕES
# =============================================================================
print("\n[5] Gerando visualizações...")

# Gráfico 1: Predito vs Real
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Treino
axes[0].scatter(y_train, y_pred_train, alpha=0.5, edgecolors='k', linewidth=0.5)
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0].set_xlabel('Tc Real (K)', fontsize=12)
axes[0].set_ylabel('Tc Predito (K)', fontsize=12)
axes[0].set_title(f'Conjunto de Treino\nR² = {r2_train:.4f}, RMSE = {rmse_train:.2f} K', fontsize=12)

# Teste
axes[1].scatter(y_test, y_pred_test, alpha=0.5, edgecolors='k', linewidth=0.5, color='green')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Tc Real (K)', fontsize=12)
axes[1].set_ylabel('Tc Predito (K)', fontsize=12)
axes[1].set_title(f'Conjunto de Teste\nR² = {r2_test:.4f}, RMSE = {rmse_test:.2f} K', fontsize=12)

plt.suptitle('SISSO/Regressão Simbólica: Predição de Tc', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / '20_sisso_predicao_tc.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Salvo: 20_sisso_predicao_tc.png")

# Gráfico 2: Distribuição dos erros
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

errors_train = y_train - y_pred_train
errors_test = y_test - y_pred_test

axes[0].hist(errors_train, bins=30, edgecolor='black', alpha=0.7)
axes[0].axvline(x=0, color='r', linestyle='--', lw=2)
axes[0].set_xlabel('Erro (K)', fontsize=12)
axes[0].set_ylabel('Frequência', fontsize=12)
axes[0].set_title(f'Distribuição de Erros (Treino)\nMédia = {np.mean(errors_train):.2f} K', fontsize=12)

axes[1].hist(errors_test, bins=30, edgecolor='black', alpha=0.7, color='green')
axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Erro (K)', fontsize=12)
axes[1].set_ylabel('Frequência', fontsize=12)
axes[1].set_title(f'Distribuição de Erros (Teste)\nMédia = {np.mean(errors_test):.2f} K', fontsize=12)

plt.suptitle('SISSO/Regressão Simbólica: Análise de Erros', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / '21_sisso_erros.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Salvo: 21_sisso_erros.png")

# Gráfico 3: Importância das features (coeficientes)
if not sisso_success and len(coef_df) > 0:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    top_coef = coef_df.head(15)
    colors = ['green' if c > 0 else 'red' for c in top_coef['Coeficiente']]
    
    bars = ax.barh(range(len(top_coef)), top_coef['Coeficiente'].values, color=colors, edgecolor='black')
    ax.set_yticks(range(len(top_coef)))
    ax.set_yticklabels(top_coef['Feature'].values, fontsize=10)
    ax.set_xlabel('Coeficiente', fontsize=12)
    ax.set_title('SISSO/Regressão Simbólica: Features Mais Importantes\n(Verde = Positivo, Vermelho = Negativo)', fontsize=12)
    ax.axvline(x=0, color='black', linestyle='-', lw=1)
    
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / '22_sisso_coeficientes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Salvo: 22_sisso_coeficientes.png")

# =============================================================================
# 6. SALVAR RESULTADOS
# =============================================================================
print("\n[6] Salvando resultados...")

results = {
    'Modelo': 'SISSO/Regressão Simbólica',
    'RMSE_Treino': rmse_train,
    'RMSE_Teste': rmse_test,
    'R2_Treino': r2_train,
    'R2_Teste': r2_test,
    'MAE_Teste': mae_test,
    'Equacao': sisso_equation,
    'N_Features_Selecionadas': len(coef_df) if not sisso_success else 'N/A'
}

# Salvar resumo
with open(GRAFICOS_DIR / 'resumo_sisso.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("SISSO - Resultados da Regressão Simbólica\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("MÉTRICAS DE DESEMPENHO:\n")
    f.write(f"  RMSE (Treino): {rmse_train:.4f} K\n")
    f.write(f"  RMSE (Teste):  {rmse_test:.4f} K\n")
    f.write(f"  R² (Treino):   {r2_train:.4f}\n")
    f.write(f"  R² (Teste):    {r2_test:.4f}\n")
    f.write(f"  MAE (Teste):   {mae_test:.4f} K\n\n")
    
    f.write("EQUAÇÃO DESCOBERTA:\n")
    f.write(f"  {sisso_equation}\n\n")
    
    if not sisso_success:
        f.write("FEATURES SELECIONADAS (Top 10):\n")
        for i, row in coef_df.head(10).iterrows():
            f.write(f"  {row['Feature']}: {row['Coeficiente']:.4f}\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("INTERPRETAÇÃO FÍSICA:\n")
    f.write("=" * 70 + "\n")
    f.write("""
A regressão simbólica/SISSO busca encontrar expressões analíticas que
relacionam os descritores físicos à temperatura crítica Tc. Isso é
particularmente valioso porque:

1. INTERPRETABILIDADE: Ao contrário de modelos de caixa-preta (Random Forest,
   XGBoost), o SISSO produz uma equação explícita que pode ser interpretada
   fisicamente.

2. DESCOBERTA DE LEIS: O SISSO pode potencialmente redescobrir relações
   conhecidas (como a fórmula de McMillan para Tc) ou sugerir novas relações
   entre descritores.

3. CONEXÃO COM A TEORIA: As features selecionadas e seus coeficientes podem
   ser comparados com previsões teóricas (e.g., teoria BCS).

LIMITAÇÕES:
- O espaço de busca é exponencialmente grande, limitando a complexidade
  das expressões que podem ser exploradas.
- A qualidade do resultado depende fortemente dos descritores disponíveis.
- Para supercondutores não convencionais, onde a teoria BCS não se aplica
  diretamente, as expressões descobertas podem ter interpretação limitada.
""")

print(f"    Salvo: resumo_sisso.txt")

print("\n" + "=" * 70)
print("ANÁLISE SISSO CONCLUÍDA")
print("=" * 70)
print(f"\nEquação descoberta: {sisso_equation}")
print(f"R² (teste): {r2_test:.4f}")
print(f"RMSE (teste): {rmse_test:.2f} K")
