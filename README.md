# TCC — Classificação de Materiais Supercondutores via Aprendizado de Máquina

**Trabalho de Conclusão de Curso** — Bacharelado em Física  
**Universidade Federal do ABC (UFABC)**  
**Autor:** Lucas Martin De Lucca  
**Título:** *Classificação de Supercondutores Utilizando Técnicas de Aprendizado de Máquina: Uma Abordagem da Física Computacional*

---

## Resumo

Este trabalho investiga a aplicação de técnicas de aprendizado de máquina para a classificação binária de materiais supercondutores a partir de descritores estruturais e eletrônicos extraídos do banco de dados SciDB. Seis modelos de classificação (Random Forest, XGBoost, LightGBM, SVM, MLP e LASSO) e um modelo de regressão simbólica (SISSO) são implementados, otimizados e comparados estatisticamente. A análise de interpretabilidade via SHAP revela que a Densidade de Estados (DOS) no nível de Fermi é o descritor mais relevante, corroborando a teoria BCS da supercondutividade.

---

## Estrutura do Repositório

```
TCC/
├── tcc_final.tex                    # Documento LaTeX do TCC (72 páginas)
├── tcc_final.pdf                    # PDF compilado
├── requirements.txt                 # Dependências Python
├── README.md                        # Este arquivo
├── .gitignore                       # Arquivos ignorados pelo Git
│
├── dados/
│   └── data_treino.csv              # Dataset bruto (2474 materiais, 116 features numéricas + 12 colunas string `atoms_*`; reduzido a 53 após pré-processamento)
│
├── dados_preprocessados/            # Outputs do pipeline (gerados pelos scripts)
│   ├── dados_tcc_v2.npz             # Arrays NumPy (X_train, X_test, y_train, y_test)
│   ├── scaler_v2.pkl                # StandardScaler ajustado no treino
│   ├── selected_features_v2.pkl     # Nomes das 53 features pós-preprocessing
│   ├── best_params.json             # Melhores hiperparâmetros (RandomizedSearchCV)
│   ├── resultados_v2.json           # Métricas de performance de todos os modelos
│   ├── comparacao_estatistica.json  # Testes McNemar, Friedman, Bootstrap CI
│   ├── interpretabilidade_results.json  # Resultados SHAP e calibração
│   └── sisso_results.json           # Resultados da regressão simbólica
│
├── notebooks_treino/                # Scripts Python do pipeline (rodar nesta ordem)
│   ├── 00_pipeline_preprocessamento.py  # Pré-processamento e split (gera dados/data_treino_v2.csv e dados_preprocessados/dados_tcc_v2.npz)
│   ├── 09_hyperparameter_tuning.py      # Otimização de hiperparâmetros
│   ├── train_all_models.py              # Re-treinamento com params otimizados
│   ├── 07_analise_comparativa.py        # Análise estatística comparativa
│   ├── 08_interpretabilidade.py         # SHAP, calibração, curvas de aprendizado
│   └── analise_sisso.py                 # Regressão simbólica (SISSO)
│
├── graficos/                        # Gráficos referenciados pelo TCC (PNG, 300 DPI)
│   ├── 01_distribuicao_tc.png       # Distribuição de Tc no dataset
│   ├── 02_correlacao_features.png   # Mapa de calor de correlação
│   ├── 03_boxplot_features.png      # Boxplots das features principais
│   ├── 07_comparacao_metricas.png   # Comparação de métricas
│   ├── 07_curvas_roc_comparacao.png # Curvas ROC sobrepostas
│   ├── 07_curvas_pr_comparacao.png  # Curvas PR sobrepostas
│   ├── 07_matrizes_confusao.png     # Matrizes de confusão (2x3)
│   ├── 07_cv_boxplot.png            # Boxplot validação cruzada
│   ├── 07_mcnemar_pvalues.png       # Teste de McNemar
│   ├── 07_bootstrap_ci.png          # Intervalos de confiança
│   ├── 07_ranking_modelos.png       # Ranking Friedman
│   ├── 08_analise_erro.png          # Análise de erro (falsos negativos)
│   ├── 08_feature_importance_comparison.png  # Importância comparada entre modelos
│   ├── 18_partial_dependence_plots.png  # PDPs para Random Forest
│   ├── 20_sisso_predicao_tc.png     # Predição SISSO
│   ├── 22_sisso_coeficientes.png    # Coeficientes SISSO
│   ├── 24_shap_summary.png          # SHAP summary plot
│   ├── 27_calibration.png           # Diagrama de confiabilidade
│   └── 28_learning_curve.png        # Curva de aprendizado
│
├── modelos/                         # Modelos treinados (.pkl)
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   ├── svm_model.pkl
│   ├── mlp_model.pkl
│   └── lasso_model.pkl
│
├── ler_arquivo.ipynb                # Extração inicial dos dados do HDF5 do SciDB (executada uma vez para gerar data_treino.csv)
└── discovery.ipynb                  # Análise exploratória inicial e limpeza dos dados
```

---

## Resultados Principais

### Performance dos Modelos (Conjunto de Teste)

| Modelo | Accuracy | F1-Score | Precision | Recall | ROC-AUC |
|--------|----------|----------|-----------|--------|---------|
| **Random Forest** | 0.6242 | **0.7232** | 0.6090 | 0.8901 | 0.6509 |
| XGBoost | **0.6586** | 0.7081 | **0.6699** | 0.7509 | **0.7243** |
| LightGBM | 0.6545 | 0.7005 | 0.6711 | 0.7326 | 0.7068 |
| SVM (RBF) | 0.5697 | 0.7164 | 0.5628 | 0.9853 | 0.6655 |
| MLP | 0.5576 | 0.6831 | 0.5646 | 0.8645 | 0.5575 |
| LASSO (L1) | 0.5515 | 0.7109 | 0.5515 | 1.0000 | 0.5000 |

### Análise Estatística

- **Teste de Friedman**: χ² = 10.83, p = 0.055 (sem diferença significativa entre modelos)
- **Bootstrap 95% CI (F1)**: Random Forest [0.682, 0.762], XGBoost [0.663, 0.751]
- **Melhor modelo**: Random Forest (F1 = 0.723, melhor equilíbrio precision/recall)

### SISSO (Regressão Simbólica)

| Dimensionalidade | R² (CV) | R² (Teste) | RMSE (K) | Features Selecionadas |
|------------------|---------|------------|----------|----------------------|
| 1D (Linear) | 0.178 | 0.189 | 15.48 | 9/10 |
| 2D (Quadrático) | 0.348 | 0.359 | 13.76 | 38/65 |
| 3D (Cúbico) | 0.450 | 0.528 | 11.81 | 33/175 |

---

## Como Reproduzir

### 1. Clonar o repositório

```bash
git clone https://github.com/LucasMDeLucca/TCC.git
cd TCC
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Executar pipeline completo

```bash
cd notebooks_treino/

# 1. Pré-processamento
python3 00_pipeline_preprocessamento.py

# 2. Otimização de hiperparâmetros (~10 min)
python3 09_hyperparameter_tuning.py

# 3. Treinamento com params otimizados
python3 train_all_models.py

# 4. Análise comparativa estatística
python3 07_analise_comparativa.py

# 5. Interpretabilidade (SHAP, calibração)
python3 08_interpretabilidade.py

# 6. SISSO (regressão simbólica)
python3 analise_sisso.py
```

### 4. Compilar o TCC (LaTeX)

```bash
pdflatex tcc_final.tex
pdflatex tcc_final.tex   # Segunda compilação para referências cruzadas
```

Ou use o [Overleaf](https://www.overleaf.com/) para compilação online.

---

## Descrição dos Scripts

| Script | Função |
|--------|--------|
| `00_pipeline_preprocessamento.py` | Carrega `data_treino.csv`, define classes (Tc>0 = SC; Tc=0 ou Tc=NaN tratados como não-SC — aproximação reconhecida no texto, ver Cap. 4.3 do TCC), trata missing values, normaliza com StandardScaler, faz split 80/20 com controle de data leakage via `GroupShuffleSplit` (separação por `group_id`). Reporta tc_positive=1362, tc_zero=1, tc_nan=1111. |
| `09_hyperparameter_tuning.py` | RandomizedSearchCV com 20 iterações e 5-fold CV para cada modelo. Salva melhores parâmetros em `best_params.json` |
| `train_all_models.py` | Re-treina todos os 6 modelos com parâmetros otimizados. Gera gráficos individuais (confusão, ROC, PR) e comparativos |
| `07_analise_comparativa.py` | Validação cruzada 5-fold, teste de McNemar, teste de Friedman + Nemenyi, intervalos de confiança Bootstrap 95% |
| `08_interpretabilidade.py` | SHAP (summary, dependence), calibração de probabilidades (Brier score), curvas de aprendizado |
| `analise_sisso.py` | Regressão simbólica SISSO-like com features polinomiais + LASSO. Testa dimensionalidades 1D-3D e extrai equação analítica |

---

## Origem dos Dados

Os dados foram extraídos do **SciDB** (Science Data Bank, Academia Chinesa de Ciências). O arquivo original, em formato HDF5 (~12 GB), contém propriedades de estrutura eletrônica calculadas via Teoria do Funcional da Densidade (DFT).

A pipeline de extração está documentada em:
1. `ler_arquivo.ipynb` — Leitura do HDF5 e extração de features
2. `discovery.ipynb` — Análise exploratória e geração do `data_treino.csv`

---

## Referências Principais

- Bardeen, J., Cooper, L. N., & Schrieffer, J. R. (1957). *Theory of Superconductivity*. Physical Review, 108(5), 1175.
- Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD 2016.
- Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS 2017.
- Stanev, V., et al. (2018). *Machine learning modeling of superconducting critical temperature*. npj Computational Materials, 4(1), 29.

---

## Licença

Este trabalho é de uso acadêmico. Para reutilização, entre em contato com o autor.
