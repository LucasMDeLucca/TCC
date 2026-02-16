# TCC — Classificação de Supercondutores com Machine Learning

**Trabalho de Conclusão de Curso** apresentado ao Curso de Bacharelado em Física da Universidade Federal do ABC (UFABC).

**Autor:** Lucas Martin De Lucca  
**Título:** *Classificação de Supercondutores Utilizando Técnicas de Aprendizado de Máquina: Uma Abordagem da Física Computacional*

---

## Resumo

Este trabalho investiga a aplicação de sete modelos de aprendizado de máquina — Random Forest, XGBoost, LightGBM, SVM, MLP, LASSO e SISSO — para classificar materiais como supercondutores ou não-supercondutores a partir de descritores de estrutura eletrônica calculados via DFT. O modelo Random Forest alcançou o melhor desempenho (F1-Score: 0.727, ROC-AUC: 0.734). A análise de interpretabilidade revelou que a densidade de estados eletrônicos (DOS) no nível de Fermi é o descritor mais preditivo, corroborando a teoria BCS.

---

## Estrutura do Repositório

```
TCC/
├── tcc_final.tex                    # Documento LaTeX do TCC (61 páginas)
├── main.tex                         # Proposta original do TCC
├── requirements.txt                 # Dependências Python
├── README.md                        # Este arquivo
│
├── dados/
│   └── data_treino.csv              # Dataset final (2.474 materiais, 130 features)
│
├── dados_preprocessados/
│   ├── dados_tcc.npz                # Dados pré-processados (treino/teste)
│   ├── scaler.pkl                   # StandardScaler ajustado
│   └── selected_features.pkl        # Lista de features selecionadas
│
├── notebooks_treino/
│   ├── 00_pipeline_preprocessamento.ipynb   # Pré-processamento dos dados
│   ├── 01_random_forest.ipynb               # Modelo Random Forest
│   ├── 02_xgboost.ipynb                     # Modelo XGBoost
│   ├── 03_svm.ipynb                         # Modelo SVM (kernel RBF)
│   ├── 04_lightgbm.ipynb                    # Modelo LightGBM
│   ├── 05_mlp.ipynb                         # Modelo MLP (Rede Neural)
│   ├── 06_lasso.ipynb                       # Modelo LASSO (Regressão L1)
│   ├── 07_analise_comparativa.ipynb         # Comparação de todos os modelos
│   ├── 08_interpretabilidade.ipynb          # SHAP, LIME, PDP, análise de erro
│   ├── analise_completa.py                  # Script autônomo (todos os modelos)
│   ├── analise_pdp_erro.py                  # Partial Dependence Plots e erros
│   ├── analise_interpretabilidade.py        # Análise SHAP e LIME
│   └── analise_sisso.py                     # Regressão simbólica (SISSO)
│
├── graficos/                        # Todos os gráficos gerados
│   ├── 01_*.png                     # Gráficos do Random Forest
│   ├── 02_*.png                     # Gráficos do XGBoost
│   ├── ...                          # (demais modelos)
│   ├── 07_*.png                     # Gráficos comparativos
│   ├── 08_*.png                     # Gráficos de interpretabilidade
│   ├── 18_partial_dependence_plots.png
│   ├── 19_analise_erro.png
│   ├── 20_sisso_predicao_tc.png     # Predição SISSO
│   ├── 21_sisso_erros.png           # Erros SISSO
│   ├── 22_sisso_coeficientes.png    # Coeficientes SISSO
│   ├── resultados_modelos.csv       # Métricas de todos os modelos
│   └── resumo_resultados.txt        # Resumo textual dos resultados
│
├── graficos_supercondutores/        # Gráficos da versão embrionária
│
├── modelos/                         # Modelos treinados (.pkl)
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   ├── svm_model.pkl
│   ├── mlp_model.pkl
│   ├── lasso_model.pkl
│   └── *_results.pkl                # Resultados de cada modelo
│
├── modelos_supercondutores/         # Modelos da versão embrionária
│
├── ler_arquivo.ipynb                # Extração dos dados do HDF5 (SciDB)
├── discovery.ipynb                  # Exploração e limpeza dos dados
└── classificacao_supercondutores_corrigido_minimal.ipynb  # Versão embrionária
```

---

## Origem dos Dados

Os dados foram extraídos do repositório **SciDB** (Scientific Data Bank) da Academia Chinesa de Ciências. O arquivo original, em formato HDF5 (~12 GB), contém propriedades de estrutura eletrônica de 2.474 materiais calculadas via Teoria do Funcional da Densidade (DFT).

A pipeline de extração está documentada em:
1. `ler_arquivo.ipynb` — Leitura do HDF5 e extração de 132 features
2. `discovery.ipynb` — Análise exploratória e geração do `data_treino.csv`

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

### 3. Executar os notebooks

Execute os notebooks na pasta `notebooks_treino/` na ordem numérica (00 a 08). Alternativamente, execute o script autônomo:

```bash
cd notebooks_treino
python analise_completa.py
```

### 4. Compilar o TCC (LaTeX)

```bash
pdflatex tcc_final.tex
pdflatex tcc_final.tex   # Segunda compilação para referências cruzadas
```

---

## Resultados Principais

| Modelo | Acurácia | Precisão | Recall | F1-Score | ROC-AUC |
|--------|----------|----------|--------|----------|---------|
| **Random Forest** | **0.6727** | **0.6729** | **0.7912** | **0.7273** | **0.7344** |
| XGBoost | 0.6606 | 0.6744 | 0.7436 | 0.7073 | 0.7270 |
| LightGBM | 0.6545 | 0.6723 | 0.7326 | 0.7011 | 0.7214 |
| SVM | 0.6303 | 0.6471 | 0.7326 | 0.6872 | 0.7061 |
| MLP | 0.6364 | 0.6614 | 0.7033 | 0.6817 | 0.7024 |
| LASSO | 0.6121 | 0.6375 | 0.7033 | 0.6688 | 0.6851 |

---

## Licença

Este trabalho é de uso acadêmico. Para reutilização, entre em contato com o autor.
