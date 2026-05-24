# Avaliação Crítica do TCC

> **AVISO:** Esta avaliação anterior (nota 8.6/10) foi feita em um ponto intermediário da revisão e está **desatualizada**. Após revisão criteriosa pós-merge dos PRs #2, #3 e #4, a nota recalibrada é **7.6/10** (ver detalhes abaixo). Os principais fatores são: artefatos em excesso no repositório, scripts obsoletos coexistindo com os atuais, e algumas discrepâncias entre o texto e os JSONs de resultados que foram subsequentemente corrigidas.

**Avaliador:** Revisor externo (doutorado em Física)
**Última atualização:** 2026-05-24

---

## Nota Recalibrada

| Critério | Peso | Nota | Justificativa |
|----------|------|------|---------------|
| Rigor Teórico (Física) | 20% | 8.5 | Revisão BCS / London / GL (em SI) corretas. Tabela de comprimentos de coerência e discussão de nós d-wave/$s_\pm$ incluídas. |
| Rigor Teórico (ML) | 20% | 8.0 | Formulação de todos os 6 modelos com referências originais. SVM com soft-margin explicitado. Falta breve discussão sobre limites teóricos de convergência. |
| Metodologia | 15% | 7.5 | Excelente controle de data leakage via `GroupShuffleSplit`. Tuning real via `RandomizedSearchCV`. **Mas:** tratamento Tc=NaN aproximado (PU Learning apenas mencionado); LASSO degenerou em trivial após tuning; escolha de `scoring='f1'` favorece recall sobre discriminação. |
| Resultados e Discussão | 20% | 7.5 | Análise estatística (Friedman, McNemar, Bootstrap 95%) e SHAP/calibração/learning-curve incluídas. SISSO atualizado para refletir JSON real. |
| Qualidade da Escrita | 10% | 7.5 | Bibliografia em ABNT NBR 6023. Estrutura clara. Capa ainda exige preencher nome do orientador (TODO). |
| Originalidade | 10% | 7.5 | Contribuição incremental honesta. Valor está na transparência metodológica. |
| Reprodutibilidade | 5% | 7.5 | README atualizado. `requirements.txt` cobre dependências do pipeline atual. |

**Nota ponderada: 7.65/10**

---

## Pontos Fortes

1. **Controle rigoroso de data leakage** via `GroupShuffleSplit` — diferencial de qualidade científica que poucos TCCs implementam.
2. **Pipeline reprodutível** com scripts numerados (00 → 09 → train_all_models → 07 → 08).
3. **Análise estatística rigorosa**: McNemar pareado, Friedman + Nemenyi, Bootstrap CI 95% (1000 reamostragens).
4. **Interpretabilidade**: SHAP summary/waterfall/dependence, calibração com Brier score, learning curve.
5. **Conexão física**: a importância da DOS no nível de Fermi (sc_DOSs_*) emerge dos modelos sem ser imposta, consistente com a expectativa BCS.
6. **Reconhecimento honesto das limitações**: o texto agora discute o tratamento aproximado de Tc=NaN, o colapso do LASSO como informação científica, e a queda de métricas pós-controle de leakage.

## Pontos a Melhorar (trabalhos futuros)

1. **Implementar PU Learning de verdade** para tratar materiais Tc=NaN como não-rotulados.
2. **Re-tunar com `scoring='roc_auc'`** ou expandir espaço de busca do LASSO para evitar degeneração trivial.
3. **Incluir descritores químicos compostos** (eletronegatividade, raio atômico etc. — não disponíveis no SciDB) para comparação com Stanev et al.
4. **Validação em dataset externo** (e.g., NIMS SuperCon) para avaliar generalização cross-domain.
5. **Deep learning** (e.g., Graph Neural Networks para estruturas cristalinas).

---

## Conclusão

Trabalho com mérito científico real, particularmente na metodologia computacional e na honestidade da análise. As correções realizadas em três rodadas de revisão sucessivas (PRs #2, #3 e #4) elevaram substancialmente a qualidade técnica em relação à versão preliminar. A nota 7.65/10 reflete um trabalho **bom**, sólido em fundamentos, mas com espaço para polimento editorial e refinamentos metodológicos antes de defesa.
