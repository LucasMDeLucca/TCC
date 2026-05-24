#!/usr/bin/env bash
# cleanup_p3.sh
# Remove artefatos da revisão crítica que vazaram para o repo,
# scripts/notebooks obsoletos, dados e diretórios legados, e figuras órfãs.
#
# Uso:
#   1. Aplique primeiro o patch:  git apply tcc_revisao_critica_p3.patch
#   2. Em seguida, rode:           bash cleanup_p3.sh
#
# O script usa `git rm` para que as remoções fiquem no índice do git.
# Você ainda precisa fazer `git commit` ao final.

set -eu

echo "=== [1/6] Removendo artefatos da revisão que vazaram para o main ==="
for f in \
    tcc_final_corrigido.tex \
    tcc_final_corrigido.pdf \
    tcc_final_corrigido_v2.tex \
    tcc_final_corrigido_v2.pdf \
    tcc_revisao_critica.patch \
    tcc_revisao_critica_p2.patch ; do
    if [ -f "$f" ]; then
        git rm "$f"
    fi
done

echo ""
echo "=== [2/6] Removendo scripts e notebooks obsoletos ==="
for f in \
    notebooks_treino/analise_completa.py \
    notebooks_treino/analise_interpretabilidade.py \
    notebooks_treino/analise_pdp_erro.py \
    notebooks_treino/00_pipeline_preprocessamento.ipynb \
    notebooks_treino/07_analise_comparativa.ipynb \
    notebooks_treino/08_interpretabilidade.ipynb ; do
    if [ -f "$f" ]; then
        git rm "$f"
    fi
done

echo ""
echo "=== [3/6] Removendo dados pre-processados obsoletos (versões v1) ==="
for f in \
    dados_preprocessados/dados_tcc.npz \
    dados_preprocessados/scaler.pkl \
    dados_preprocessados/selected_features.pkl ; do
    if [ -f "$f" ]; then
        git rm "$f"
    fi
done

echo ""
echo "=== [4/6] Removendo diretórios legados da versão embrionária ==="
if [ -d graficos_supercondutores ]; then
    git rm -rf graficos_supercondutores/
fi
if [ -d modelos_supercondutores ]; then
    git rm -rf modelos_supercondutores/
fi

echo ""
echo "=== [5/6] Removendo figuras órfãs em graficos/ (não referenciadas no TCC) ==="
# Lista derivada de comparar `\includegraphics{...}` no .tex com `ls graficos/`.
# Apenas figs NÃO referenciadas são removidas.
ORPHAN_FIGS=(
    graficos/01_distribuicao_classes.png
    graficos/01_random_forest_confusion_matrix.png
    graficos/01_random_forest_feature_importance.png
    graficos/01_random_forest_pr_curve.png
    graficos/01_random_forest_roc_curve.png
    graficos/01_rf_confusion_matrix.png
    graficos/01_rf_feature_importance.png
    graficos/01_rf_pr_curve.png
    graficos/01_rf_roc_curve.png
    graficos/02_xgb_confusion_matrix.png
    graficos/02_xgb_feature_importance.png
    graficos/02_xgb_roc_curve.png
    graficos/02_xgboost_confusion_matrix.png
    graficos/02_xgboost_feature_importance.png
    graficos/02_xgboost_pr_curve.png
    graficos/02_xgboost_roc_curve.png
    graficos/03_lightgbm_confusion_matrix.png
    graficos/03_lightgbm_feature_importance.png
    graficos/03_lightgbm_pr_curve.png
    graficos/03_lightgbm_roc_curve.png
    graficos/03_svm_confusion_matrix.png
    graficos/03_svm_roc_curve.png
    graficos/04_lgb_feature_importance.png
    graficos/04_svm_confusion_matrix.png
    graficos/04_svm_pr_curve.png
    graficos/04_svm_roc_curve.png
    graficos/05_mlp_confusion_matrix.png
    graficos/05_mlp_loss_curve.png
    graficos/05_mlp_pr_curve.png
    graficos/05_mlp_prob_distribution.png
    graficos/05_mlp_roc_curve.png
    graficos/06_lasso_coef_distribution.png
    graficos/06_lasso_coefficients.png
    graficos/06_lasso_confusion_matrix.png
    graficos/06_lasso_feature_importance.png
    graficos/06_lasso_pr_curve.png
    graficos/06_lasso_prob_distribution.png
    graficos/06_lasso_roc_curve.png
    graficos/07_heatmap_metricas.png
    graficos/08_features_acertos_vs_erros.png
    graficos/08_partial_dependence_plots.png
    graficos/08_shap_summary_bar.png
    graficos/10_importancia_lightgbm.png
    graficos/12_shap_summary_bar.png
    graficos/13_shap_summary_dot.png
    graficos/14_shap_dependence_0_fermi_line_std.png
    graficos/14_shap_dependence_1_sc_DOSs_max.png
    graficos/14_shap_dependence_2_sc_DOSs_mean.png
    graficos/14_shap_dependence_3_position_3_2.png
    graficos/15_shap_force_plot_vp.png
    graficos/16_shap_force_plot_fn.png
    graficos/17_lime_explanation_fn.png
    graficos/21_sisso_erros.png
    graficos/23_nemenyi_diagram.png
    graficos/04_comparacao_metricas.png
    graficos/05_curvas_roc.png
    graficos/06_curvas_precision_recall.png
    graficos/08_ranking_modelos.png
    graficos/09_importancia_features.png
    graficos/11_validacao_cruzada.png
    graficos/19_analise_erro.png
)
for f in "${ORPHAN_FIGS[@]}"; do
    if [ -f "$f" ]; then
        git rm "$f"
    fi
done

echo ""
echo "=== [6/6] Recompilando PDF para refletir o .tex atualizado ==="
if command -v pdflatex >/dev/null 2>&1; then
    pdflatex -interaction=nonstopmode tcc_final.tex >/tmp/p3_pdflatex.log 2>&1 || true
    pdflatex -interaction=nonstopmode tcc_final.tex >/tmp/p3_pdflatex.log 2>&1 || true
    # limpar arquivos auxiliares
    rm -f tcc_final.aux tcc_final.log tcc_final.lof tcc_final.lot tcc_final.out tcc_final.toc
    git add tcc_final.pdf
    echo "    PDF recompilado e adicionado ao índice."
else
    echo "    AVISO: pdflatex não encontrado. Recompile o PDF manualmente:"
    echo "       pdflatex tcc_final.tex && pdflatex tcc_final.tex"
    echo "       git add tcc_final.pdf"
fi

echo ""
echo "============================================================"
echo "Limpeza concluída. Status do índice:"
git status --short
echo ""
echo "Próximos passos:"
echo "  git commit -m 'chore: limpa artefatos de revisão, scripts obsoletos e figuras órfãs'"
echo "  git push origin HEAD"
