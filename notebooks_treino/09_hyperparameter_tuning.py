"""
09_hyperparameter_tuning.py
============================
Otimização de hiperparâmetros para todos os 6 modelos via RandomizedSearchCV.
Salva os melhores parâmetros em best_params.json para uso nos notebooks 01-06.
"""

import numpy as np
import json
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
warnings.filterwarnings('ignore')

# --- Carregar dados pré-processados ---
DATA_PATH = '../dados_preprocessados/dados_tcc_v2.npz'
OUTPUT_PATH = '../dados_preprocessados/best_params.json'

print("="*70)
print("OTIMIZAÇÃO DE HIPERPARÂMETROS — RandomizedSearchCV")
print("="*70)

data = np.load(DATA_PATH, allow_pickle=True)
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

print(f"\nDados carregados: X_train={X_train.shape}, X_test={X_test.shape}")

# --- Configuração ---
N_ITER = 20
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SCORING = 'f1'
RANDOM_STATE = 42

# =============================================================================
# DEFINIÇÃO DOS ESPAÇOS DE BUSCA
# =============================================================================

param_spaces = {
    'random_forest': {
        'model': RandomForestClassifier(random_state=RANDOM_STATE),
        'params': {
            'n_estimators': [100, 200, 500, 1000],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
    },
    'xgboost': {
        'model': XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0),
        'params': {
            'n_estimators': [100, 200, 500],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [0.1, 1, 10]
        }
    },
    'lightgbm': {
        'model': LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1, n_jobs=1),
        'params': {
            'n_estimators': [100, 200, 500],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [0.1, 1, 10],
            'num_leaves': [15, 31, 63, 127]
        }
    },
    'svm': {
        'model': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
        'params': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }
    },
    'mlp': {
        'model': MLPClassifier(random_state=RANDOM_STATE, max_iter=1000, early_stopping=True),
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100), (200, 100, 50)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.0001, 0.001, 0.01],
            'activation': ['relu']
        }
    },
    'lasso': {
        'model': LogisticRegression(penalty='l1', solver='saga', random_state=RANDOM_STATE, max_iter=5000),
        'params': {
            'C': [0.001, 0.01, 0.1, 1, 10]
        }
    }
}

# =============================================================================
# EXECUÇÃO DA BUSCA
# =============================================================================

best_params = {}
best_scores = {}

for name, config in param_spaces.items():
    print(f"\n{'='*50}")
    print(f"Otimizando: {name.upper()}")
    print(f"{'='*50}")
    
    model = config['model']
    params = config['params']
    
    # Calcular n_iter real (não pode ser maior que o espaço total)
    total_combinations = 1
    for v in params.values():
        total_combinations *= len(v)
    n_iter_actual = min(N_ITER, total_combinations)
    
    search = RandomizedSearchCV(
        model, params,
        n_iter=n_iter_actual,
        cv=CV,
        scoring=SCORING,
        random_state=RANDOM_STATE,
        n_jobs=2,
        verbose=0
    )
    
    search.fit(X_train, y_train)
    
    best_params[name] = {}
    for k, v in search.best_params_.items():
        # Convert numpy types to native Python types for JSON serialization
        if isinstance(v, (np.integer,)):
            best_params[name][k] = int(v)
        elif isinstance(v, (np.floating,)):
            best_params[name][k] = float(v)
        elif isinstance(v, (np.ndarray,)):
            best_params[name][k] = v.tolist()
        elif isinstance(v, tuple):
            best_params[name][k] = list(v)
        else:
            best_params[name][k] = v
    
    best_scores[name] = float(search.best_score_)
    
    print(f"  Melhor F1 (CV): {search.best_score_:.4f}")
    print(f"  Melhores parâmetros: {search.best_params_}")

# =============================================================================
# SALVAR RESULTADOS
# =============================================================================

output = {
    'best_params': best_params,
    'best_cv_scores': best_scores,
    'config': {
        'n_iter': N_ITER,
        'cv_folds': 5,
        'scoring': SCORING,
        'random_state': RANDOM_STATE
    }
}

with open(OUTPUT_PATH, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*70}")
print("OTIMIZAÇÃO CONCLUÍDA")
print(f"{'='*70}")
print(f"\nResultados salvos em: {OUTPUT_PATH}")
print(f"\nResumo dos melhores scores (F1 CV):")
for name, score in sorted(best_scores.items(), key=lambda x: -x[1]):
    print(f"  {name:15s}: {score:.4f}")
