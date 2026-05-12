from xgboost import XGBRegressor
from config.hyperparameters.tree_models import N_JOBS, RANDOM_STATE

def return_params(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 150, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate',  0.01, 0.12, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 6),
        'gamma': trial.suggest_float('gamma', 0.0, 1.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 2.0),
    }

def build_model(params):
    return XGBRegressor(
        **params,
        objective='reg:squarederror',
        eval_metric='rmse',
        early_stopping_rounds=20,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        tree_method='hist'
    )

def train_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

def save_best_params(params, path_dir):
    with open(path_dir + "model_params.txt", "w") as f:
        f.write("Optimized params by Optuna:\n")
        f.write(f"n_estimators: {params['n_estimators']}\n")
        f.write(f"max depth: {params['max_depth']}\n")
        f.write(f"learning_rate: {params['learning_rate']}\n")
        f.write(f"subsample: {params['subsample']}\n")
        f.write(f"colsample_bytree: {params['colsample_bytree']}\n")
        f.write(f"min_child_weight: {params['min_child_weight']}\n")
        f.write(f"gamma: {params['gamma']}\n")
        f.write(f"reg_alpha: {params['reg_alpha']}\n")
        f.write(f"reg_lambda: {params['reg_lambda']}\n\n")
        f.write("Fixed params:\n")
        f.write(f"n jobs: {N_JOBS}\n")
        f.write(f"random state: {RANDOM_STATE}\n")