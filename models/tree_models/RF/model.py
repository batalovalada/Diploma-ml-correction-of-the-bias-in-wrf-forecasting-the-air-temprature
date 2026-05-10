from sklearn.ensemble import RandomForestRegressor
from config.hyperparameters.tree_models import N_JOBS, RANDOM_STATE

def return_params(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 35),
        'max_features': trial.suggest_categorical('max_features', ["sqrt", "log2", 0.5, 0.7]),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
    }

def build_model(params):
    return RandomForestRegressor(
        **params,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE
    )

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)

def save_best_params(params, path_dir):
    with open(path_dir + "model_params.txt", "w") as f:
        f.write("Optimized params by Optuna:\n")
        f.write(f"n_estimators: {params['n_estimators']}\n")
        f.write(f"max depth: {params['max_depth']}\n")
        f.write(f"max features: {params['max_features']}\n")
        f.write(f"min samples leaf: {params['min_samples_leaf']}\n")
        f.write(f"min_samples_split: {params['min_samples_split']}\n\n")
        f.write("Fixed params:\n")
        f.write(f"n jobs: {N_JOBS}\n")
        f.write(f"random state: {RANDOM_STATE}\n")