from torch.utils.data import DataLoader
import optuna
import copy

from model import *
from config.models_params.convlstm import EPOCHS
from metrics.metrics import *
from visualization.save_metrics import save_metrics
from visualization.save_plots import *

#================== paths ===============================
path_processed = '../../data/ConvLSTM/base/processed/'
path_norm = '../../data/ConvLSTM/base/norm_params/'
path_results = '../../reports/results/ConvLSTM/base/'

# ============= parameters =============================
device = torch.device('cpu')

# ======================================================
# load npz data function
def load_npz(path):
    data = np.load(path)
    return data['X'], data['y'], data['mask']

def load_norm_params(path):
    data = np.load(path, allow_pickle=True).item()
    return float(data['mean']), float(data['std'])

# X, y, y_mask to tensor function
def tensor_data(X, y, mask):
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.bool)
    )

# loss function =====================================
def masked_mse(pred, target, mask):
    return torch.mean((pred[mask] - target[mask])**2)

def train_model(params, Train_Dataset, Val_Dataset):
    Train_Loader = DataLoader(Train_Dataset, batch_size=params['batch'], shuffle=True)
    Val_Loader = DataLoader(Val_Dataset, batch_size=params['batch'], shuffle=False)

    model = ConvLSTM(hidden_dim=params['hidden_dim']).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    # =================== Training ==============================
    best_val = float('inf')
    counter = 0
    patience = 5
    best_model_state = None

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for X_batch, y_batch, mask_batch in Train_Loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = masked_mse(pred, y_batch, mask_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(Train_Loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for X_batch, y_batch, mask_batch in Val_Loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                mask_batch = mask_batch.to(device)
                pred = model(X_batch)
                loss = masked_mse(pred, y_batch, mask_batch)
                val_loss += loss.item()

        val_loss /= len(Val_Loader)
        val_losses.append(val_loss)

        # early stopping
        if val_loss < best_val:
            best_val = val_loss
            counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping")
            break

    model.load_state_dict(best_model_state)
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }


# optuna function ====================================
def objective(trial):
    params={
        'batch': trial.suggest_categorical('batch', [16, 24]),
        'lr': trial.suggest_float('lr', 1e-5, 3e-3, log=True),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [16, 32, 48, 64]),
    }

    result = train_model(params, Train_Dataset, Val_Dataset)
    val_loss = min(result['val_losses'])
    return val_loss

#================= load data =======================
X_train, y_train, mask_train = load_npz(path_processed + 'train.npz')
X_val, y_val, mask_val = load_npz(path_processed + 'val.npz')
X_test, y_test, mask_test = load_npz(path_processed + 'test.npz')

T2_wrf_test = np.load(path_processed+'t2_wrf_test.npy')
T2_era5_test = np.load(path_processed+'t2_era5_test.npy')

# ================= to tensors =========================
X_train, y_train, mask_train = tensor_data(X_train, y_train, mask_train)
X_val, y_val, mask_val = tensor_data(X_val, y_val, mask_val)
X_test, y_test, mask_test  = tensor_data(X_test, y_test, mask_test)

# ============= create Datasets, DataLoaders ================
Train_Dataset = WeatherDataset(X_train, y_train, mask_train)
Val_Dataset = WeatherDataset(X_val, y_val, mask_val)
Test_Dataset = WeatherDataset(X_test, y_test, mask_test)

# ================ select model's params with Optuna ====================================
study = optuna.create_study(direction="minimize")

study.optimize(objective, n_trials=20)
best_params = study.best_params

print("Best params:", best_params)
print("Best value:", study.best_value)

# ===================== training ConvLSTM model ======================
final_train = train_model(
    best_params,
    Train_Dataset,
    Val_Dataset
)
model = final_train['model']
train_losses = final_train['train_losses']
val_losses = final_train['val_losses']

# ===================== Test =======================
Test_Loader = DataLoader(Test_Dataset, batch_size=best_params['batch'], shuffle=False) #with latlon batch size

model.eval()

all_preds = []
all_targets = []
all_masks = []

with torch.no_grad():
    for X_batch, y_batch, mask_batch in Test_Loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        mask_batch = mask_batch.to(device)

        pred = model(X_batch)

        all_preds.append(pred.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())
        all_masks.append(mask_batch.cpu().numpy())

y_pred = np.concatenate(all_preds, axis=0)
y_true = np.concatenate(all_targets, axis=0)
y_mask = np.concatenate(all_masks, axis=0)

# =========== remove normalize =======================
y_mean, y_std = load_norm_params(path_norm+'norm_params_y.npy')

y_pred = y_pred * y_std + y_mean
y_true = y_true * y_std + y_mean

# ============ recover T2 ==============================
T2_corrected = T2_wrf_test + y_pred
# return nan by mask, because model interp values in nodes, where era5 = nan
T2_corrected[~y_mask] = np.nan

# ============= Metrics ==========================
rmse = masked_rmse(y_pred, y_true, y_mask)
mae = masked_mae(y_pred, y_true, y_mask)
bias = masked_bias(y_pred, y_true, y_mask)
corr = masked_corr(y_pred, y_true, y_mask)

print("\n\nTest metrics:")
print(f"RMSE: {rmse}")
print(f"MAE:  {mae}")
print(f"Bias: {bias}")
print(f"Corr: {corr}")

# rmse metric to compare WRF and ConvLSTM
rmse_wrf = masked_rmse(T2_wrf_test, T2_era5_test, y_mask)
rmse_corr = masked_rmse(T2_corrected, T2_era5_test, y_mask)

print(f"WRF RMSE: {rmse_wrf}")
print(f"Corrected RMSE: {rmse_corr}")

# vizualization ========================================
# loss plot
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Validation")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training curve (ConvLSTM)")
plt.savefig(path_results+"loss_curve.png")
plt.close()

# scatter plot
save_scatter_plot(y_pred, y_true, y_mask, path_results, 'ConvLSTM')

# Spatial RMSE map
save_rmse_map(y_pred, y_true, y_mask, path_results, 'ConvLSTM')

# ConvLSTM visualization
save_corrected_map(T2_wrf_test, T2_era5_test, T2_corrected, path_results, 'ConvLSTM')

# save ConvLSTM =======================================
# save model
torch.save(model.state_dict(), path_results+"model.pth")

# save results
np.savez(path_results+"ConvLSTM.npz",
         y_pred=y_pred,
         y_true=y_true,
         mask=y_mask)

# save metrics
save_metrics(path_results, rmse, mae, bias, corr, rmse_wrf, rmse_corr)

# save train params
with open(path_results + "model_params.txt", "w") as f:
    f.write("Optimized params by Optuna:\n")
    f.write(f"batch size: {best_params['batch']}\n")
    f.write(f"hidden dim: {best_params['hidden_dim']}\n")
    f.write(f"lr: {best_params['lr']}\n\n")
    f.write("Fixed params:\n")
    f.write(f"epochs: {EPOCHS}\n")
    f.write(f"input dim: {INPUT_DIM}\n")
    f.write(f"kernel size: {KERNEL_SIZE}\n")