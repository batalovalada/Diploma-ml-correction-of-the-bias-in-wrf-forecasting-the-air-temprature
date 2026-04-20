import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

#================== paths ===============================
path_processed = '../../data/ConvLSTM/processed/'
path_norm = '../../data/ConvLSTM/norm_params/'
path_results = '../../reports/results/ConvLSTM/'

# ============= parameters =============================
BATCH_SIZE = 8
INPUT_DIM = 11 # selected features count
HIDDEN_DIM = 32
KERNEL_SIZE = 3
EPOCHS = 20

device = torch.device('cpu')
lookback = 4
horizon = 1

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
        torch.tensor(mask, dtype=torch.float32)
    )

# loss function =====================================
def masked_mse(pred, target, mask):
    return (((pred - target)**2)*mask).sum()/mask.sum()

# metrics ============================================
def masked_rmse(pred, target, mask):
    return np.sqrt(np.sum(((pred - target) ** 2) * mask) / np.sum(mask))

def masked_mae(pred, target, mask):
    return np.sum(np.abs(pred - target) * mask) / np.sum(mask)

def masked_bias(pred, target, mask):
    return np.sum((pred - target) * mask) / np.sum(mask)

def masked_corr(pred, target, mask):
    pred = pred[mask > 0]
    target = target[mask > 0]
    return np.corrcoef(pred, target)[0, 1]

# ======================================================
class WeatherDataset(Dataset):
    def __init__(self, X, y, mask):
        self.X = X
        self.y = y
        self.mask = mask

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx]

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()

        # padding to save map's size 5x5 + 1 -> 7x7 -> 3x3
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            input_dim+hidden_dim,
            4*hidden_dim,
            kernel_size,
            padding=padding,
        )
        self.hidden_dim = hidden_dim

    def forward(self, x, h, c):
        # x shape = (batch, channels, H, W)
        # h, c shape = (batch, hidden_dim, H, W)
        combined = torch.cat([x, h], dim=1)
        conv_out = self.conv(combined)

        i, f, o, g = torch.chunk(conv_out, 4, dim=1)
        i = torch.sigmoid(i) #input
        f = torch.sigmoid(f) #forget
        o = torch.sigmoid(o) #output
        g = torch.tanh(g) #candidate

        c_next = f*c + i*g
        h_next = o*torch.tanh(c_next)

        return h_next, c_next

# ConvLSTM model
class ConvLSTM(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()

        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size=KERNEL_SIZE)
        self.output = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x):
        # x shape = (batch, time, channels, H, W)
        batch, seq_len, C, H, W = x.shape
        h = torch.zeros(batch, HIDDEN_DIM, H, W, device=x.device)
        c = torch.zeros(batch, HIDDEN_DIM, H, W, device=x.device)

        for t in range(seq_len):
            h, c = self.cell(x[:, t], h, c)

        out = self.output(h)
        return out.squeeze(1) # shape (batch, H, W)


#================= download data =======================
X_train, y_train, mask_train = load_npz(path_processed + 'train.npz')
X_val, y_val, mask_val = load_npz(path_processed + 'val.npz')
X_test, y_test, mask_test = load_npz(path_processed + 'test.npz')

T2_wrf_test = xr.open_dataset(path_processed+'t2_wrf_test.nc')
T2_era5_test = xr.open_dataset(path_processed+'t2_era5_test.nc')

# ================= to tensors =========================
X_train, y_train, mask_train = tensor_data(X_train, y_train, mask_train)
X_val, y_val, mask_val = tensor_data(X_val, y_val, mask_val)
X_test, y_test, mask_test  = tensor_data(X_test, y_test, mask_test )

# ============= create Datasets, DataLoaders ================
Train_Dataset = WeatherDataset(X_train, y_train, mask_train)
Val_Dataset = WeatherDataset(X_val, y_val, mask_val)
Test_Dataset = WeatherDataset(X_test, y_test, mask_test)

Train_Loader = DataLoader(Train_Dataset, batch_size=BATCH_SIZE, shuffle=True)
Val_Loader   = DataLoader(Val_Dataset, batch_size=BATCH_SIZE, shuffle=False)
Test_Loader   = DataLoader(Test_Dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===================== ConvLSTM model ======================
model = ConvLSTM(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# =================== Training ==============================
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
    print(f'Epoch {epoch+1} loss: {train_loss}')

    # Validation
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for X_batch, y_batch, mask_batch in Val_Loader:
            pred = model(X_batch)
            loss = masked_mse(pred, y_batch, mask_batch)
            val_loss += loss.item()

    val_loss /= len(Val_Loader)
    val_losses.append(val_loss)

    print(f'Validation loss: {val_loss}')

# ===================== Test =======================
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
offset = lookback + horizon - 1
T2_wrf_ds= T2_wrf_test.isel(time=slice(offset, None)).fillna(0)
T2_era5_ds = T2_era5_test.isel(time=slice(offset, None)).fillna(0)

T2_wrf = T2_wrf_ds['T2'].values
T2_era5 =T2_era5_ds['t2m'].values

T2_corrected = T2_wrf + y_pred

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

# rmse metric to compare WRF and ConvLSTM ConvLSTM
rmse_wrf = masked_rmse(T2_wrf, T2_era5, y_mask)
rmse_corr = masked_rmse(T2_corrected, T2_era5, y_mask)

print(f"WRF RMSE: {rmse_wrf}")
print(f"Corrected RMSE: {rmse_corr}")

# vizualization ========================================
# loss plot--------------------------------
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Validation")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training curve")
plt.savefig(path_results+"loss_curve.png")
plt.close()

# scatter plot ----------------------------
mask_flat = y_mask.flatten().astype(bool)
lims = [
    min(y_true.min(), y_pred.min()),
    max(y_true.max(), y_pred.max())
]

plt.figure(figsize=(10, 6))
plt.scatter(y_true.flatten()[mask_flat], y_pred.flatten()[mask_flat], alpha=0.3)
plt.plot(lims, lims, 'r--')
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title("Prediction vs Truth")
plt.savefig(path_results+"compare.png")
plt.close()

# Spatial RMSE map -------------------------
diff = (y_pred - y_true) ** 2
numerator = np.sum(diff * y_mask, axis=0)
denominator = np.sum(y_mask, axis=0)
denominator = np.where(denominator == 0, np.nan, denominator)

rmse_map = np.sqrt(numerator / denominator)
plt.figure(figsize=(10, 6))
plt.imshow(rmse_map, cmap="viridis")
plt.colorbar()
plt.title("RMSE map")
plt.savefig(path_results+"rmse_map.png")
plt.close()

# ConvLSTM visualization ----------------------
time_i = 0
vmin = min(T2_wrf.min(), T2_corrected.min(), T2_era5.min())
vmax = max(T2_wrf.max(), T2_corrected.max(), T2_era5.max())
plt.figure(figsize=(12, 4))

# WRF
plt.subplot(1, 3, 1)
plt.title("WRF")
plt.imshow(T2_wrf[time_i], cmap="coolwarm", vmin=vmin, vmax=vmax)
plt.colorbar(fraction=0.046, pad=0.04)

# Corrected
plt.subplot(1, 3, 2)
plt.title("ConvLSTM corrected")
plt.imshow(T2_corrected[time_i], cmap="coolwarm", vmin=vmin, vmax=vmax)
plt.colorbar(fraction=0.046, pad=0.04)

# ERA5
plt.subplot(1, 3, 3)
plt.title("ERA5 (truth)")
plt.imshow(T2_era5[time_i], cmap="coolwarm", vmin=vmin, vmax=vmax)
plt.colorbar(fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(path_results+"results_map.png")
plt.close()

# save ConvLSTM =======================================
np.savez(path_results+"ConvLSTM.npz",
         y_pred=y_pred,
         y_true=y_true,
         mask=y_mask)

# save model
torch.save(model.state_dict(), path_results+"model.pth")

# save metrics
with open(path_results+"metrics.txt", "w") as f:
    f.write(f"RMSE: {rmse}\n")
    f.write(f"MAE: {mae}\n")
    f.write(f"Bias: {bias}\n")
    f.write(f"Corr: {corr}\n")
