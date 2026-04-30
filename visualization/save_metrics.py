def save_metrics(path_dir, rmse, mae, bias, corr, rmse_wrf, rmse_corr):
    with open(path_dir + "metrics.txt", "w") as f:
        f.write(f"RMSE: {rmse}\n")
        f.write(f"MAE: {mae}\n")
        f.write(f"Bias: {bias}\n")
        f.write(f"Corr: {corr}\n\n")
        f.write(f"Compare WRF and Corrected\n")
        f.write(f"WRF RMSE: {rmse_wrf}\n")
        f.write(f"Corrected RMSE: {rmse_corr}\n")