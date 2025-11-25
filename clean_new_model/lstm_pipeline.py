import os
import random
from typing import Dict, List, Tuple, Any, Optional
from copy import deepcopy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0, # tenemos una unica capa de LSTM, por lo que no aplicamos dropout
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        pred = self.fc(last)
        return pred

# Creamos las ventanas de tiempo para el dataset
def _make_windows(array2d: np.ndarray, seq: int, target_col_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(array2d) - seq):
        win = array2d[i:i + seq]
        X.append(win[:-1])
        y.append([win[-1, target_col_idx]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def run_experiment(
    data_dir: str,
    features: List[str],
    features_to_scale: List[str],
    tickers: List[str],
    test_tickers: Optional[List[str]] = None,
    seq: int = 50,
    split: float = 0.85,
    batch_size: int = 32,
    epochs: int = 20,
    hidden_dim: int = 64,
    num_layers: int = 1,
    dropout: float = 0.0,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
    # Early stopping config
    val_frac: float = 0.1,           # fracción de ventanas de train reservadas para validación (temporal, al final). solo si hay early stopping
    early_stopping: bool = True,
    patience: int = 5,
    min_delta: float = 0.0,
) -> Dict[str, Any]:
    """
    Train and evaluate an LSTM to predict next-day Adj close (original scale) using configurable features.
    - Scaling per ticker fitted on train only (MinMax) to avoid leakage, applied ONLY to `features_to_scale`.
    - Windows of length `seq`; X: first seq-1 rows; y: Adj close at step seq (scaled target col).
    - If `test_tickers` is provided, evaluation is done only on that subset (must be subset of `tickers`).
    - Returns original-scale metrics per ticker, and weighted averages (when we want to give more weight to the tickers with more data).
    - NRMSE se calcula normalizando por la desviación estándar de y_true.
    """

    # Validations
    if 'Adj Close' not in features:
        raise ValueError("'Adj Close' must be included in features because it is the prediction target.")
    target_idx = features.index('Adj Close')
    if features_to_scale is None:
        raise ValueError("`features_to_scale` is required and must be a (possibly empty) list of feature names to scale.")
    unknown = [c for c in features_to_scale if c not in features]
    if unknown:
        raise ValueError(f"`features_to_scale` contains columns not in `features`: {unknown}")
    # Map scaling names to indices
    scale_indices = [features.index(c) for c in features_to_scale]
    scaled_set = set(scale_indices)

    # Prepare train/test tickers
    train_tickers = list(tickers)
    eval_tickers = list(test_tickers) if test_tickers is not None else list(tickers)
    missing_eval = [t for t in eval_tickers if t not in train_tickers]
    if missing_eval:
        raise ValueError(f"All test_tickers must be included in train tickers. Missing in train: {missing_eval}")

    # Seeding for reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device_t = torch.device(device)

    # Escaleado min-max por ticker (solo columnas seleccionadas)
    splits: Dict[str, Dict[str, Any]] = {}
    for t in train_tickers:
        path = os.path.join(data_dir, t)
        # Si no trae extensión, intentamos con .csv
        if not os.path.splitext(path)[1]:
            if os.path.exists(path + ".csv"):
                path = path + ".csv"
        df = pd.read_csv(path)
        if not all(col in df.columns for col in features):
            missing = [c for c in features if c not in df.columns]
            raise ValueError(f"{t} missing columns: {missing}")
        n = len(df)
        i_split = int(n * split)
        train_df = df.iloc[:i_split].reset_index(drop=True)
        test_df = df.iloc[i_split:].reset_index(drop=True)

        Xtr = train_df[features].astype('float32').values
        Xte = test_df[features].astype('float32').values

        # Guardar fechas del split de test si existen (para alinear gráfico luego)
        test_dates = None
        if 'Date' in df.columns:
            try:
                test_dates = test_df['Date'].tolist()
            except Exception:
                test_dates = None

        # ColumnTransformer: escala solo columnas seleccionadas, deja el resto igual (passthrough)
        ct = ColumnTransformer(
            transformers=[('scale', MinMaxScaler(), scale_indices)], # solo escala las columnas seleccionadas
            remainder='passthrough',
            sparse_threshold=0
        )
        Xtr_scaled = ct.fit_transform(Xtr).astype('float32')
        Xte_scaled = ct.transform(Xte).astype('float32')

        # Guardar scaler interno y posición del target dentro de las columnas escaladas
        scaler = ct.named_transformers_['scale'] if len(scale_indices) > 0 else None
        if target_idx in scale_indices:
            target_scale_pos = scale_indices.index(target_idx)
            target_scaled = True
        else:
            target_scale_pos = None
            target_scaled = False
        splits[t] = {
            'train_scaled': Xtr_scaled,
            'test_scaled': Xte_scaled,
            'ct': ct,
            'scaler': scaler,
            'target_scale_pos': target_scale_pos,
            'target_scaled': target_scaled,
            'test_dates': test_dates,
        }

    # Windows per ticker y split train/val por ticker (temporal: al final)
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    for t in train_tickers:
        Xt, yt = _make_windows(splits[t]['train_scaled'], seq, target_idx)
        if len(yt) == 0:
            continue
        if val_frac > 0.0:
            n = len(yt)
            n_val = int(n * val_frac)
            if n_val > 0 and n - n_val > 0:
                X_train_list.append(Xt[: n - n_val])
                y_train_list.append(yt[: n - n_val])
                X_val_list.append(Xt[n - n_val :])
                y_val_list.append(yt[n - n_val :])
            else:
                # si la serie es muy corta, usar todo como train
                X_train_list.append(Xt)
                y_train_list.append(yt)
        else:
            X_train_list.append(Xt)
            y_train_list.append(yt)
    if not X_train_list:
        raise ValueError("No training windows available; check data length and seq.")
    X_train = np.concatenate(X_train_list)
    y_train = np.concatenate(y_train_list)
    if X_val_list:
        X_val = np.concatenate(X_val_list)
        y_val = np.concatenate(y_val_list)
    else:
        X_val = np.zeros((0, seq - 1, len(features)), dtype=np.float32)
        y_val = np.zeros((0, 1), dtype=np.float32)

    # Model and training (igual que vimos en clase)
    model = LSTMPredictor(input_dim=len(features), hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device_t)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32, device=device_t),
        torch.tensor(y_train, dtype=torch.float32, device=device_t),
    )
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = None
    if len(y_val) > 0:
        ds_val = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )
        val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    # Entrenamiento con early stopping 
    best_state: Optional[Dict[str, Any]] = None
    best_val: float = float("inf")
    epochs_no_improve = 0
    best_epoch: int = -1
    stopped_epoch: Optional[int] = None

    def evaluate_loss(loader: DataLoader) -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device_t)
                yb = yb.to(device_t)
                out = model(xb)
                l = loss_fn(out, yb).item()
                losses.append(l * len(xb))
        return float(np.sum(losses) / max(1, len(loader.dataset)))

    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        if early_stopping and val_loader is not None:
            val_loss = evaluate_loss(val_loader)
            # mejora si desciende al menos min_delta
            if val_loss < best_val - min_delta:
                best_val = val_loss
                best_state = deepcopy(model.state_dict())
                epochs_no_improve = 0
                best_epoch = ep + 1
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    stopped_epoch = ep + 1
                    break
        # si no hay val, no hacemos early stop
    if stopped_epoch is None:
        stopped_epoch = epochs

    # Build test windows concatenated
    X_test_list, y_test_list = [], []
    per_ticker_counts: Dict[str, int] = {}
    for t in eval_tickers:
        Xt, yt = _make_windows(splits[t]['test_scaled'], seq, target_idx)
        per_ticker_counts[t] = int(len(yt))
        if len(yt) == 0:
            continue
        X_test_list.append(Xt)
        y_test_list.append(yt)
    if not X_test_list:
        raise ValueError("No test windows available; check data length and seq.")
    X_test = np.concatenate(X_test_list)
    y_test = np.concatenate(y_test_list)

    # Forecasting en test set
    ds_test = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )
    test_dl = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    model.eval()
    # Restaurar mejores pesos (si existen)
    if early_stopping and val_loader is not None and best_state is not None:
        model.load_state_dict(best_state)
    preds_all, trues_all = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device_t)
            out = model(xb).cpu().numpy().reshape(-1)
            preds_all.append(out)
            trues_all.append(yb.numpy().reshape(-1))
    y_pred_dl = np.concatenate(preds_all)
    y_true_dl = np.concatenate(trues_all)

    # Original-scale metrics per ticker and weighted/global
    per_ticker_metrics: Dict[str, Dict[str, float]] = {}
    per_ticker_abs_errors: Dict[str, List[float]] = {}
    per_ticker_series: Dict[str, Dict[str, List[float]]] = {}
    mae_list: List[float] = []
    mse_list: List[float] = []
    nrmse_list: List[float] = []
    r2_list: List[float] = []
    weights: List[int] = []
    wmape_num_list: List[float] = []
    wmape_den_list: List[float] = []
    mape_list: List[float] = []
    hit_num_list: List[int] = []
    hit_den_list: List[int] = []

    off = 0
    for t in eval_tickers:
        m = per_ticker_counts.get(t, 0)
        if m <= 0:
            continue
        y_true_norm = y_true_dl[off:off + m]
        y_pred_norm = y_pred_dl[off:off + m]
        off += m

        target_scaled = splits[t]['target_scaled']
        if target_scaled:
            scaler = splits[t]['scaler']
            j = splits[t]['target_scale_pos']
            inv_true = (y_true_norm - scaler.min_[j]) / scaler.scale_[j]
            inv_pred = (y_pred_norm - scaler.min_[j]) / scaler.scale_[j]
        else:
            inv_true = y_true_norm
            inv_pred = y_pred_norm

        # Guardar series para graficar (escala original) y fechas alineadas si están disponibles
        key = os.path.splitext(t)[0]
        aligned_dates: Optional[List[Any]] = None
        if splits[t].get('test_dates') is not None:
            all_dates = splits[t]['test_dates']
            try:
                # m = número de ventanas de test
                # y_true/pred corresponden a índices [seq-1, n_test-2]
                n_test = len(all_dates)
                start_idx = max(0, seq - 1)
                end_idx_exclusive = max(start_idx, n_test - 1)  # -1 para excluir el último, igual a [: -1]
                aligned_dates = all_dates[start_idx:end_idx_exclusive]
                # Ajustar por si difiere en 1 por borde
                if len(aligned_dates) != len(inv_true):
                    # fallback: truncar a longitud mínima común
                    k = min(len(aligned_dates), len(inv_true))
                    aligned_dates = aligned_dates[:k]
                    inv_true = inv_true[:k]
                    inv_pred = inv_pred[:k]
                    m = k
            except Exception:
                aligned_dates = None

        per_ticker_series[key] = {
            'dates': aligned_dates,
            'y_true': list(map(float, inv_true)),
            'y_pred': list(map(float, inv_pred)),
        }

        # Serie de errores absolutos por muestra (escala original)
        abs_errs = np.abs(inv_true - inv_pred)
        per_ticker_abs_errors[key] = abs_errs.tolist()

        mae_t = float(mean_absolute_error(inv_true, inv_pred))
        mse_t = float(mean_squared_error(inv_true, inv_pred))
        rmse_t = float(np.sqrt(mean_squared_error(inv_true, inv_pred)))
        std_y = float(np.std(inv_true))
        denom = std_y if std_y > 1e-12 else 1.0
        nrmse_t = float(rmse_t / denom)
        r2_t = float(r2_score(inv_true, inv_pred))
        mape_t = float(mean_absolute_percentage_error(inv_true, inv_pred))
        wmape_den = float(np.sum(np.abs(inv_true)))
        wmape_num = float(np.sum(np.abs(inv_true - inv_pred)))
        wmape_t = float(wmape_num / wmape_den) if wmape_den > 0 else float('nan')
        # Hit rate por ticker (dirección del cambio de precio)
        # Comparar signo de (true_t - true_{t-1}) vs (pred_t - true_{t-1})
        if m > 1:
            true_prev = inv_true[:-1]
            true_curr = inv_true[1:]
            pred_curr = inv_pred[1:]
            true_sign = np.sign(true_curr - true_prev)
            pred_sign = np.sign(pred_curr - true_prev)
            hits = (pred_sign == true_sign).astype(np.int32)
            hit_rate_t = float(np.mean(hits))
            hit_num = int(np.sum(hits))
            hit_den = int(len(hits))
        else:
            hit_rate_t = float('nan')
            hit_num = 0
            hit_den = 0
        per_ticker_metrics[os.path.splitext(t)[0]] = {
            'MAE': mae_t,
            'MSE': mse_t,
            'NRMSE': nrmse_t,
            'R2': r2_t,
            'MAPE': mape_t,
            'WMAPE': wmape_t,
            'HIT_RATE': hit_rate_t,
            'n': int(m),
        }
        mae_list.append(mae_t)
        mse_list.append(mse_t)
        nrmse_list.append(nrmse_t)
        r2_list.append(r2_t)
        weights.append(int(m))
        wmape_num_list.append(wmape_num)
        wmape_den_list.append(wmape_den)
        mape_list.append(mape_t)
        hit_num_list.append(hit_num)
        hit_den_list.append(hit_den)

    # Global concatenated original-scale metrics
    # Concatenamos los arrays de true y pred de todos los tickers en test set
    all_true_orig, all_pred_orig = [], []
    off = 0
    for t in eval_tickers:
        m = per_ticker_counts.get(t, 0)
        if m <= 0:
            continue
        y_true_norm = y_true_dl[off:off + m]
        y_pred_norm = y_pred_dl[off:off + m]
        off += m
        target_scaled = splits[t]['target_scaled']
        if target_scaled:
            scaler = splits[t]['scaler']
            j = splits[t]['target_scale_pos']
            inv_true = (y_true_norm - scaler.min_[j]) / scaler.scale_[j]
            inv_pred = (y_pred_norm - scaler.min_[j]) / scaler.scale_[j]
        else:
            inv_true = y_true_norm
            inv_pred = y_pred_norm
        all_true_orig.append(inv_true)
        all_pred_orig.append(inv_pred)

    y_true_orig_global = np.concatenate(all_true_orig)
    y_pred_orig_global = np.concatenate(all_pred_orig)
    global_metrics = {
        'MAE': float(mean_absolute_error(y_true_orig_global, y_pred_orig_global)),
        'MSE': float(mean_squared_error(y_true_orig_global, y_pred_orig_global)),
        'NRMSE': float(np.sqrt(mean_squared_error(y_true_orig_global, y_pred_orig_global)) / max(1e-12, float(np.std(y_true_orig_global)))),
        'R2': float(r2_score(y_true_orig_global, y_pred_orig_global)),
        'MAPE': float(mean_absolute_percentage_error(y_true_orig_global, y_pred_orig_global)),
        'WMAPE': float(np.sum(np.abs(y_true_orig_global - y_pred_orig_global)) / max(1e-12, np.sum(np.abs(y_true_orig_global)))),
        # hit-rate global calculado como suma de hits por ticker / suma de denominadores por ticker
        'HIT_RATE': float(np.sum(hit_num_list) / max(1, np.sum(hit_den_list))) if hit_den_list else None,
        'total_n': int(len(y_true_orig_global)),
    }
# metrica ponderada por el numero de muestras de cada ticker
    weighted_metrics = {
        'MAE_weighted': float(np.average(mae_list, weights=weights)) if weights else None,
        'MSE_weighted': float(np.average(mse_list, weights=weights)) if weights else None,
        'NRMSE_weighted': float(np.average(nrmse_list, weights=weights)) if weights else None,
        'R2_weighted': float(np.average(r2_list, weights=weights)) if weights else None,
        'MAPE_weighted': float(np.average(mape_list, weights=weights)) if weights else None,
        # WMAPE ponderado por |y_true| (equivale a WMAPE global si conjuntos no se solapan)
        'WMAPE_weighted': float(np.sum(wmape_num_list) / max(1e-12, np.sum(wmape_den_list))) if wmape_den_list else None,
        # HIT ponderado por número de comparaciones válidas por ticker
        'HIT_RATE_weighted': float(np.sum(hit_num_list) / max(1, np.sum(hit_den_list))) if hit_den_list else None,
        'total_n': int(sum(weights)) if weights else 0,
    }

    return {
        'features': features,
        'train_tickers': train_tickers,
        'test_tickers': eval_tickers,
        'global_metrics': global_metrics,
        'weighted_metrics': weighted_metrics,
        'per_ticker': per_ticker_metrics,
        'per_ticker_abs_errors': per_ticker_abs_errors,
        'per_ticker_series': per_ticker_series,  # series originales para graficar
        'early_stopping': { # lo usamos para monitorizar el loss de validación en las exploraciones intermedias
            'enabled': bool(early_stopping and val_loader is not None),
            'best_val_loss': None if best_val == float('inf') else best_val,
            'patience': patience,
            'min_delta': min_delta,
            'best_epoch': None if best_epoch == -1 else best_epoch,
            'stopped_epoch': stopped_epoch,
            'early_stopped': bool((early_stopping and val_loader is not None) and (stopped_epoch is not None and stopped_epoch < epochs)),
        },
    }


