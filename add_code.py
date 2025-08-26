import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# -----------------------
# Parse command-line args
# -----------------------
parser = argparse.ArgumentParser(description='Disease prediction script for Dengue Fever (XGB / RF / LSTM)')
parser.add_argument('--input_file', type=str, default='Filtered_Disease_Data.csv',
                    help='Path to the input CSV file containing disease data')
parser.add_argument('--n_trials', type=int, default=50,
                    help='Number of trials for Optuna optimization (default: 50)')
parser.add_argument('--model', type=str, default='xgb', choices=['xgb', 'rf', 'lstm'],
                    help='Model to use: xgb (XGBoost), rf (RandomForest), lstm (Keras LSTM)')
parser.add_argument('--window_size', type=int, default=8,
                    help='Window size for LSTM sequences (default: 8)')
parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
args = parser.parse_args()

# -----------------------
# Housekeeping
# -----------------------
np.random.seed(args.seed)
optuna.logging.set_verbosity(optuna.logging.WARNING)

output_folder = 'disease_prediction_results_dengue'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

results_file = os.path.join(output_folder, 'prediction_results_dengue.txt')
predict_file = os.path.join(output_folder, 'predict_data_dengue.csv')
combined_file = os.path.join(output_folder, 'combined_data_dengue.csv')

df = pd.read_csv(args.input_file)

# list districts
all_district_cols = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q12',
                     'GV', 'TB', 'BT', 'PN', 'CC', 'HM', 'TD', 'BC', 'NB', 'CG', 'TP', 'BTA']

# Dictionary for predict (distr)
predictions_2024_all = {district: [] for district in all_district_cols}

# -------------
# Helper: LSTM
# -------------
def build_lstm_sequences(nam, tuan, values, window):
    """
    Build sliding-window sequences from ordered arrays: nam, tuan, values
    X shape: (n_samples, window, 1)
    y shape: (n_samples,)
    Also returns arrays year_target, week_target and year_week_index for plotting/splitting.
    """
    X, y = [], []
    year_target, week_target = [], []
    for i in range(window, len(values)):
        X.append(values[i-window:i])
        y.append(values[i])
        year_target.append(nam[i])
        week_target.append(tuan[i])
    X = np.array(X).reshape(-1, window, 1)
    y = np.array(y)
    year_target = np.array(year_target)
    week_target = np.array(week_target)
    year_week_idx = (year_target - 2022) * 52 + week_target
    return X, y, year_target, week_target, year_week_idx

def train_lstm_with_optuna(X_train, y_train, n_trials=20, seed=42):
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models, callbacks, optimizers
        tf.keras.utils.set_random_seed(seed)
    except Exception as e:
        raise ImportError(
            "TensorFlow is required for LSTM. Please install with `pip install tensorflow` "
            "or `pip install tensorflow-cpu`.\nOriginal error: " + str(e)
        )

    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial):
        units = trial.suggest_int('units', 16, 128, step=16)
        dropout = trial.suggest_float('dropout', 0.0, 0.4)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        epochs = trial.suggest_int('epochs', 40, 150)

        fold_mse = []
        for tr_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            model = models.Sequential([
                layers.Input(shape=(X_tr.shape[1], X_tr.shape[2])),
                layers.LSTM(units, return_sequences=False),
                layers.Dropout(dropout),
                layers.Dense(1)
            ])
            opt = optimizers.Adam(learning_rate=lr)
            model.compile(optimizer=opt, loss='mse')

            es = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
            model.fit(X_tr, y_tr,
                      validation_data=(X_val, y_val),
                      epochs=epochs,
                      batch_size=batch_size,
                      verbose=0,
                      callbacks=[es])
            y_pred = model.predict(X_val, verbose=0).reshape(-1)
            fold_mse.append(mean_squared_error(y_val, y_pred))

            # Free resources between folds
            tf.keras.backend.clear_session()

        return float(np.mean(fold_mse))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params

    # Train final model
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks, optimizers
    tf.keras.utils.set_random_seed(seed)

    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
        layers.LSTM(best['units'], return_sequences=False),
        layers.Dropout(best['dropout']),
        layers.Dense(1)
    ])
    opt = optimizers.Adam(learning_rate=best['lr'])
    model.compile(optimizer=opt, loss='mse')
    es = callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=0)
    model.fit(X_train, y_train,
              validation_split=0.1,
              epochs=best['epochs'],
              batch_size=best['batch_size'],
              verbose=0,
              callbacks=[es])
    return model, best

# --------------------------
# Core per-district pipeline
# --------------------------
def process_district(district):
    # Filter data riêng cho bệnh "Sốt xuất huyết Dengue"
    data = df[df['T_BTT'] == 'Sốt xuất huyết Dengue'][['NAM', 'TUAN', district]].copy()
    data = data.dropna().sort_values(['NAM', 'TUAN']).reset_index(drop=True)

    # Tính đặc trưng theo tuần (cho mô hình cây)
    data['sin_week'] = np.sin(2 * np.pi * data['TUAN'] / 52)
    data['cos_week'] = np.cos(2 * np.pi * data['TUAN'] / 52)

    # historical weekly avg dùng 2022-2023
    hist_avg_map = data[data['NAM'].isin([2022, 2023])].groupby('TUAN')[district].mean().to_dict()
    overall_hist_avg = np.mean(list(hist_avg_map.values())) if len(hist_avg_map) else 0.0
    data['historical_weekly_avg'] = data['TUAN'].map(hist_avg_map).fillna(overall_hist_avg)

    # Lags, rolling cho mô hình cây
    data[f'{district}_lag1'] = data[district].shift(1)
    data[f'{district}_lag2'] = data[district].shift(2)
    data[f'{district}_lag3'] = data[district].shift(3)
    data[f'{district}_rolling_mean_3'] = data[district].rolling(window=3).mean()
    data[f'{district}_rolling_std_3'] = data[district].rolling(window=3).std()
    data[f'{district}_rolling_max_3'] = data[district].rolling(window=3).max()
    data[f'{district}_diff'] = data[district] - data[f'{district}_lag1']

    # Dữ liệu cho train/test theo năm
    data_feat = data.dropna().copy()
    data_feat['Year_Week'] = (data_feat['NAM'] - 2022) * 52 + data_feat['TUAN']

    # Split theo năm: train < 2022, test in [2022, 2023]
    train_mask_tree = data_feat['NAM'] < 2022
    test_mask_tree = data_feat['NAM'].isin([2022, 2023])

    # -----------------------
    # Branch by model choice
    # -----------------------
    model_name = args.model.lower()

    if model_name in ['xgb', 'rf']:
        # Feature set cho mô hình cây
        FEATURES = ['TUAN', 'NAM', 'sin_week', 'cos_week', 'historical_weekly_avg',
                    f'{district}_lag1', f'{district}_lag2', f'{district}_lag3',
                    f'{district}_rolling_mean_3', f'{district}_rolling_std_3',
                    f'{district}_rolling_max_3', f'{district}_diff']

        train_data = data_feat[train_mask_tree]
        test_data = data_feat[test_mask_tree]

        if len(train_data) == 0 or len(test_data) == 0:
            raise ValueError("Không đủ dữ liệu train/test theo mốc năm đã định (train < 2022, test ∈ {2022,2023}).")

        X_train = train_data[FEATURES]
        y_train = train_data[district]
        X_test = test_data[FEATURES]
        y_test = test_data[district]

        if model_name == 'xgb':
            # ---- XGBoost + Optuna (giữ nguyên tinh thần code cũ) ----
            def objective(trial):
                param = {
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'random_state': args.seed,
                    'objective': 'reg:squarederror'
                }
                num_boost_round = trial.suggest_int('num_boost_round', 100, 500)

                tscv = TimeSeriesSplit(n_splits=5)
                mse_scores = []
                for train_idx, val_idx in tscv.split(X_train):
                    X_tr, X_val = X_train.iloc[train_idx].copy(), X_train.iloc[val_idx].copy()
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    dtrain = xgb.DMatrix(X_tr, label=y_tr)
                    dval = xgb.DMatrix(X_val, label=y_val)
                    model = xgb.train(param, dtrain, num_boost_round=num_boost_round,
                                      evals=[(dval, 'eval')], early_stopping_rounds=10, verbose_eval=False)
                    y_pred_val = model.predict(dval)
                    mse_scores.append(mean_squared_error(y_val, y_pred_val))
                return np.mean(mse_scores)

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=args.n_trials)
            best_params = study.best_params
            num_boost_round = best_params.pop('num_boost_round')

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            best_model = xgb.train(best_params, dtrain, num_boost_round=num_boost_round,
                                   evals=[(dtest, 'test')], early_stopping_rounds=10, verbose_eval=False)

            y_pred_test = best_model.predict(dtest)

        elif model_name == 'rf':
            # ---- Random Forest + Optuna ----
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 200, 1200, step=200),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'random_state': args.seed,
                    'n_jobs': -1
                }
                tscv = TimeSeriesSplit(n_splits=5)
                mse_scores = []
                for train_idx, val_idx in tscv.split(X_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    model = RandomForestRegressor(**params)
                    model.fit(X_tr, y_tr)
                    y_pred_val = model.predict(X_val)
                    mse_scores.append(mean_squared_error(y_val, y_pred_val))
                return float(np.mean(mse_scores))

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=args.n_trials)
            best_params = study.best_params

            best_model = RandomForestRegressor(random_state=args.seed, n_jobs=-1, **best_params)
            best_model.fit(X_train, y_train)
            y_pred_test = best_model.predict(X_test)

        # ------------- Metrics -------------
        y_test_array = np.array(y_test)
        y_pred_test_array = np.array(y_pred_test)
        absolute_errors = np.abs(y_test_array - y_pred_test_array)
        total_absolute_error = float(absolute_errors.sum())
        average_absolute_error = float(absolute_errors.mean())
        total_actual_cases = float(y_test_array.sum())
        total_error_ratio = (total_absolute_error / total_actual_cases) * 100 if total_actual_cases > 0 else 0.0
        mse = float(mean_squared_error(y_test, y_pred_test))

        # ------------- Forecast 2024 (feature-based) -------------
        last_2023 = data[data['NAM'] == 2023]
        if last_2023.empty:
            raise ValueError("Không có dữ liệu năm 2023 để khởi tạo dự báo 2024.")

        # Chuỗi giá trị thật 2023 (để làm seed cho lag/rolling)
        historical_values = list(data[data['NAM'] == 2023][district].values)
        predictions_2024 = []

        for week in range(1, 53):
            sin_week = np.sin(2 * np.pi * week / 52)
            cos_week = np.cos(2 * np.pi * week / 52)
            historical_weekly_avg = hist_avg_map.get(week, overall_hist_avg)

            if len(historical_values) >= 3:
                lag1, lag2, lag3 = historical_values[-1], historical_values[-2], historical_values[-3]
            elif len(historical_values) == 2:
                lag1, lag2, lag3 = historical_values[-1], historical_values[-2], historical_values[-2]
            else:
                lag1 = lag2 = lag3 = historical_values[-1]

            if len(historical_values) >= 3:
                rolling_mean_3 = float(np.mean(historical_values[-3:]))
                rolling_std_3 = float(np.std(historical_values[-3:]))
                rolling_max_3 = float(np.max(historical_values[-3:]))
            else:
                rolling_mean_3 = float(np.mean(historical_values))
                rolling_std_3 = float(np.std(historical_values)) if len(historical_values) > 1 else 0.0
                rolling_max_3 = float(np.max(historical_values))

            diff = lag1 - lag2 if len(historical_values) >= 2 else 0.0

            X_week = pd.DataFrame({
                'TUAN': [week], 'NAM': [2024], 'sin_week': [sin_week], 'cos_week': [cos_week],
                'historical_weekly_avg': [historical_weekly_avg],
                f'{district}_lag1': [lag1], f'{district}_lag2': [lag2], f'{district}_lag3': [lag3],
                f'{district}_rolling_mean_3': [rolling_mean_3], f'{district}_rolling_std_3': [rolling_std_3],
                f'{district}_rolling_max_3': [rolling_max_3], f'{district}_diff': [diff]
            })

            if model_name == 'xgb':
                d_week = xgb.DMatrix(X_week[FEATURES])
                pred = float(best_model.predict(d_week)[0])
            else:  # rf
                pred = float(best_model.predict(X_week[FEATURES])[0])

            pred = max(pred, 0.0)
            # floor nhẹ theo historical avg để tránh rớt quá thấp
            if historical_weekly_avg > 0 and pred < historical_weekly_avg * 0.5:
                pred = historical_weekly_avg * 0.75

            predictions_2024.append(pred)
            historical_values.append(pred)

        predictions_2024_all[district] = np.round(predictions_2024).astype(int)

        # ------------- Plot -------------
        plt.figure(figsize=(14, 7))
        plt.plot(test_data['Year_Week'], y_test_array, marker='o', linestyle='-', label='Actual 2022-2023')
        plt.plot(test_data['Year_Week'], y_pred_test_array, marker='x', linestyle='--', label=f'Predicted 2022-2023 ({model_name.upper()})')
        year_week_2024 = (2024 - 2022) * 52 + np.arange(1, 53)
        plt.plot(year_week_2024, predictions_2024_all[district], marker='o', linestyle='-', label='Predicted 2024')
        plt.axvline(x=52, color='gray', linestyle='--', label='End of 2022')
        plt.axvline(x=104, color='gray', linestyle='--', label='End of 2023')
        plt.title(f'Dự đoán và thực tế số ca Sốt xuất huyết Dengue tại {district} (Model: {model_name.upper()})')
        plt.xlabel('Thời gian (từ 2022)')
        plt.ylabel('Số ca bệnh')
        plt.legend()
        plt.grid(True)
        plt.xticks(ticks=[0, 26, 52, 78, 104, 130, 156],
                   labels=['2022-W1', '2022-W26', '2023-W1', '2023-W26', '2024-W1', '2024-W26', '2025-W1'])
        plot_path = os.path.join(output_folder, f'prediction_plot_{district}_dengue_{model_name}.png')
        plt.savefig(plot_path)
        plt.close()

    elif model_name == 'lstm':
        # ----- LSTM pipeline: dùng chuỗi giá trị (1 biến) với cửa sổ window_size -----
        window = args.window_size

        # Chuẩn bị chuỗi theo thời gian
        nam_arr = data['NAM'].values
        tuan_arr = data['TUAN'].values
        val_arr = data[district].values.astype('float32')

        if len(val_arr) < window + 60:
            raise ValueError(f"Chuỗi {district} quá ngắn cho LSTM (cần tối thiểu ~{window+60} quan sát).")

        X_all, y_all, year_tgt, week_tgt, year_week_idx = build_lstm_sequences(nam_arr, tuan_arr, val_arr, window)

        # Split theo năm của TARGET (điểm dự báo)
        train_mask = year_tgt < 2022
        test_mask = np.isin(year_tgt, [2022, 2023])

        if not train_mask.any() or not test_mask.any():
            raise ValueError("Không đủ dữ liệu train/test theo mốc năm đã định cho LSTM (train < 2022, test ∈ {2022,2023}).")

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]
        year_week_test = year_week_idx[test_mask]

        # Train LSTM với Optuna
        model, best_params = train_lstm_with_optuna(X_train, y_train, n_trials=args.n_trials, seed=args.seed)

        # Dự báo test
        try:
            import tensorflow as tf  # just to be safe for prediction
        except:
            pass
        y_pred_test = model.predict(X_test, verbose=0).reshape(-1)

        # Metrics
        y_test_array = y_test.astype(float)
        y_pred_test_array = y_pred_test.astype(float)
        absolute_errors = np.abs(y_test_array - y_pred_test_array)
        total_absolute_error = float(absolute_errors.sum())
        average_absolute_error = float(absolute_errors.mean())
        total_actual_cases = float(y_test_array.sum())
        total_error_ratio = (total_absolute_error / total_actual_cases) * 100 if total_actual_cases > 0 else 0.0
        mse = float(mean_squared_error(y_test_array, y_pred_test_array))

        # ----- Forecast 2024: iterative next-step từ cửa sổ cuối 2023 -----
        # Lấy phần cuối cùng trước tuần đầu 2024
        # Tìm chỉ số cuối cùng của năm 2023 trong dữ liệu gốc
        last_idx_2023 = np.where(nam_arr == 2023)[0]
        if len(last_idx_2023) == 0:
            raise ValueError("Không có dữ liệu năm 2023 để khởi tạo dự báo 2024 (LSTM).")
        last_idx_2023 = last_idx_2023[-1]

        # Lấy cửa sổ cuối cùng kết thúc ở tuần cuối 2023
        # Nếu thiếu quan sát đầu, fallback lấy từ cuối chuỗi
        end_for_window = last_idx_2023 + 1  # vị trí sau cùng 2023
        start_for_window = max(0, end_for_window - window)
        last_window = val_arr[start_for_window:end_for_window].tolist()
        if len(last_window) < window:
            # nếu vẫn thiếu, thêm các giá trị đầu chuỗi để đủ
            prefix_needed = window - len(last_window)
            last_window = val_arr[:prefix_needed].tolist() + last_window

        predictions_2024 = []
        for week in range(1, 53):
            x_in = np.array(last_window[-window:], dtype='float32').reshape(1, window, 1)
            pred = float(model.predict(x_in, verbose=0).reshape(-1)[0])
            pred = max(pred, 0.0)

            # Floor dựa trên historical avg theo tuần (nếu khả dụng)
            hist_avg = hist_avg_map.get(week, overall_hist_avg)
            if hist_avg > 0 and pred < 0.5 * hist_avg:
                pred = 0.75 * hist_avg

            predictions_2024.append(pred)
            last_window.append(pred)

        predictions_2024_all[district] = np.round(predictions_2024).astype(int)

        # Plot
        plt.figure(figsize=(14, 7))
        plt.plot(year_week_test, y_test_array, marker='o', linestyle='-', label='Actual 2022-2023')
        plt.plot(year_week_test, y_pred_test_array, marker='x', linestyle='--', label='Predicted 2022-2023 (LSTM)')
        year_week_2024 = (2024 - 2022) * 52 + np.arange(1, 53)
        plt.plot(year_week_2024, predictions_2024_all[district], marker='o', linestyle='-', label='Predicted 2024')
        plt.axvline(x=52, color='gray', linestyle='--', label='End of 2022')
        plt.axvline(x=104, color='gray', linestyle='--', label='End of 2023')
        plt.title(f'Dự đoán và thực tế số ca Sốt xuất huyết Dengue tại {district} (Model: LSTM, window={window})')
        plt.xlabel('Thời gian (từ 2022)')
        plt.ylabel('Số ca bệnh')
        plt.legend()
        plt.grid(True)
        plt.xticks(ticks=[0, 26, 52, 78, 104, 130, 156],
                   labels=['2022-W1', '2022-W26', '2023-W1', '2023-W26', '2024-W1', '2024-W26', '2025-W1'])
        plot_path = os.path.join(output_folder, f'prediction_plot_{district}_dengue_lstm.png')
        plt.savefig(plot_path)
        plt.close()

    else:
        raise ValueError("Unknown model")

    # --------- Write results ----------
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"Mô hình: {model_name.upper()} | Quận/huyện: {district}\n")
        f.write(f"Mean Squared Error (2022-2023): {mse:.2f}\n")
        f.write(f"Trung bình sai số ca dự đoán 2022-2023: {average_absolute_error:.2f} ca\n")
        f.write(f"Tổng số ca thực tế 2022-2023: {total_actual_cases:.0f} ca\n")
        f.write(f"Tổng sai số tuyệt đối 2022-2023: {total_absolute_error:.2f} ca\n")
        f.write(f"Tỷ lệ sai lệch tổng: {total_error_ratio:.2f}%\n")
        f.write(f"Tổng số ca dự đoán cho 2024: {predictions_2024_all[district].sum():.0f} ca\n")
        f.write(f"{'-'*70}\n")

# --------------
# Run all units
# --------------
for district in all_district_cols:
    print(f"Đang xử lý {district} với mô hình {args.model.upper()}...")
    try:
        process_district(district)
    except Exception as e:
        print(f"Lỗi khi xử lý {district}: {str(e)}")
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Mô hình {args.model.upper()} - Lỗi khi xử lý {district}: {str(e)}\n")

# Tạo bảng dự báo 2024 (giữ nguyên format cũ)
predict_df_2024 = pd.DataFrame({
    'ID': range(len(df[df['T_BTT'] == 'Sốt xuất huyết Dengue']),
                len(df[df['T_BTT'] == 'Sốt xuất huyết Dengue']) + 52),
    'NAM': [2024] * 52,
    'TUAN': range(1, 53),
    'T_BTT': ['Sốt xuất huyết Dengue'] * 52
})
for district in all_district_cols:
    # nếu quận nào lỗi thì điền 0
    col = predictions_2024_all.get(district, [])
    if len(col) == 52:
        predict_df_2024[district] = col
    else:
        predict_df_2024[district] = [0]*52

predict_df_2024.to_csv(predict_file, index=False)

# Ghép dữ liệu gốc + dự báo 2024
original_df = df[df['T_BTT'] == 'Sốt xuất huyết Dengue']
combined_df = pd.concat([original_df, predict_df_2024], ignore_index=True)
combined_df.to_csv(combined_file, index=False)

print(f"Đã hoàn thành! Kết quả được lưu trong folder: {output_folder}")
print(f"- File dự đoán 2024: {predict_file}")
print(f"- File kết hợp dữ liệu gốc và dự đoán: {combined_file}")
                                                                                                                                              