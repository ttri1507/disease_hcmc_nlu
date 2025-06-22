import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import optuna
import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Disease prediction script for WebGIS integration')
parser.add_argument('--input_file', type=str, default='Filtered_Disease_Data.csv',
                    help='Path to the input CSV file containing disease data')
parser.add_argument('--n_trials', type=int, default=50,
                    help='Number of trials for Optuna optimization (default: 50)')
args = parser.parse_args()

# turn-off optuna log
optuna.logging.set_verbosity(optuna.logging.WARNING)

# create folder
output_folder = 'disease_prediction_results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# result_path
results_file = os.path.join(output_folder, 'prediction_results.txt')
predict_file = os.path.join(output_folder, 'predict_data_tieuchay.csv')
combined_file = os.path.join(output_folder, 'combined_data_tieuchay.csv')  # File mới kết hợp dữ liệu gốc và dự đoán

df = pd.read_csv(args.input_file)

# list districts
all_district_cols = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q12',
                    'GV', 'TB', 'BT', 'PN', 'CC', 'HM', 'TD', 'BC', 'NB', 'CG', 'TP', 'BTA']

# Dictionary for predict (distr)
predictions_2024_all = {district: [] for district in all_district_cols}

def process_district(district):
    # Filter data 
    tieu_chay_district = df[df['T_BTT'] == 'Tiêu chảy'][['NAM', 'TUAN', district]].copy()

    # add feature
    tieu_chay_district['sin_week'] = np.sin(2 * np.pi * tieu_chay_district['TUAN'] / 52)
    tieu_chay_district['cos_week'] = np.cos(2 * np.pi * tieu_chay_district['TUAN'] / 52)
    historical_avg = tieu_chay_district[tieu_chay_district['NAM'].isin([2022, 2023])].groupby('TUAN')[district].mean().to_dict()
    tieu_chay_district['historical_weekly_avg'] = tieu_chay_district['TUAN'].map(historical_avg)

    # add lag & rolling features
    tieu_chay_district[f'{district}_lag1'] = tieu_chay_district[district].shift(1)
    tieu_chay_district[f'{district}_lag2'] = tieu_chay_district[district].shift(2)
    tieu_chay_district[f'{district}_lag3'] = tieu_chay_district[district].shift(3)
    tieu_chay_district[f'{district}_rolling_mean_3'] = tieu_chay_district[district].rolling(window=3).mean()
    tieu_chay_district[f'{district}_rolling_std_3'] = tieu_chay_district[district].rolling(window=3).std()
    tieu_chay_district[f'{district}_rolling_max_3'] = tieu_chay_district[district].rolling(window=3).max()
    tieu_chay_district[f'{district}_diff'] = tieu_chay_district[district] - tieu_chay_district[f'{district}_lag1']
    tieu_chay_district = tieu_chay_district.dropna()

    # Split data create copy data
    train_data = tieu_chay_district[tieu_chay_district['NAM'] < 2022].copy()
    test_data = tieu_chay_district[tieu_chay_district['NAM'].isin([2022, 2023])].copy()

    X_train = train_data[['TUAN', 'NAM', 'sin_week', 'cos_week', 'historical_weekly_avg',
                         f'{district}_lag1', f'{district}_lag2', f'{district}_lag3',
                         f'{district}_rolling_mean_3', f'{district}_rolling_std_3',
                         f'{district}_rolling_max_3', f'{district}_diff']]
    y_train = train_data[district]
    X_test = test_data[['TUAN', 'NAM', 'sin_week', 'cos_week', 'historical_weekly_avg',
                       f'{district}_lag1', f'{district}_lag2', f'{district}_lag3',
                       f'{district}_rolling_mean_3', f'{district}_rolling_std_3',
                       f'{district}_rolling_max_3', f'{district}_diff']]
    y_test = test_data[district]

    # Optuna optimization
    def objective(trial):
        param = {
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'random_state': 42,
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
                            evals=[(dval, 'eval')], early_stopping_rounds=10,
                            verbose_eval=False)
            y_pred_val = model.predict(dval)
            mse = mean_squared_error(y_val, y_pred_val)
            mse_scores.append(mse)

        return np.mean(mse_scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.n_trials)
    best_params = study.best_params
    num_boost_round = best_params.pop('num_boost_round')

    # Train final model
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    best_model = xgb.train(best_params, dtrain, num_boost_round=num_boost_round,
                         evals=[(dtest, 'test')], early_stopping_rounds=10, verbose_eval=False)

    # Predict test
    y_pred_test = best_model.predict(dtest)

    # Calculate errors
    y_test_array = np.array(y_test)
    y_pred_test_array = np.array(y_pred_test)
    absolute_errors = np.abs(y_test_array - y_pred_test_array)
    total_absolute_error = absolute_errors.sum()
    average_absolute_error = absolute_errors.mean()
    total_actual_cases = y_test_array.sum()
    total_error_ratio = (total_absolute_error / total_actual_cases) * 100 if total_actual_cases > 0 else 0
    mse = mean_squared_error(y_test, y_pred_test)

    # Predict 2024
    last_week_2023 = tieu_chay_district[tieu_chay_district['NAM'] == 2023].iloc[-1]
    historical_values = list(tieu_chay_district[tieu_chay_district['NAM'] == 2023][district].values)
    predictions_2024 = []

    for week in range(1, 53):
        sin_week = np.sin(2 * np.pi * week / 52)
        cos_week = np.cos(2 * np.pi * week / 52)
        historical_weekly_avg = historical_avg.get(week, np.mean(list(historical_avg.values())))

        if len(historical_values) >= 3:
            lag1, lag2, lag3 = historical_values[-1], historical_values[-2], historical_values[-3]
        elif len(historical_values) == 2:
            lag1, lag2, lag3 = historical_values[-1], historical_values[-2], historical_values[-2]
        else:
            lag1 = lag2 = lag3 = historical_values[-1]

        if len(historical_values) >= 3:
            rolling_mean_3 = np.mean(historical_values[-3:])
            rolling_std_3 = np.std(historical_values[-3:])
            rolling_max_3 = np.max(historical_values[-3:])
        else:
            rolling_mean_3 = np.mean(historical_values)
            rolling_std_3 = np.std(historical_values) if len(historical_values) > 1 else 0
            rolling_max_3 = np.max(historical_values)

        diff = lag1 - lag2 if len(historical_values) >= 2 else 0

        X_week = pd.DataFrame({
            'TUAN': [week], 'NAM': [2024], 'sin_week': [sin_week], 'cos_week': [cos_week],
            'historical_weekly_avg': [historical_weekly_avg],
            f'{district}_lag1': [lag1], f'{district}_lag2': [lag2], f'{district}_lag3': [lag3],
            f'{district}_rolling_mean_3': [rolling_mean_3], f'{district}_rolling_std_3': [rolling_std_3],
            f'{district}_rolling_max_3': [rolling_max_3], f'{district}_diff': [diff]
        })
        d_week = xgb.DMatrix(X_week)

        pred = max(best_model.predict(d_week)[0], 0)
        if pred < historical_weekly_avg * 0.5:
            pred = historical_weekly_avg * 0.75
        predictions_2024.append(pred)
        historical_values.append(pred)
        
    # save predict to dictionary
    predictions_2024_all[district] = np.round(predictions_2024).astype(int)

    # Plotting
    test_data.loc[:, 'Year_Week'] = (test_data['NAM'] - 2022) * 52 + test_data['TUAN']  # Đảm bảo gán trực tiếp
    prediction_df_2024 = pd.DataFrame({
        'Year': [2024] * 52,
        'Week': range(1, 53),
        'Predicted_Cases_2024': predictions_2024_all[district]
    })
    prediction_df_2024.loc[:, 'Year_Week'] = (prediction_df_2024['Year'] - 2022) * 52 + prediction_df_2024['Week']

    plt.figure(figsize=(14, 7))
    plt.plot(test_data['Year_Week'], y_test, marker='o', linestyle='-', color='green', label='Actual 2022-2023')
    plt.plot(test_data['Year_Week'], y_pred_test, marker='x', linestyle='--', color='red', label='Predicted 2022-2023')
    plt.plot(prediction_df_2024['Year_Week'], prediction_df_2024['Predicted_Cases_2024'], marker='o', linestyle='-', color='blue', label='Predicted 2024')
    plt.axvline(x=52, color='gray', linestyle='--', label='End of 2022')
    plt.axvline(x=104, color='gray', linestyle='--', label='End of 2023')
    plt.title(f'Dự đoán và thực tế số ca bệnh tiêu chảy tại {district} năm 2022-2024')
    plt.xlabel('Thời gian (từ 2022)')
    plt.ylabel('Số ca bệnh')
    plt.legend()
    plt.grid(True)
    plt.xticks(ticks=[0, 26, 52, 78, 104, 130, 156], labels=['2022-W1', '2022-W26', '2023-W1', '2023-W26', '2024-W1', '2024-W26', '2025-W1'])

    # save fig
    plot_path = os.path.join(output_folder, f'prediction_plot_{district}.png')
    plt.savefig(plot_path)
    plt.close()

    # write results
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Kết quả cho quận/huyện: {district}\n")
        f.write(f"Mean Squared Error (2022-2023): {mse:.2f}\n")
        f.write(f"Trung bình sai số ca dự đoán của 2022-2023: {average_absolute_error:.2f} ca\n")
        f.write(f"Tổng số ca thực tế trong 2022-2023: {total_actual_cases:.0f} ca\n")
        f.write(f"Tổng sai số tuyệt đối trong 2022-2023: {total_absolute_error:.2f} ca\n")
        f.write(f"Tỷ lệ sai lệch tổng: {total_error_ratio:.2f}%\n")
        f.write(f"Tổng số ca dự đoán cho 2024: {predictions_2024_all[district].sum():.0f} ca\n")
        f.write(f"{'-'*50}\n")

# handle any districts
for district in all_district_cols:
    print(f"Đang xử lý {district}...")
    try:
        process_district(district)
    except Exception as e:
        print(f"Lỗi khi xử lý {district}: {str(e)}")
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Lỗi khi xử lý {district}: {str(e)}\n")

# create dataframe
predict_df_2024 = pd.DataFrame({
    'ID': range(len(df[df['T_BTT'] == 'Tiêu chảy']), len(df[df['T_BTT'] == 'Tiêu chảy']) + 52),
    'NAM': [2024] * 52,
    'TUAN': range(1, 53),
    'T_BTT': ['Tiêu chảy'] * 52
})
for district in all_district_cols:
    predict_df_2024[district] = predictions_2024_all[district]

# predict for 2024
predict_df_2024.to_csv(predict_file, index=False)

# "fusion" data
original_df = df[df['T_BTT'] == 'Tiêu chảy']  # Lấy dữ liệu gốc chỉ cho 'Tiêu chảy'
combined_df = pd.concat([original_df, predict_df_2024], ignore_index=True)

# merge data
combined_df.to_csv(combined_file, index=False)

print(f"Đã hoàn thành! Kết quả được lưu trong folder: {output_folder}")
print(f"- File dự đoán 2024: {predict_file}")
print(f"- File kết hợp dữ liệu gốc và dự đoán: {combined_file}")