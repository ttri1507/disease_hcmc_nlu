# Dự án Dự báo Sớm Bệnh Truyền Nhiễm tại TP.HCM

Dự án này bao gồm các script Python để tiền xử lý dữ liệu và dự báo số ca bệnh Tiêu chảy và Sốt xuất huyết Dengue tại 24 quận/huyện của TP.HCM, sử dụng mô hình XGBoost và tích hợp với WebGIS. Dưới đây là hướng dẫn chi tiết để cài đặt và sử dụng các file code.

Nội dung 7.22: Phát triển module tự động hóa quy trình các mô hình dự báo sớm BTN liên quan đến các yếu tố BĐKH theo phân vùng ở quy mô cộng đồng	

Yêu cầu: Module tự động hóa quy trình dự báo BTN dựa trên mô hình AI

## Nội dung dự án

- `preprocessing_data.py`: Tiền xử lý dữ liệu từ file Excel, tạo hai file CSV: `Filtered_Disease_Data.csv` (chưa gộp cột) và `Filtered_Disease_Data_Merged.csv` (đã gộp cột `Q2`, `Q9`, `TD` thành `TP_TD`).
- `disease_prediction.py`: Dự báo số ca bệnh Tiêu chảy dựa trên file `Filtered_Disease_Data.csv`.
- `disease_prediction_dengue.py`: Dự báo số ca bệnh Sốt xuất huyết Dengue dựa trên file `Filtered_Disease_Data.csv`.

## Yêu cầu hệ thống

- Môi trường: Google Colab hoặc máy cục bộ với Python 3.7 hoặc cao hơn.
- Thư viện: Xem file `requirements.txt` để cài đặt các thư viện cần thiết.

## Hướng dẫn cài đặt

1. **Chuẩn bị môi trường trên Google Colab**:
   - Mở Google Colab và tạo một notebook mới.
   - Tải file Excel `SỐ LIỆU NGHIÊN CỨU Dich Benh_010724.xlsx` lên thư mục `/content/sample_data/` trên Colab (kéo thả file vào thanh bên trái hoặc sử dụng lệnh `files.upload()`).
   - Cài đặt các thư viện cần thiết bằng cách chạy lệnh sau trong ô code:
     ```bash
     !pip install -r requirements.txt
     ```
   - Tải ba file Python (`preprocessing_data.py`, `disease_prediction.py`, `disease_prediction_dengue.py`) lên cùng thư mục với notebook.

2. **Cài đặt trên máy cục bộ**:
   - Cài đặt Python 3.7+ từ [python.org](https://www.python.org/downloads/).
   - Tạo thư mục dự án và đặt file Excel, các file Python, và `requirements.txt` vào đó.
   - Mở terminal, di chuyển đến thư mục dự án và chạy:
     ```bash
     pip install -r requirements.txt
     ```

## Hướng dẫn sử dụng

### 1. Chạy tiền xử lý dữ liệu (`preprocessing_data.py`)
- **Mục đích**: Tạo hai file CSV từ file Excel đầu vào.
- **Cách chạy**:
  - Trên Google Colab: Chạy file `preprocessing_data.py` bằng lệnh:
    ```bash
    !python preprocessing_data.py
    ```
  - Kết quả: Hai file `Filtered_Disease_Data.csv` và `Filtered_Disease_Data_Merged.csv` sẽ được tạo trong thư mục hiện tại.
- **Lưu ý**: Đảm bảo file Excel nằm tại đường dẫn `/content/sample_data/SỐ LIỆU NGHIÊN CỨU Dich Benh_010724.xlsx`. Nếu chạy cục bộ, điều chỉnh đường dẫn trong code.

### 2. Chạy dự báo Tiêu chảy (`disease_prediction.py`)
- **Mục đích**: Dự báo số ca bệnh Tiêu chảy cho năm 2024 dựa trên file `Filtered_Disease_Data.csv`.
- **Cách chạy**:
  - Mặc định (sử dụng file mặc định):
    ```bash
    !python disease_prediction.py
    ```
  - Với file tùy chỉnh:
    ```bash
    !python disease_prediction.py --input_file /path/to/Filtered_Disease_Data.csv
    ```
  - Với số vòng lặp Optuna tùy chỉnh (mặc định 50):
    ```bash
    !python disease_prediction.py --input_file /path/to/Filtered_Disease_Data.csv --n_trials 100
    ```
- **Kết quả**:
  - Thư mục `disease_prediction_results` chứa:
    - `prediction_results.txt`: File văn bản ghi chỉ số MSE, sai số, và tổng số ca dự đoán.
    - `predict_data_tieuchay.csv`: File CSV chứa dự đoán cho năm 2024.
    - `combined_data_tieuchay.csv`: File CSV kết hợp dữ liệu gốc và dự đoán.
    - Các file PNG (`prediction_plot_{district}.png`): Biểu đồ so sánh thực tế và dự báo.

### 3. Chạy dự báo Sốt xuất huyết Dengue (`disease_prediction_dengue.py`)
- **Mục đích**: Dự báo số ca bệnh Sốt xuất huyết Dengue cho năm 2024 dựa trên file `Filtered_Disease_Data.csv`.
- **Cách chạy**:
  - Mặc định (sử dụng file mặc định):
    ```bash
    !python disease_prediction_dengue.py
    ```
  - Với file tùy chỉnh:
    ```bash
    !python disease_prediction_dengue.py --input_file /path/to/Filtered_Disease_Data.csv
    ```
  - Với số vòng lặp Optuna tùy chỉnh (mặc định 50):
    ```bash
    !python disease_prediction_dengue.py --input_file /path/to/Filtered_Disease_Data.csv --n_trials 100
    ```
- **Kết quả**:
  - Thư mục `disease_prediction_results_dengue` chứa:
    - `prediction_results_dengue.txt`: File văn bản ghi chỉ số MSE, sai số, và tổng số ca dự đoán.
    - `predict_data_dengue.csv`: File CSV chứa dự đoán cho năm 2024.
    - `combined_data_dengue.csv`: File CSV kết hợp dữ liệu gốc và dự đoán.
    - Các file PNG (`prediction_plot_{district}_dengue.png`): Biểu đồ so sánh thực tế và dự báo.

## Lưu ý
- Đảm bảo file đầu vào `Filtered_Disease_Data.csv` có cấu trúc phù hợp (các cột `NAM`, `TUAN`, `T_BTT`, và 24 quận/huyện).
- Nếu gặp lỗi, kiểm tra đường dẫn file hoặc cài đặt lại các thư viện theo `requirements.txt`.
- Dữ liệu đầu ra có thể được tải xuống từ Google Colab bằng cách nhấp chuột phải vào thư mục kết quả và chọn "Download".


