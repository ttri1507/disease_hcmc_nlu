import pandas as pd


xls = "/content/sample_data/SỐ LIỆU NGHIÊN CỨU Dich Benh_010724.xlsx"
tieu_chay_df = pd.read_excel(xls, sheet_name='TIEU CHAY')
thuong_han_df = pd.read_excel(xls, sheet_name='THUONG HAN')
sxh_df = pd.read_excel(xls, sheet_name='SXH')

tieu_chay_filtered = tieu_chay_df[tieu_chay_df['T_BTT'] == 'Tiêu chảy']
thuong_han_filtered = thuong_han_df[thuong_han_df['T_BTT'] == 'Thương hàn']
sxh_filtered = sxh_df[sxh_df['T_BTT'] == 'Sốt xuất huyết Dengue']


combined_df = pd.concat([tieu_chay_filtered, thuong_han_filtered, sxh_filtered])


combined_df = combined_df.rename(columns={combined_df.columns[0]: 'ID'})


combined_df['T_BTT'] = combined_df['T_BTT'].replace({
    'Sốt xuất huyết Dengue': 'sxh',
    'Tiêu chảy': 'tieuchay',
    'Thương hàn': 'thương hàn'
})

output_file_unmerged = 'Filtered_Disease_Data.csv'
combined_df.to_csv(output_file_unmerged, index=False)


combined_df_merged = combined_df.copy()
combined_df_merged['TP_TD'] = combined_df_merged[['Q2', 'Q9', 'TD']].sum(axis=1)


combined_df_merged = combined_df_merged.drop(['Q2', 'Q9', 'TD'], axis=1)


output_file_merged = 'Filtered_Disease_Data_Merged.csv'
combined_df_merged.to_csv(output_file_merged, index=False)

print(f"Đã xử lý và lưu kết quả vào: {output_file_unmerged} và {output_file_merged}")