import pandas as pd

# CSV 파일 로드
csv_file_path = 'VGG-Foley-Sound_CLAP_combined.csv'
df = pd.read_csv(csv_file_path)

# 'predicted_materials' 열에서 빈 값 (NaN 또는 비어있는 값)이 있는 행 제거
df_cleaned = df.dropna(subset=['Combined_label'])

# 결과를 새로운 CSV 파일로 저장
output_file_path = 'VGG-Foley-Sound_CLAP_combined_delete_None.csv'
df_cleaned.to_csv(output_file_path, index=False)

print(f"Rows with missing values in 'predicted_materials' have been removed. Cleaned file saved to {output_file_path}.")
