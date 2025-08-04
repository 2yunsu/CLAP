import pandas as pd

# CSV 파일 로드
df = pd.read_csv("VGG-Foley-Sound_dataset_with_CLAP_predictions_all_label.csv")

# alpha 가중치 설정
alpha = 0.2

# 'predicted_materials'에 대한 매핑 (예: material에 대해 사전 정의된 매핑 딕셔너리)
material_mapping = {
    'rock': 'rock',
    'leaf': 'leaf',
    'water': 'water',
    'wood': 'wood',
    'plastic-bag': 'plastic-bag',
    'ceramic': 'ceramic',
    'metal': 'metal',
    'dirt': 'dirt',
    'cloth': 'cloth',
    'plastic': 'plastic',
    'tile': 'tile',
    'gravel': 'gravel',
    'paper': 'paper',
    'drywall': 'drywall',
    'glass': 'glass',
    'grass': 'grass',
    'carpet': 'carpet'
}

# 각 클래스 열 이름 목록
class_columns = ['rock', 'leaf', 'water', 'wood', 'plastic-bag', 'ceramic', 'metal', 
                 'dirt', 'cloth', 'plastic', 'tile', 'gravel', 'paper', 'drywall', 
                 'glass', 'grass', 'carpet']

# 라벨을 계산하여 새로운 열에 추가하는 함수 정의
def calculate_label(row, alpha):
    # 'predicted_materials'의 예측값 가져오기
    predicted_material = row['predicted_materials']
    
    # predicted_material에 해당하는 confidence 값 (0.7 * predicted_material의 가중치)
    material_confidence = alpha
    
    # 나머지 클래스들에 대한 confidence 값 (0.3 * 각 클래스의 예측값)
    class_confidences = (1 - alpha) * row[class_columns]
    
    # 'predicted_material'에 대한 가중치 적용
    if predicted_material in material_mapping:
        material_class = material_mapping[predicted_material]
        class_confidences[material_class] += material_confidence
    
    # 가장 높은 값을 가진 클래스 선택
    return class_confidences.idxmax()

# 각 행에 대해 라벨 계산 후 'final_label' 열로 추가
df['Combined_label'] = df.apply(calculate_label, axis=1, alpha=alpha)

# 결과 CSV 저장
df.to_csv("VGG-Foley-Sound_CLAP_combined.csv", index=False)

print("새로운 라벨이 추가된 CSV 파일이 'output_with_labels.csv'에 저장되었습니다.")
