"""
This is an example using CLAP for zero-shot inference.
"""
from msclap import CLAP
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import random
import torch
import numpy as np

#random seed
random_seed = 0
torch.manual_seed(random_seed)  # torch
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
np.random.seed(random_seed)  # numpy
random.seed(random_seed)  # random


# CSV 파일 불러오기
df = pd.read_csv('/home/CLAP/examples/root_path/vgg-foley-sound/meta/VGG-Foley-Sound_dataset_delete_None.csv')
youtube_ids = df['YouTube_ID']

# Define classes for zero-shot
classes = [
    'rock', 'leaf', 'water', 'wood', 'plastic-bag', 'ceramic', 'metal', 'dirt', 
    'cloth', 'plastic', 'tile', 'gravel', 'paper', 'drywall', 'glass', 'grass', 
    'carpet'
]

# 각 클래스를 열로 추가 (신뢰도 값을 저장할 열)
for cls in classes:
    df[cls] = None

# Add prompt
prompt = 'this is a sound of '
class_prompts = [prompt + x for x in classes]

# Load and initialize CLAP
clap_model = CLAP(version='2023', use_cuda=True)

failed = []

# Loop through each YouTube_ID
for i in tqdm(range(len(youtube_ids))):
    try:
        audio_files = [f'/home/CLAP/examples/root_path/vgg-foley-sound/audio/{youtube_ids[i]}.wav']

        # compute text embeddings from natural text
        text_embeddings = clap_model.get_text_embeddings(class_prompts)

        # compute the audio embeddings from an audio file
        audio_embeddings = clap_model.get_audio_embeddings(audio_files, resample=True)

        # compute the similarity between audio_embeddings and text_embeddings
        similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)

        # softmax to normalize the similarities
        similarity = F.softmax(similarity, dim=1)

        for idx, cls in enumerate(classes):
            confidence = similarity[0, idx].item()  # 각 클래스에 대한 확률 퍼센트로 변환
            df.at[i, cls] = confidence

        # # Get the top prediction (highest similarity)
        # values, indices = similarity[0].topk(1)

        # # 가장 높은 확률의 클래스 인덱스 추출
        # top_class = classes[indices.item()]
        # confidence = values.item()

        # # 해당 예측 결과와 신뢰도 값을 df에 추가
        # df.at[i, 'CLAP_prediction'] = top_class
        # df.at[i, 'CLAP_prediction_confidence'] = confidence

        # 예측 결과 출력
        # print(f"Prediction for {youtube_ids[i]}: {top_class} ({confidence:.2f})")
    except:
        failed.append(youtube_ids[i])
        continue

# print(f"Failed to predict {len(failed)} audio files.")

# 수정된 df를 CSV로 저장
df.to_csv('/home/CLAP/examples/root_path/vgg-foley-sound/meta/VGG-Foley-Sound_dataset_with_CLAP_predictions_all_label.csv', index=False)

print("CLAP predictions and confidence scores saved to CSV.")
