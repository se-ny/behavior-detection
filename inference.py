import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
import os

# ─────────────────────────────
# 1. LSTM 모델 정의 (학습때랑 동일해야 함)
# ─────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=99, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()

# ─────────────────────────────
# 2. 모델 로드
# ─────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMClassifier().to(device)
model.load_state_dict(torch.load('behavior/lstm_model.pth', map_location=device))
model.eval()
print(f"✅ 모델 로드 완료 | 디바이스: {device}")
import joblib
scaler = joblib.load('behavior/scaler.pkl')
print("✅ scaler 로드 완료")

# ─────────────────────────────
# 3. MediaPipe 초기화
# ─────────────────────────────
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

SEQUENCE_LEN = 30

# ─────────────────────────────
# 4. 영상 하나 inference 함수
# ─────────────────────────────
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 영상 못 열었어: {video_path}")
        return None

    frame_buffer = []
    results_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)

        if result.pose_landmarks:
            landmarks = []
            for lm in result.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            frame_buffer.append(landmarks)

            # 30프레임 쌓이면 예측
            if len(frame_buffer) == SEQUENCE_LEN:
                seq = np.array(frame_buffer, dtype=np.float32)
                seq = scaler.transform(seq)
                seq_tensor = torch.tensor(seq.astype(np.float32)).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(seq_tensor)
                    prob = torch.sigmoid(logits).item()
                    pred = 1 if prob > 0.5 else 0
                    results_list.append((pred, prob))

                frame_buffer = []  # 버퍼 초기화

    cap.release()

    if not results_list:
        print(f"⚠️ 예측 결과 없음 (관절 미검출): {video_path}")
        return None

    # 영상 전체 결과 집계
    abnormal_count = sum(1 for r, _ in results_list if r == 1)
    total = len(results_list)
    abnormal_ratio = abnormal_count / total
    final_pred = 1 if abnormal_ratio >= 0.3 else 0  # 30% 이상 이상이면 abnormal

    return {
        'video': os.path.basename(video_path),
        'final': '🚨 ABNORMAL' if final_pred == 1 else '✅ NORMAL',
        'abnormal_ratio': f"{abnormal_ratio:.2%}",
        'total_sequences': total
    }

# ─────────────────────────────
# 5. val 폴더 전체 테스트
# ─────────────────────────────
val_folders = [
    'behavior/val/abnormal/videos',
    'behavior/val/normal/videos'
]

for folder in val_folders:
    if not os.path.exists(folder):
        print(f"⚠️ 폴더 없음: {folder}")
        continue

    print(f"\n📂 {folder}")
    print("-" * 50)

    for fname in os.listdir(folder):
        if not fname.endswith('.mp4'):
            continue
        video_path = os.path.join(folder, fname)
        result = predict_video(video_path)
        if result:
            print(f"{result['final']} | {result['video']} | 이상비율: {result['abnormal_ratio']} | 시퀀스수: {result['total_sequences']}")

pose.close()
print("\n✅ inference 완료!")