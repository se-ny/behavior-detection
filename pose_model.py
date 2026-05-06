import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib

# ─────────────────────────────
# 1. 데이터 로드 & 합치기 (train + val)
# ─────────────────────────────
train_abnormal = pd.read_csv('behavior/train/abnormal/labels/pose_data.csv')
train_normal   = pd.read_csv('behavior/train/normal/labels/pose_data.csv')
val_abnormal   = pd.read_csv('behavior/val/abnormal/labels/pose_data.csv')
val_normal     = pd.read_csv('behavior/val/normal/labels/pose_data.csv')

df = pd.concat([train_abnormal, train_normal, val_abnormal, val_normal], ignore_index=True)
print(f"✅ 전체 데이터: {len(df)}개 (train+val 합침)")

# ─────────────────────────────
# 2. 영상 이름 추출 & 시퀀스 묶기
# ─────────────────────────────
def get_video_name(path):
    filename = path.split('/')[-1]
    if '__frame_' in filename:
        return filename.split('__frame_')[0]
    elif '_frame_' in filename:
        return filename.split('_frame_')[0]
    else:
        return filename

df['video_name'] = df['file_path'].apply(get_video_name)

SEQUENCE_LEN = 30
feature_cols = [c for c in df.columns if c not in ['file_path', 'label', 'video_name']]

sequences, labels = [], []

for video, group in df.groupby('video_name'):
    group = group.sort_values('file_path').reset_index(drop=True)
    feats = group[feature_cols].values
    label = group['label'].iloc[0]

    for i in range(0, len(feats) - SEQUENCE_LEN + 1, SEQUENCE_LEN):
        seq = feats[i:i+SEQUENCE_LEN]
        if len(seq) == SEQUENCE_LEN:
            sequences.append(seq)
            labels.append(label)

X = np.array(sequences, dtype=np.float32)
y = np.array(labels, dtype=np.float32)

print(f"✅ 전체 시퀀스: {len(X)}개")
print(f"   abnormal: {int(y.sum())}개 / normal: {int((y==0).sum())}개")

# ─────────────────────────────
# 3. 정규화 & 분할 (test용 20%만 따로)
# ─────────────────────────────
N, T, F = X.shape
scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1, F)).reshape(N, T, F)
joblib.dump(scaler, 'behavior/scaler.pkl')
print("✅ scaler 저장 완료")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────
# 4. 클래스 가중치
# ─────────────────────────────
class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32)
print(f"✅ 클래스 가중치 - normal: {class_weights[0]:.2f}, abnormal: {class_weights[1]:.2f}")

# ─────────────────────────────
# 5. Dataset & DataLoader
# ─────────────────────────────
class PoseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(PoseDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader  = DataLoader(PoseDataset(X_test,  y_test),  batch_size=32)

# ─────────────────────────────
# 6. LSTM 모델
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMClassifier().to(device)
pos_weight = pos_weight.to(device)
print(f"✅ 학습 디바이스: {device}")

# ─────────────────────────────
# 7. 학습
# ─────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

EPOCHS = 30
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct = 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out, _ = model.lstm(X_batch)
        logits = model.fc[:-1](out[:, -1, :]).squeeze()
        pred = torch.sigmoid(logits)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += ((pred > 0.5) == y_batch).sum().item()

    acc = correct / len(X_train)
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {acc:.4f}")

# ─────────────────────────────
# 8. test로 최종 평가
# ─────────────────────────────
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        out, _ = model.lstm(X_batch)
        logits = model.fc[:-1](out[:, -1, :]).squeeze()
        pred = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        all_preds.extend(pred)
        all_labels.extend(y_batch.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

acc       = (all_preds == all_labels).mean()
tp = ((all_preds == 1) & (all_labels == 1)).sum()
fp = ((all_preds == 1) & (all_labels == 0)).sum()
fn = ((all_preds == 0) & (all_labels == 1)).sum()
precision = tp / (tp + fp + 1e-8)
recall    = tp / (tp + fn + 1e-8)
f1        = 2 * precision * recall / (precision + recall + 1e-8)

print(f"\n🎉 Test Accuracy:  {acc:.4f}")
print(f"   Precision:     {precision:.4f}")
print(f"   Recall:        {recall:.4f}")
print(f"   F1 Score:      {f1:.4f}")

# ─────────────────────────────
# 9. 모델 저장
# ─────────────────────────────
torch.save(model.state_dict(), 'behavior/lstm_model.pth')
print("\n✅ 모델 저장 완료: behavior/lstm_model.pth")