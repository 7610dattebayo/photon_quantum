import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 再現性の確保
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 0. データセット生成処理 (前述と同様)
# =====================================================================
SQRT_PI = np.sqrt(np.pi)
HALF_SQRT_PI = SQRT_PI / 2.0


def generate_gkp_balanced_dataset(
    n_samples_per_class=2500,
    epsilon_range=(0.15, 0.35),
    T_range=(0.80, 0.98),
    sigma_phase_range=(0.00, 0.12),
    seed=42,
):
    np.random.seed(seed)
    candidate_size = n_samples_per_class * 100

    epsilons = np.random.uniform(
        epsilon_range[0], epsilon_range[1], candidate_size
    )
    Ts = np.random.uniform(T_range[0], T_range[1], candidate_size)
    sigma_phases = np.random.uniform(
        sigma_phase_range[0], sigma_phase_range[1], candidate_size
    )

    sigma_gkps = np.sqrt(epsilons / 2.0)
    sigma_losses = np.sqrt((1.0 - Ts) / 2.0)
    rms_qs = np.sqrt(2.0 * (1.0 / (2.0 * epsilons)) + 1.0)
    sigma_phase_qs = rms_qs * sigma_phases

    sigma_qs = np.sqrt(sigma_gkps**2 + sigma_losses**2 + sigma_phase_qs**2)
    sigma_ps = np.sqrt(sigma_gkps**2 + sigma_losses**2)

    q_displacements = np.random.normal(0, sigma_qs, candidate_size)
    p_displacements = np.random.normal(0, sigma_ps, candidate_size)

    delta_q = (q_displacements + HALF_SQRT_PI) % SQRT_PI - HALF_SQRT_PI
    delta_p = (p_displacements + HALF_SQRT_PI) % SQRT_PI - HALF_SQRT_PI

    q_shifts = np.round(q_displacements / SQRT_PI).astype(int)
    p_shifts = np.round(p_displacements / SQRT_PI).astype(int)

    is_z_error = q_shifts % 2 != 0
    is_x_error = p_shifts % 2 != 0

    labels = np.zeros(candidate_size, dtype=int)
    labels[is_x_error & ~is_z_error] = 1
    labels[~is_x_error & is_z_error] = 2
    labels[is_x_error & is_z_error] = 3

    df_candidate = pd.DataFrame(
        {
            "delta_q": delta_q,
            "delta_p": delta_p,
            "epsilon": epsilons,
            "T_loss": Ts,
            "sigma_phase": sigma_phases,
            "label": labels,
        }
    )

    balanced_dfs = []
    for label_id in [0, 1, 2, 3]:
        df_label = df_candidate[df_candidate["label"] == label_id]
        available_count = len(df_label)
        if available_count < n_samples_per_class:
            balanced_dfs.append(df_label)
        else:
            balanced_dfs.append(
                df_label.sample(n=n_samples_per_class, random_state=seed)
            )

    return (
        pd.concat(balanced_dfs)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


# データ作成
df_balanced = generate_gkp_balanced_dataset(n_samples_per_class=2500)

# =====================================================================
# 1. 特徴量エンジニアリング (新機能)
# =====================================================================
df_feat = df_balanced.copy()

# 非線形特徴量の追加
df_feat["delta_qp"] = df_feat["delta_q"] * df_feat["delta_p"]  # 交叉項 (Yエラー検出用)
df_feat["delta_r2"] = (
    df_feat["delta_q"] ** 2 + df_feat["delta_p"] ** 2
)  # 二乗距離項
df_feat["abs_delta_q"] = np.abs(df_feat["delta_q"])  # 絶対値
df_feat["abs_delta_p"] = np.abs(df_feat["delta_p"])

feature_cols = [
    "delta_q",
    "delta_p",
    "abs_delta_q",
    "abs_delta_p",
    "delta_qp",
    "delta_r2",
    "epsilon",
    "T_loss",
    "sigma_phase",
]

X = df_feat[feature_cols].values
y = df_feat["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


# =====================================================================
# 2. 残差結合付き ResNet 型 MLP デコーダーモデル
# =====================================================================
class ResBlock(nn.Module):

    def __init__(self, hidden_dim):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.SiLU()  # ReLUより滑らかな活性化関数 SiLU (Swish)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        return self.act(out + residual)  # スキップ接続


class EnhancedGKPDecoder(nn.Module):

    def __init__(self, input_dim=9, hidden_dim=128, num_classes=4):
        super(EnhancedGKPDecoder, self).__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        self.block1 = ResBlock(hidden_dim)
        self.block2 = ResBlock(hidden_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.in_proj(x)
        x = self.block1(x)
        x = self.block2(x)
        return self.out_proj(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnhancedGKPDecoder(input_dim=len(feature_cols)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    model.parameters(), lr=0.003, weight_decay=1e-4
)  # AdamW & Weight Decay
epochs = 50
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs
)  # コサイン減衰スケジューラー

# =====================================================================
# 3. 訓練ループ (Training Loop)
# =====================================================================
print(f"チューニング版モデルの訓練を開始します (Device: {device})...\n")

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_X.size(0)

    scheduler.step()
    epoch_loss = running_loss / len(train_loader.dataset)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {epoch_loss:.4f}")

# =====================================================================
# 4. 評価 (Evaluation)
# =====================================================================
model.eval()
correct = 0
total = 0

class_correct = [0] * 4
class_total = [0] * 4
label_names = ["0: I", "1: X", "2: Z", "3: Y"]

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)

        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

        c = (predicted == batch_y).squeeze()
        for i in range(len(batch_y)):
            label = batch_y[i].item()
            class_total[label] += 1
            class_correct[label] += c[i].item()

print(f"\n--- チューニング後の評価結果 ---")
print(f"全体精度 (Accuracy): {100 * correct / total:.2f}%\n")
print("クラス別精度:")
for i in range(4):
    if class_total[i] > 0:
        acc = 100 * class_correct[i] / class_total[i]
        print(
            f"  {label_names[i]}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})"
        )
