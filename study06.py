import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

SQRT_PI = np.sqrt(np.pi)
HALF_SQRT_PI = SQRT_PI / 2.0


# ---------------------------------------------------------------------
# 1. 均衡（Balanced）データセット生成関数
# ---------------------------------------------------------------------
def generate_gkp_balanced_dataset(
    n_samples_per_class=2500,
    epsilon_range=(0.15, 0.35),
    T_range=(0.80, 0.98),
    sigma_phase_range=(0.00, 0.12),
    seed=42,
):
    np.random.seed(seed)
    # 実用域などノイズが小さい場合はエラー発生率が低いため、大きめの候補を作成
    candidate_size = n_samples_per_class * 500

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
    labels[is_x_error & ~is_z_error] = 1  # X
    labels[~is_x_error & is_z_error] = 2  # Z
    labels[is_x_error & is_z_error] = 3  # Y

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


def add_features(df):
    df_feat = df.copy()
    df_feat["delta_qp"] = df_feat["delta_q"] * df_feat["delta_p"]
    df_feat["delta_r2"] = df_feat["delta_q"] ** 2 + df_feat["delta_p"] ** 2
    df_feat["abs_delta_q"] = np.abs(df_feat["delta_q"])
    df_feat["abs_delta_p"] = np.abs(df_feat["delta_p"])
    return df_feat


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

# ---------------------------------------------------------------------
# 2. 学習用データ（広域ノイズ）とモデルの訓練
# ---------------------------------------------------------------------
print("1. 学習用データセットを作成してモデルを訓練中...")
df_train_raw = generate_gkp_balanced_dataset(
    n_samples_per_class=25000,
    epsilon_range=(0.15, 0.35),
    T_range=(0.80, 0.98),
    sigma_phase_range=(0.00, 0.12),
    seed=42,
)
df_train_feat = add_features(df_train_raw)

X_train = df_train_feat[feature_cols]
y_train = df_train_feat["label"]

model_lgb = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    random_state=42,
    verbose=-1,
)
model_lgb.fit(X_train, y_train)

# ---------------------------------------------------------------------
# 3. テスト用データ（実用域ノイズ & 均衡化）の評価
# ---------------------------------------------------------------------
print("\n2. 実用領域（均衡データ）のデータセットを生成して評価中...")
df_eval_raw = generate_gkp_balanced_dataset(
    n_samples_per_class=1000,
    epsilon_range=(0.05, 0.15),
    T_range=(0.95, 0.99),
    sigma_phase_range=(0.00, 0.03),
    seed=100,
)
df_eval_feat = add_features(df_eval_raw)

X_eval = df_eval_feat[feature_cols]
y_eval = df_eval_feat["label"]

y_pred_eval = model_lgb.predict(X_eval)

# ---------------------------------------------------------------------
# 4. 評価と可視化（0除算対策付き）
# ---------------------------------------------------------------------
acc_eval = accuracy_score(y_eval, y_pred_eval) * 100
print(f"\n==========================================")
print(f"   実用領域（各クラス均等）における評価   ")
print(f"==========================================")
print(f"全体精度 (Accuracy): {acc_eval:.2f}%\n")

label_names = ["I", "X", "Z", "Y"]
cm = confusion_matrix(y_eval, y_pred_eval, labels=[0, 1, 2, 3])

# 安全な正規化（0除算回避）
row_sums = cm.sum(axis=1, keepdims=True)
cm_norm = np.divide(
    cm.astype("float"), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0
)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_names,
    yticklabels=label_names,
)
plt.title("Confusion Matrix (Counts)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.subplot(1, 2, 2)
sns.heatmap(
    cm_norm * 100,
    annot=True,
    fmt=".1f",
    cmap="Greens",
    xticklabels=label_names,
    yticklabels=label_names,
)
plt.title("Normalized Confusion Matrix (%)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()
plt.show()
