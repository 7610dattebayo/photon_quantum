import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 固定定数
SQRT_PI = np.sqrt(np.pi)
HALF_SQRT_PI = SQRT_PI / 2.0


# =====================================================================
# 1. データセット生成処理 (GKP Balanced Dataset)
# =====================================================================
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


# =====================================================================
# 2. メイン実行処理
# =====================================================================
if __name__ == "__main__":
    print("1. データセットを生成中...")
    df_balanced = generate_gkp_balanced_dataset(
        n_samples_per_class=2500, seed=42
    )

    print("2. 特徴量エンジニアリングを実行中...")
    df_feat = df_balanced.copy()
    df_feat["delta_qp"] = df_feat["delta_q"] * df_feat["delta_p"]
    df_feat["delta_r2"] = df_feat["delta_q"] ** 2 + df_feat["delta_p"] ** 2
    df_feat["abs_delta_q"] = np.abs(df_feat["delta_q"])
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

    print("3. LightGBM モデルの訓練を開始します...")
    model_lgb = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )
    model_lgb.fit(X_train, y_train)

    print("4. テストデータでの評価を実行中...")
    y_pred = model_lgb.predict(X_test)

    print("\n==========================================")
    print("           LightGBM 評価結果             ")
    print("==========================================")
    print(f"全体精度 (Accuracy): {accuracy_score(y_test, y_pred) * 100:.2f}%\n")

    class_total = [0] * 4
    class_correct = [0] * 4
    label_names = ["0: I", "1: X", "2: Z", "3: Y"]

    for true_label, pred_label in zip(y_test, y_pred):
        class_total[true_label] += 1
        if true_label == pred_label:
            class_correct[true_label] += 1

    print("クラス別精度:")
    for i in range(4):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(
                f"  {label_names[i]}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})"
            )
