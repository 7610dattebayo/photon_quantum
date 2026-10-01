import lightgbm as lgb
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SQRT_PI = np.sqrt(np.pi)
HALF_SQRT_PI = SQRT_PI / 2.0


# =====================================================================
# 1. データ生成＆特徴量作成関数
# =====================================================================
def generate_gkp_balanced_dataset(
    n_samples_per_class=25000,
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

# =====================================================================
# 2. モデルの学習
# =====================================================================
print("1. データセットの作成とモデルの訓練を実行中...")
df_train_raw = generate_gkp_balanced_dataset(
    n_samples_per_class=25000, seed=42
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


# =====================================================================
# 3. 判定境界の描画関数（raw文字列でSyntaxWarning対策）
# =====================================================================
def plot_decision_boundary(
    model,
    feature_cols,
    epsilon_val=0.20,
    T_val=0.90,
    sigma_phase_val=0.05,
    grid_resolution=300,
):
    q_range = np.linspace(-HALF_SQRT_PI, HALF_SQRT_PI, grid_resolution)
    p_range = np.linspace(-HALF_SQRT_PI, HALF_SQRT_PI, grid_resolution)
    QQ, PP = np.meshgrid(q_range, p_range)

    grid_df = pd.DataFrame(
        {
            "delta_q": QQ.ravel(),
            "delta_p": PP.ravel(),
            "epsilon": epsilon_val,
            "T_loss": T_val,
            "sigma_phase": sigma_phase_val,
        }
    )

    grid_df["delta_qp"] = grid_df["delta_q"] * grid_df["delta_p"]
    grid_df["delta_r2"] = grid_df["delta_q"] ** 2 + grid_df["delta_p"] ** 2
    grid_df["abs_delta_q"] = np.abs(grid_df["delta_q"])
    grid_df["abs_delta_p"] = np.abs(grid_df["delta_p"])

    X_grid = grid_df[feature_cols]

    Z_pred = model.predict(X_grid)
    Z_pred = Z_pred.reshape(QQ.shape)

    plt.figure(figsize=(7, 6))
    cmap = plt.cm.get_cmap("Set1", 4)

    contour = plt.contourf(
        QQ,
        PP,
        Z_pred,
        levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
        cmap=cmap,
        alpha=0.6,
    )

    cbar = plt.colorbar(contour, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["0: I", "1: X", "2: Z", "3: Y"])

    border = HALF_SQRT_PI / 2.0
    rect = patches.Rectangle(
        (-border, -border),
        2 * border,
        2 * border,
        linewidth=2,
        edgecolor="black",
        facecolor="none",
        linestyle="--",
        label="Ideal Boundary",
    )
    plt.gca().add_patch(rect)

    plt.axhline(0, color="gray", linewidth=0.5)
    plt.axvline(0, color="gray", linewidth=0.5)

    # raw文字列 r"..." を使用して SyntaxWarning を回避
    plt.title(
        rf"GKP Decision Boundary in Syndrome Space\n($\epsilon={epsilon_val}, T={T_val}, \sigma_\phi={sigma_phase_val}$)"
    )
    plt.xlabel(r"$\Delta q$ (Displacement along q)")
    plt.ylabel(r"$\Delta p$ (Displacement along p)")
    plt.xlim(-HALF_SQRT_PI, HALF_SQRT_PI)
    plt.ylim(-HALF_SQRT_PI, HALF_SQRT_PI)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.show()


# =====================================================================
# 4. 実行
# =====================================================================
print("2. 判定境界（カラーマップ）を描画中...")
plot_decision_boundary(
    model_lgb,
    feature_cols,
    epsilon_val=0.20,
    T_val=0.90,
    sigma_phase_val=0.05,
)
