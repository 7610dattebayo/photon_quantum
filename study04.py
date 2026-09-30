import numpy as np
import pandas as pd

SQRT_PI = np.sqrt(np.pi)
HALF_SQRT_PI = SQRT_PI / 2.0


def generate_gkp_balanced_dataset(
n_samples_per_class=10000, # 各クラスごとの目標サンプル数 (計 4 * 10000 = 40000件)
epsilon_range=(
0.15,
0.35,
), # 有限エネルギーノイズを拡大 (デフォルト: 0.10~0.20)
T_range=(0.80, 0.98), # 【方法A】光子損失を拡大 (デフォルト: 0.95~1.00)
sigma_phase_range=(
0.00,
0.12,
), # 位相ジッターを拡大 (デフォルト: 0.00~0.05)
seed=42,
):
"""
ノイズ範囲を広げて過剰サンプリングを行い、 各ラベル (0:I, 1:X, 2:Z, 3:Y) が均等になるよう抽出したデータセットを生成する
"""
np.random.seed(seed)

# 均等なサンプルを集めるため、一時的に大きめの候補データセットを生成
# ノイズが大きい領域でも Y エラーは起きにくいため、高めの倍率で生成
candidate_size = n_samples_per_class * 100

epsilons = np.random.uniform(
epsilon_range[0], epsilon_range[1], candidate_size
)
Ts = np.random.uniform(T_range[0], T_range[1], candidate_size)
sigma_phases = np.random.uniform(
sigma_phase_range[0], sigma_phase_range[1], candidate_size
)

# ノイズ標準偏差の計算
sigma_gkps = np.sqrt(epsilons / 2.0)
sigma_losses = np.sqrt((1.0 - Ts) / 2.0)
rms_qs = np.sqrt(2.0 * (1.0 / (2.0 * epsilons)) + 1.0)
sigma_phase_qs = rms_qs * sigma_phases

sigma_qs = np.sqrt(sigma_gkps**2 + sigma_losses**2 + sigma_phase_qs**2)
sigma_ps = np.sqrt(sigma_gkps**2 + sigma_losses**2)

# 変位サンプリング
q_displacements = np.random.normal(0, sigma_qs, candidate_size)
p_displacements = np.random.normal(0, sigma_ps, candidate_size)

# シンドローム Δq, Δp
delta_q = (q_displacements + HALF_SQRT_PI) % SQRT_PI - HALF_SQRT_PI
delta_p = (p_displacements + HALF_SQRT_PI) % SQRT_PI - HALF_SQRT_PI

# エラー判定
q_shifts = np.round(q_displacements / SQRT_PI).astype(int)
p_shifts = np.round(p_displacements / SQRT_PI).astype(int)

is_z_error = q_shifts % 2 != 0
is_x_error = p_shifts % 2 != 0

labels = np.zeros(candidate_size, dtype=int)
labels[is_x_error & ~is_z_error] = 1 # X
labels[~is_x_error & is_z_error] = 2 # Z
labels[is_x_error & is_z_error] = 3 # Y

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

# -----------------------------------------------------------------
# クラスごとに指定件数ずつ抽出して結合 (Balanced Sampling)
# -----------------------------------------------------------------
balanced_dfs = []
for label_id in [0, 1, 2, 3]:
df_label = df_candidate[df_candidate["label"] == label_id]
available_count = len(df_label)

if available_count < n_samples_per_class:
print(
f"Warning: Label {label_id} の生成数が目標({n_samples_per_class})に届かず、{available_count}件のみ抽出しました。"
)
balanced_dfs.append(df_label)
else:
balanced_dfs.append(
df_label.sample(n=n_samples_per_class, random_state=seed)
)

# シャッフルしてインデックスを振り直す
df_balanced = (
pd.concat(balanced_dfs)
.sample(frac=1.0, random_state=seed)
.reset_index(drop=True)
)

return df_balanced


# 実行例
if __name__ == "__main__":
df_balanced = generate_gkp_balanced_dataset(n_samples_per_class=2500)
print("生成データセットのラベル分布:")
print(df_balanced["label"].value_counts())
