import gymnasium as gym
from gymnasium import spaces
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

SQRT_PI = np.sqrt(np.pi)
HALF_SQRT_PI = SQRT_PI / 2.0


# ---------------------------------------------------------------------
# 1. LightGBMモデル学習用のデータ生成 & 訓練関数
# ---------------------------------------------------------------------
def generate_gkp_balanced_dataset(
    n_samples_per_class=5000,
    epsilon_range=(0.10, 0.45),
    T_range=(0.80, 0.98),
    sigma_phase_range=(0.00, 0.20),
    seed=42,
):
    np.random.seed(seed)
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

print("1. 広域ノイズデータによるLightGBMデコーダーの学習...")
df_train_raw = generate_gkp_balanced_dataset(
    n_samples_per_class=5000, seed=42
)
df_train_feat = add_features(df_train_raw)

X_train = df_train_feat[feature_cols]
y_train = df_train_feat["label"]

model_lgb = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    random_state=42,
    verbose=-1,
)
model_lgb.fit(X_train, y_train)
print("-> LightGBMモデルの学習が完了しました。")


# ---------------------------------------------------------------------
# 2. Gym環境（過剰補正の抑制 & シェイピング報酬）
# ---------------------------------------------------------------------
def calculate_gkp_error_label(delta_q_raw, delta_p_raw):
    q_shifts = np.round(delta_q_raw / SQRT_PI).astype(int)
    p_shifts = np.round(delta_p_raw / SQRT_PI).astype(int)

    is_z_error = q_shifts % 2 != 0
    is_x_error = p_shifts % 2 != 0

    if not is_x_error and not is_z_error:
        return 0
    elif is_x_error and not is_z_error:
        return 1
    elif not is_x_error and is_z_error:
        return 2
    else:
        return 3


def create_feature_vector(delta_q, delta_p, epsilon, T_loss, sigma_phase):
    return pd.DataFrame(
        [
            {
                "delta_q": delta_q,
                "delta_p": delta_p,
                "abs_delta_q": np.abs(delta_q),
                "abs_delta_p": np.abs(delta_p),
                "delta_qp": delta_q * delta_p,
                "delta_r2": delta_q**2 + delta_p**2,
                "epsilon": epsilon,
                "T_loss": T_loss,
                "sigma_phase": sigma_phase,
            }
        ]
    )


class DynamicGKPControlEnv(gym.Env):

    def __init__(self, trained_lgb_model):
        super().__init__()
        self.lgb_decoder = trained_lgb_model

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

        # アクション範囲（細かな微調整を可能に）
        self.action_space = spaces.Box(
            low=-0.005, high=0.005, shape=(2,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.base_epsilon = 0.20
        self.base_T = 0.90
        self.base_sigma_phase = 0.05

        self.current_phase_bias = 0.0
        self.current_r_adjust = 0.0

        self.step_count = 0
        obs = self._get_observation()
        return obs, {}

    def _get_observation(self):
        eff_epsilon = np.clip(
            self.base_epsilon - self.current_r_adjust, 0.05, 0.45
        )
        eff_sigma_phase = np.clip(
            self.base_sigma_phase - self.current_phase_bias, 0.0, 0.3
        )

        sigma_gkp = np.sqrt(eff_epsilon / 2.0)
        sigma_loss = np.sqrt((1.0 - self.base_T) / 2.0)
        rms_q = np.sqrt(2.0 * (1.0 / (2.0 * eff_epsilon)) + 1.0)
        sigma_phase_q = rms_q * eff_sigma_phase

        sigma_q = np.sqrt(sigma_gkp**2 + sigma_loss**2 + sigma_phase_q**2)
        sigma_p = np.sqrt(sigma_gkp**2 + sigma_loss**2)

        q_disp = np.random.normal(0, sigma_q)
        p_disp = np.random.normal(0, sigma_p)

        self.last_q_disp = q_disp
        self.last_p_disp = p_disp

        self.delta_q = (q_disp + HALF_SQRT_PI) % SQRT_PI - HALF_SQRT_PI
        self.delta_p = (p_disp + HALF_SQRT_PI) % SQRT_PI - HALF_SQRT_PI

        return np.array(
            [
                self.delta_q,
                self.delta_p,
                eff_epsilon,
                self.base_T,
                eff_sigma_phase,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        self.step_count += 1

        # 1. ノイズの自然ドリフト
        self.base_sigma_phase += 0.001
        self.base_epsilon += 0.0005

        # 2. 制御量の更新（上限をドリフト量に制限し、過剰補正を防止）
        self.current_phase_bias = np.clip(
            self.current_phase_bias + action[0], 0.0, self.base_sigma_phase
        )
        self.current_r_adjust = np.clip(
            self.current_r_adjust + action[1], 0.0, 0.15
        )

        obs = self._get_observation()

        true_label = calculate_gkp_error_label(
            self.last_q_disp, self.last_p_disp
        )

        X_feat = create_feature_vector(
            obs[0], obs[1], obs[2], obs[3], obs[4]
        )
        pred_label = self.lgb_decoder.predict(X_feat)[0]

        # 3. 報酬設計（結果判定 + 実効ノイズレベルに応じた連続報酬）
        if pred_label == true_label:
            reward = 1.0 if true_label == 0 else 0.5
        else:
            reward = -2.0

        # ノイズの抑制具合に対する継続的報酬
        eff_epsilon = obs[2]
        eff_sigma_phase = obs[4]
        reward -= eff_sigma_phase * 3.0 + eff_epsilon * 1.0

        return obs, reward, False, False, {"true_error": true_label}


# ---------------------------------------------------------------------
# 3. RLエージェントの学習
# ---------------------------------------------------------------------
print("\n2. 動的ノイズ環境の構築とPPO学習の開始...")
env = DynamicGKPControlEnv(trained_lgb_model=model_lgb)
check_env(env)

model_ppo = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    verbose=0,
)
model_ppo.learn(total_timesteps=40000)
print("-> PPOエージェントの学習が完了しました。")

# ---------------------------------------------------------------------
# 4. 「RL制御あり」vs「制御なし（放置）」の評価比較
# ---------------------------------------------------------------------
print("\n--- 動的ドリフト下での制御性能評価 (150ステップ) ---")

steps_to_eval = 150


def run_simulation(use_rl=True):
    env_eval = DynamicGKPControlEnv(trained_lgb_model=model_lgb)
    obs, _ = env_eval.reset()
    rewards = []
    errors = []
    effective_sigma_phases = []

    for step in range(steps_to_eval):
        if use_rl:
            action, _ = model_ppo.predict(obs, deterministic=True)
        else:
            action = np.array([0.0, 0.0], dtype=np.float32)

        obs, reward, _, _, info = env_eval.step(action)
        rewards.append(reward)
        errors.append(info["true_error"])
        effective_sigma_phases.append(obs[4])

    return rewards, errors, effective_sigma_phases


rewards_rl, errors_rl, sigma_rl = run_simulation(use_rl=True)
rewards_no_control, errors_no_control, sigma_no_control = run_simulation(
    use_rl=False
)

print(
    f"【RL制御あり】 合計獲得報酬: {sum(rewards_rl):.1f} / 物理エラー発生回数: {sum(1 for e in errors_rl if e != 0)} 回"
)
print(
    f"【制御なし】   合計獲得報酬: {sum(rewards_no_control):.1f} / 物理エラー発生回数: {sum(1 for e in errors_no_control if e != 0)} 回"
)

# グラフ描画
plt.figure(figsize=(10, 4))
plt.plot(sigma_no_control, label="No Control (Drifting)", color="red")
plt.plot(sigma_rl, label="RL Controlled", color="green")
plt.title("Phase Noise Drift Suppression by RL Agent")
plt.xlabel("Step")
plt.ylabel("Effective Sigma Phase")
plt.legend()
plt.grid(True)
plt.show()
