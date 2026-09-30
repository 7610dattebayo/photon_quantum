import numpy as np
import scipy.stats as stats
from scipy.special import erf

# ---------------------------------------------------------
# 1. パラメータ設定
# ---------------------------------------------------------
EPSILON = 0.15       # GKPエンベロープパラメータ
T_LOSS = 0.97        # 透過率 97%
SIGMA_PHASE = 0.015  # 位相誤差 0.015 rad

sqrt_pi = np.sqrt(np.pi)
half_sqrt_pi = sqrt_pi / 2.0

# ---------------------------------------------------------
# 2. 物理ノイズから位置・運動量変位 (Displacement) への換算
# ---------------------------------------------------------
# ① GKP自身の有限エネルギーによる標準偏差
sigma_gkp = np.sqrt(EPSILON / 2.0)

# ② 光子損失 (Loss) による位置・運動量の分散増加
# Loss Channel (T): q -> sqrt(T)*q + sqrt(1-T)*x_vacuum
# GKP格子のスケール収縮と真空ノイズの付加
sigma_loss_q = np.sqrt((1.0 - T_LOSS) / 2.0)
sigma_loss_p = np.sqrt((1.0 - T_LOSS) / 2.0)

# ③ 位相誤差 (Phase Jitter) による変位
# R(phi) による回転: q -> q*cos(phi) - p*sin(phi)
# GKP状態の平均的な |q|, |p| の大きさ（エネルギー）に依存
# 平均光子数 <n> ≈ 1/(2*EPSILON)
n_avg = 1.0 / (2.0 * EPSILON)
rms_q = np.sqrt(2 * n_avg + 1)  # 実効的なqの広がり
sigma_phase_q = rms_q * SIGMA_PHASE

# ---------------------------------------------------------
# 3. 合成標準偏差 (Total Noise)
# ---------------------------------------------------------
# position (q) 軸方向のエラー分散 -> Z エラーの原因
sigma_total_q = np.sqrt(sigma_gkp**2 + sigma_loss_q**2 + sigma_phase_q**2)

# momentum (p) 軸方向のエラー分散 -> X エラーの原因
sigma_total_p = np.sqrt(sigma_gkp**2 + sigma_loss_p**2)

# ---------------------------------------------------------
# 4. 判定境界 (± sqrt(pi)/2) を超える確率の計算 (1D ガウス積分)
# ---------------------------------------------------------
def error_probability(sigma):
    """標準偏差 sigma のガウスノイズが GKP 判定境界 (sqrt(pi)/2) を超える確率"""
    # 境界外 ( |x| > sqrt(pi)/2 ) の確率密度和 (テール確率)
    # テール確率 = 2 * (1 - CDF(boundary))
    prob_tail = 2.0 * (1.0 - stats.norm.cdf(half_sqrt_pi, loc=0, scale=sigma))
    return prob_tail

# q軸の判定失敗率 (Z論理エラー確率)
p_err_q = error_probability(sigma_total_q)

# p軸の判定失敗率 (X論理エラー確率)
p_err_p = error_probability(sigma_total_p)

# ---------------------------------------------------------
# 5. パウリ確率 (P_I, P_X, P_Z, P_Y) の合成
# ---------------------------------------------------------
P_I = (1.0 - p_err_p) * (1.0 - p_err_q)
P_X = p_err_p * (1.0 - p_err_q)
P_Z = (1.0 - p_err_p) * p_err_q
P_Y = p_err_p * p_err_q

print("===== 正しい理論的パウリ論理エラー率 =====")
print(f"パラメータ: EPSILON={EPSILON}, T={T_LOSS}, sigma_phase={SIGMA_PHASE} rad")
print(f"----------------------------------------")
print(f"q軸標準偏差 (Zエラー要因) sigma_q : {sigma_total_q:.4f} (境界: {half_sqrt_pi:.4f})")
print(f"p軸標準偏差 (Xエラー要因) sigma_p : {sigma_total_p:.4f} (境界: {half_sqrt_pi:.4f})")
print(f"----------------------------------------")
print(f"Identity 確率 (P_I) : {P_I:.2%}")
print(f"Pauli X  確率 (P_X) : {P_X:.2%}")
print(f"Pauli Z  確率 (P_Z) : {P_Z:.2%}")
print(f"Pauli Y  確率 (P_Y) : {P_Y:.2%}")
print(f"総論理エラー率 (1 - P_I) : {1.0 - P_I:.2%}")
