import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.integrate import solve_ivp

# ====================== 1. 定义系统参数（工程常用典型值） ======================
# 转动惯量 (kg·m²)
J1 = 0.5  # 第一个刚体的转动惯量
J2 = 0.3  # 第二个刚体的转动惯量

# 扭转刚度 (N·m/rad)
k1 = 100  # 第一个扭转弹簧刚度
k2 = 80  # 第二个扭转弹簧刚度

# 阻尼参数（Rayleigh阻尼）
alpha = 0.1  # 质量比例阻尼系数
beta = 0.001  # 刚度比例阻尼系数

# 激励参数
M0 = 50  # 激励力矩幅值 (N·m)
omega_exc = 15  # 激励频率（选取接近一阶固有频率的值，易观察共振）
t_total = 30  # 时域仿真总时间 从10s延长至30s
dt = 0.01  # 时间步长

# 频率扫描范围 (rad/s)
omega_range = np.linspace(0, 50, 1000)  # 0到50rad/s，取1000个点

# ====================== 2. 构建系统矩阵 ======================
# 质量矩阵 M
M = np.array([
    [J1, 0],
    [0, J2]
])

# 刚度矩阵 K
K = np.array([
    [k1 + k2, -k2],
    [-k2, k2]
])

# 阻尼矩阵 C (Rayleigh阻尼: C = α*M + β*K)
C = alpha * M + beta * K

# ====================== 3. 求解固有频率和模态向量 ======================
# 求解广义特征值问题: Kφ = λMφ (λ = ω_n²)
eigenvalues, eigenvectors = eig(K, M)

# 提取固有频率（排序，确保从小到大）
omega_n = np.sqrt(np.real(eigenvalues))  # 取实部（避免数值误差导致的虚部）
idx = np.argsort(omega_n)
omega_n = omega_n[idx]
phi = eigenvectors[:, idx]  # 模态矩阵，每一列是一个模态向量

print("===== 系统固有特性 =====")
print(f"第一阶固有频率 ω_n1 = {omega_n[0]:.2f} rad/s")
print(f"第二阶固有频率 ω_n2 = {omega_n[1]:.2f} rad/s")
print(f"激励频率 ω_exc = {omega_exc} rad/s")
print(f"模态矩阵 Φ = \n{phi.round(4)}")

# ====================== 4. 计算幅频特性（稳态响应幅值） ======================
# 初始化幅值存储数组
theta1_amp = np.zeros_like(omega_range)  # θ1的幅值
theta2_amp = np.zeros_like(omega_range)  # θ2的幅值

# 激励向量幅值 F0 = [M0, 0]
F0 = np.array([M0, 0])

# 遍历每个激励频率，计算响应幅值
for i, omega in enumerate(omega_range):
    # 频域下的动力学矩阵: (-ω²M + jωC + K)
    dynamic_matrix = -omega ** 2 * M + 1j * omega * C + K

    # 求解频响函数 H = (K - ω²M + jωC)^-1
    try:
        H = np.linalg.inv(dynamic_matrix)
    except np.linalg.LinAlgError:
        theta1_amp[i] = 0
        theta2_amp[i] = 0
        continue

    # 计算响应幅值: Θ = H * F0，幅值为复数的模
    theta_amp = np.abs(H @ F0)
    theta1_amp[i] = theta_amp[0]
    theta2_amp[i] = theta_amp[1]


# ====================== 5. 求解时域振动响应 ======================
def vibration_ode(t, y):
    """
    状态空间方程：将二阶微分方程转为一阶方程组
    y = [θ1, θ2, dθ1/dt, dθ2/dt]
    返回 dy/dt = [dθ1/dt, dθ2/dt, d²θ1/dt², d²θ2/dt²]
    """
    # 提取状态变量
    theta = y[:2]  # 位移
    d_theta = y[2:]  # 速度

    # 简谐激励：F(t) = [M0*cos(ω_exc*t), 0]
    F = np.array([M0 * np.cos(omega_exc * t), 0])

    # 二阶微分方程：M·d²θ/dt² + C·dθ/dt + K·θ = F
    M_inv = np.linalg.inv(M)
    dd_theta = M_inv @ (F - C @ d_theta - K @ theta)

    return np.concatenate([d_theta, dd_theta])


# 初始条件：静止开始
y0 = np.array([0.0, 0.0, 0.0, 0.0])
t_span = (0, t_total)
t_eval = np.arange(0, t_total, dt)

# 求解ODE（龙格-库塔法）
sol = solve_ivp(
    fun=vibration_ode,
    t_span=t_span,
    y0=y0,
    t_eval=t_eval,
    method='RK45',
    rtol=1e-6
)

# 提取时域结果
t = sol.t
theta1_t = sol.y[0]  # θ1随时间变化
theta2_t = sol.y[1]  # θ2随时间变化

# ====================== 6. 绘制幅频特性+时域振动曲线 ======================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 支持负号

# 调整画布尺寸适配30s长时域
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# ===== 子图1：幅频特性曲线 =====
ax1.plot(omega_range, theta1_amp, label=r'$\theta_1$ 幅值', linewidth=2, color='#1f77b4')
ax1.plot(omega_range, theta2_amp, label=r'$\theta_2$ 幅值', linewidth=2, color='#ff7f0e', linestyle='--')
# 标注固有频率和激励频率
ax1.axvline(x=omega_n[0], color='gray', linestyle=':', label=f'一阶固有频率 {omega_n[0]:.2f} rad/s')
ax1.axvline(x=omega_n[1], color='gray', linestyle='-.', label=f'二阶固有频率 {omega_n[1]:.2f} rad/s')
ax1.axvline(x=omega_exc, color='red', linestyle='-', alpha=0.5, label=f'激励频率 {omega_exc} rad/s')
# 幅频图表属性
ax1.set_xlabel('激励频率 ω (rad/s)', fontsize=12)
ax1.set_ylabel('稳态响应幅值 (rad)', fontsize=12)
ax1.set_title('二自由度受迫振动幅频特性曲线', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 50)

# ===== 子图2：时域振动响应曲线（30s长时） =====
ax2.plot(t, theta1_t, label=r'$\theta_1$ 振动响应', linewidth=2, color='#1f77b4')
ax2.plot(t, theta2_t, label=r'$\theta_2$ 振动响应', linewidth=2, color='#ff7f0e', linestyle='--')
# 标注瞬态/稳态分界（5s为界）
ax2.axvline(x=5, color='green', linestyle=':', alpha=0.7, label='瞬态→稳态分界 (t=5s)')
# 时域图表属性
ax2.set_xlabel('时间 t (s)', fontsize=12)
ax2.set_ylabel('位移 (rad)', fontsize=12)
ax2.set_title(f'二自由度受迫振动时域响应（激励频率 {omega_exc} rad/s，仿真时间30s）', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, t_total)

# 调整布局并保存
plt.tight_layout()
plt.savefig('vibration_amplitude_time_frequency_long.png', dpi=300, bbox_inches='tight')
plt.show()

# ====================== 7. 输出关键结果 ======================
print("\n===== 幅频特性关键结果 =====")
max_theta1 = np.max(theta1_amp)
max_omega1 = omega_range[np.argmax(theta1_amp)]
print(f"θ1 最大幅值 = {max_theta1:.4f} rad (对应频率 {max_omega1:.2f} rad/s)")

max_theta2 = np.max(theta2_amp)
max_omega2 = omega_range[np.argmax(theta2_amp)]
print(f"θ2 最大幅值 = {max_theta2:.4f} rad (对应频率 {max_omega2:.2f} rad/s)")

print("\n===== 时域振动关键结果（30s长时） =====")
# 分阶段计算幅值：瞬态段(0-5s)、稳态段(5-30s)
transient_idx = int(5 / dt)  # 5s对应的索引
theta1_transient_amp = np.max(np.abs(theta1_t[:transient_idx]))
theta1_steady_amp = np.max(np.abs(theta1_t[transient_idx:]))
theta2_transient_amp = np.max(np.abs(theta2_t[:transient_idx]))
theta2_steady_amp = np.max(np.abs(theta2_t[transient_idx:]))

print(f"θ1 瞬态段(0-5s)最大幅值 = {theta1_transient_amp:.4f} rad")
print(f"θ1 稳态段(5-30s)最大幅值 = {theta1_steady_amp:.4f} rad")
print(f"θ2 瞬态段(0-5s)最大幅值 = {theta2_transient_amp:.4f} rad")
print(f"θ2 稳态段(5-30s)最大幅值 = {theta2_steady_amp:.4f} rad")

# 输出10s/20s/30s时刻的位移，观察稳态稳定性
for t_check in [10, 20, 30]:
    idx = np.argmin(np.abs(t - t_check))
    theta1_val = theta1_t[idx]
    theta2_val = theta2_t[idx]
    print(f"t={t_check}s时，θ1位移 = {theta1_val:.4f} rad，θ2位移 = {theta2_val:.4f} rad")