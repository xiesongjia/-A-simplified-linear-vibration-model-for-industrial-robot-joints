import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

# ====================== 1. 定义系统参数（无阻尼，工程常用典型值） ======================
# 转动惯量 (kg·m²)
J1 = 0.5  # 第一个刚体的转动惯量
J2 = 0.3  # 第二个刚体的转动惯量

# 扭转刚度 (N·m/rad)
k1 = 100  # 第一个扭转弹簧刚度
k2 = 80  # 第二个扭转弹簧刚度

# 激励参数
M0 = 50  # 激励力矩幅值 (N·m)

# 频率扫描范围 (rad/s)
omega_range = np.linspace(0, 50, 1000)  # 0到50rad/s，取1000个点

# ====================== 2. 构建系统矩阵（无阻尼） ======================
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

# 无阻尼：删除阻尼矩阵C的定义

# ====================== 3. 求解固有频率和模态向量 ======================
# 求解广义特征值问题: Kφ = λMφ (λ = ω_n²)
eigenvalues, eigenvectors = eig(K, M)

# 提取固有频率（排序，确保从小到大）
omega_n = np.sqrt(np.real(eigenvalues))  # 取实部（避免数值误差导致的虚部）
idx = np.argsort(omega_n)
omega_n = omega_n[idx]
phi = eigenvectors[:, idx]  # 模态矩阵，每一列是一个模态向量

print("===== 无阻尼系统固有特性 =====")
print(f"第一阶固有频率 ω_n1 = {omega_n[0]:.2f} rad/s")
print(f"第二阶固有频率 ω_n2 = {omega_n[1]:.2f} rad/s")
print(f"模态矩阵 Φ = \n{phi.round(4)}")

# ====================== 4. 计算无阻尼幅频特性（稳态响应幅值） ======================
# 初始化幅值存储数组
theta1_amp = np.zeros_like(omega_range)  # θ1的幅值
theta2_amp = np.zeros_like(omega_range)  # θ2的幅值

# 激励向量幅值 F0 = [M0, 0]
F0 = np.array([M0, 0])

# 遍历每个激励频率，计算响应幅值（无阻尼版）
for i, omega in enumerate(omega_range):
    # 无阻尼动力学矩阵: (-ω²M + K)
    dynamic_matrix = -omega ** 2 * M + K

    # 计算行列式，判断是否接近共振（避免奇异矩阵）
    det_val = np.linalg.det(dynamic_matrix)
    if abs(det_val) < 1e-6:  # 接近共振点，幅值设为一个大值（理论无穷）
        theta1_amp[i] = 20  # 自定义大值，便于绘图展示共振峰
        theta2_amp[i] = 20
        continue

    # 求解频响函数 H = (K - ω²M)^-1
    H = np.linalg.inv(dynamic_matrix)

    # 计算响应幅值: Θ = H * F0，幅值为复数的模（无阻尼时为实数）
    theta_amp = np.abs(H @ F0)
    theta1_amp[i] = theta_amp[0]
    theta2_amp[i] = theta_amp[1]

# ====================== 5. 绘制无阻尼幅频特性曲线 ======================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 支持负号
plt.figure(figsize=(10, 6))

# 绘制θ1和θ2的幅频曲线（无阻尼）
plt.plot(omega_range, theta1_amp, label=r'$\theta_1$ 幅值', linewidth=2, color='#1f77b4')
plt.plot(omega_range, theta2_amp, label=r'$\theta_2$ 幅值', linewidth=2, color='#ff7f0e', linestyle='--')

# 标注固有频率（共振点）
plt.axvline(x=omega_n[0], color='gray', linestyle=':', label=f'一阶固有频率 {omega_n[0]:.2f} rad/s')
plt.axvline(x=omega_n[1], color='gray', linestyle='-.', label=f'二阶固有频率 {omega_n[1]:.2f} rad/s')

# 设置图表属性（适配无阻尼特性）
plt.xlabel('激励频率 ω (rad/s)', fontsize=12)
plt.ylabel('响应幅值 (rad)', fontsize=12)
plt.title('无阻尼二自由度受迫振动幅频特性曲线', fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(0, 50)
plt.ylim(0, 25)  # 限定y轴范围，清晰展示共振峰

# 保存图片
plt.savefig('undamped_vibration_amplitude_frequency.png', dpi=300, bbox_inches='tight')
plt.show()

# ====================== 6. 输出关键结果 ======================
print("\n===== 无阻尼幅频特性关键结果 =====")
# 找到θ1的最大幅值及对应频率（排除自定义共振值）
valid_theta1 = theta1_amp[theta1_amp < 19]  # 排除自定义的20
if len(valid_theta1) > 0:
    max_theta1 = np.max(valid_theta1)
    max_omega1 = omega_range[np.argmax(theta1_amp[theta1_amp < 19])]
else:
    max_theta1 = "接近无穷（共振）"
    max_omega1 = omega_n[0]
print(f"θ1 最大幅值（非共振区） = {max_theta1} rad (对应频率 {max_omega1:.2f} rad/s)")

# 找到θ2的最大幅值及对应频率
valid_theta2 = theta2_amp[theta2_amp < 19]
if len(valid_theta2) > 0:
    max_theta2 = np.max(valid_theta2)
    max_omega2 = omega_range[np.argmax(theta2_amp[theta2_amp < 19])]
else:
    max_theta2 = "接近无穷（共振）"
    max_omega2 = omega_n[1]
print(f"θ2 最大幅值（非共振区） = {max_theta2} rad (对应频率 {max_omega2:.2f} rad/s)")
print(f"\n注：无阻尼系统在固有频率 {omega_n[0]:.2f} 和 {omega_n[1]:.2f} rad/s 处发生共振，幅值理论上趋于无穷")