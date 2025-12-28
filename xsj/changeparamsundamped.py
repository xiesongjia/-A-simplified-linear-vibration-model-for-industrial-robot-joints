import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
import warnings

warnings.filterwarnings('ignore')  # 忽略数值计算警告

# ====================== 1. 基础参数配置（可修改） ======================
# 基准参数（无阻尼二自由度系统）
base_params = {
    'J1': 0.5,  # 基准转动惯量1 (kg·m²)
    'J2': 0.3,  # 基准转动惯量2 (kg·m²)
    'k1': 100,  # 基准扭转刚度1 (N·m/rad)
    'k2': 80,  # 基准扭转刚度2 (N·m/rad)
    'M0': 50  # 激励力矩幅值 (N·m)
}

# 敏感性分析配置（指定要分析的参数、变化范围和步长）
# 示例：分析k2（刚度2）在60~100之间变化，步长20；J1在0.3~0.7之间变化，步长0.2
sensitivity_config = {
    'param_name': 'k2',  # 待分析的参数名（可选：J1/J2/k1/k2）
    'param_range': np.arange(60, 110, 20),  # 参数变化范围
    'omega_range': np.linspace(0, 60, 1000)  # 激励频率扫描范围
}


# ====================== 2. 核心计算函数 ======================
def calculate_vibration(params, omega_range):
    """
    计算给定参数下的系统固有特性和幅频特性
    params: 字典，包含J1/J2/k1/k2/M0
    omega_range: 激励频率数组
    return: 固有频率数组、θ1幅值数组、θ2幅值数组
    """
    # 构建矩阵
    M = np.array([[params['J1'], 0], [0, params['J2']]])
    K = np.array([[params['k1'] + params['k2'], -params['k2']], [-params['k2'], params['k2']]])

    # 求解固有频率
    eigenvalues, _ = eig(K, M)
    omega_n = np.sqrt(np.real(eigenvalues))
    omega_n = np.sort(omega_n)  # 排序：一阶、二阶

    # 计算幅频特性
    theta1_amp = np.zeros_like(omega_range)
    theta2_amp = np.zeros_like(omega_range)
    F0 = np.array([params['M0'], 0])

    for i, omega in enumerate(omega_range):
        dynamic_matrix = -omega ** 2 * M + K
        det_val = np.linalg.det(dynamic_matrix)

        # 共振点处理
        if abs(det_val) < 1e-6:
            theta1_amp[i] = 30
            theta2_amp[i] = 30
            continue

        H = np.linalg.inv(dynamic_matrix)
        theta_amp = np.abs(H @ F0)
        theta1_amp[i] = theta_amp[0]
        theta2_amp[i] = theta_amp[1]

    return omega_n, theta1_amp, theta2_amp


# ====================== 3. 批量计算不同参数下的结果 ======================
# 存储结果的字典
results = {
    'param_values': [],  # 参数取值
    'omega_n_list': [],  # 各参数下的固有频率
    'theta1_amp_list': [],  # 各参数下的θ1幅值
    'theta2_amp_list': []  # 各参数下的θ2幅值
}

# 遍历参数范围，批量计算
target_param = sensitivity_config['param_name']
for param_val in sensitivity_config['param_range']:
    # 复制基准参数，修改待分析参数
    current_params = base_params.copy()
    current_params[target_param] = param_val

    # 计算振动特性
    omega_n, theta1_amp, theta2_amp = calculate_vibration(
        current_params, sensitivity_config['omega_range']
    )

    # 存储结果
    results['param_values'].append(param_val)
    results['omega_n_list'].append(omega_n)
    results['theta1_amp_list'].append(theta1_amp)
    results['theta2_amp_list'].append(theta2_amp)

    # 打印当前参数的关键结果
    print(f"===== 参数 {target_param} = {param_val} =====")
    print(f"一阶固有频率: {omega_n[0]:.2f} rad/s, 二阶固有频率: {omega_n[1]:.2f} rad/s")
    print(f"θ1最大幅值（非共振）: {np.max(theta1_amp[theta1_amp < 29]):.4f} rad")
    print(f"θ2最大幅值（非共振）: {np.max(theta2_amp[theta2_amp < 29]):.4f} rad\n")

# ====================== 4. 可视化参数敏感性 ======================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建2个子图：θ1幅频曲线 + θ2幅频曲线
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
colors = plt.cm.viridis(np.linspace(0, 1, len(results['param_values'])))  # 配色方案

# 绘制每条参数对应的曲线
for i, param_val in enumerate(results['param_values']):
    omega_range = sensitivity_config['omega_range']
    theta1_amp = results['theta1_amp_list'][i]
    theta2_amp = results['theta2_amp_list'][i]
    omega_n = results['omega_n_list'][i]

    # 绘制θ1曲线
    ax1.plot(omega_range, theta1_amp, color=colors[i],
             label=f'{target_param}={param_val}, ω_n1={omega_n[0]:.2f} rad/s', linewidth=2)
    # 绘制θ2曲线
    ax2.plot(omega_range, theta2_amp, color=colors[i],
             label=f'{target_param}={param_val}, ω_n2={omega_n[1]:.2f} rad/s', linewidth=2)

# 子图1配置（θ1）
ax1.set_title(f'{target_param}参数敏感性分析 - θ₁幅频特性', fontsize=14, fontweight='bold')
ax1.set_ylabel('θ₁ 幅值 (rad)', fontsize=12)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 35)

# 子图2配置（θ2）
ax2.set_title(f'{target_param}参数敏感性分析 - θ₂幅频特性', fontsize=14, fontweight='bold')
ax2.set_xlabel('激励频率 ω (rad/s)', fontsize=12)
ax2.set_ylabel('θ₂ 幅值 (rad)', fontsize=12)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 35)
ax2.set_xlim(0, 60)

# 整体布局调整
plt.tight_layout()
# 保存图片
plt.savefig(f'{target_param}_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ====================== 5. 输出参数敏感性总结 ======================
print("===== 参数敏感性分析总结 =====")
# 提取固有频率变化
omega_n1_values = [omega_n[0] for omega_n in results['omega_n_list']]
omega_n2_values = [omega_n[1] for omega_n in results['omega_n_list']]

print(f"{target_param}从{results['param_values'][0]}变化到{results['param_values'][-1]}时：")
print(
    f"一阶固有频率变化：{omega_n1_values[0]:.2f} → {omega_n1_values[-1]:.2f} (变化量: {omega_n1_values[-1] - omega_n1_values[0]:.2f} rad/s)")
print(
    f"二阶固有频率变化：{omega_n2_values[0]:.2f} → {omega_n2_values[-1]:.2f} (变化量: {omega_n2_values[-1] - omega_n2_values[0]:.2f} rad/s)")