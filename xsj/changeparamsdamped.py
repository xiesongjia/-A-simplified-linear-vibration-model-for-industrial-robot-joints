import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
import warnings

warnings.filterwarnings('ignore')  # 忽略数值计算警告

# ====================== 1. 全局配置（可自定义） ======================
# 基准参数（带Rayleigh阻尼的二自由度系统）
base_params = {
    'J1': 0.5,  # 转动惯量1 (kg·m²)
    'J2': 0.3,  # 转动惯量2 (kg·m²)
    'k1': 100,  # 扭转刚度1 (N·m/rad)
    'k2': 80,  # 扭转刚度2 (N·m/rad)
    'alpha': 0.1,  # 质量比例阻尼系数
    'beta': 0.001,  # 刚度比例阻尼系数
    'M0': 50  # 激励力矩幅值 (N·m)
}

# 待分析的参数列表及变化范围（可增删）
# 格式：{参数名: [变化范围, 步长], ...}
analysis_params = {
    'alpha': [0.05, 0.2, 0.05],  # 阻尼α：0.05~0.2，步长0.05
    'beta': [0.0005, 0.002, 0.0005],  # 阻尼β：0.0005~0.002，步长0.0005
    'k1': [80, 140, 20],  # 刚度k1：80~140，步长20
    'k2': [60, 120, 20],  # 刚度k2：60~120，步长20
    'J1': [0.3, 0.9, 0.2],  # 转动惯量J1：0.3~0.9，步长0.2
    'J2': [0.1, 0.5, 0.1]  # 转动惯量J2：0.1~0.5，步长0.1
}

# 激励频率扫描范围（覆盖所有参数下的固有频率）
omega_range = np.linspace(0, 70, 1000)

# 图片保存路径（当前目录，可修改）
save_path = "./sensitivity_plots/"
import os

if not os.path.exists(save_path):
    os.makedirs(save_path)


# ====================== 2. 核心计算函数 ======================
def calculate_vibration(params, omega_range):
    """
    计算给定参数下的系统固有特性和幅频特性（带阻尼）
    params: 字典，包含所有系统参数
    omega_range: 激励频率数组
    return: 固有频率数组、θ1幅值数组、θ2幅值数组、阻尼比ζ1/ζ2
    """
    # 构建矩阵
    M = np.array([[params['J1'], 0], [0, params['J2']]])
    K = np.array([[params['k1'] + params['k2'], -params['k2']], [-params['k2'], params['k2']]])
    C = params['alpha'] * M + params['beta'] * K  # Rayleigh阻尼

    # 求解固有频率（无阻尼固有频率）
    eigenvalues, eigenvectors = eig(K, M)
    omega_n = np.sqrt(np.real(eigenvalues))
    idx = np.argsort(omega_n)
    omega_n = omega_n[idx]  # 排序：一阶、二阶
    phi = eigenvectors[:, idx]  # 模态矩阵

    # 计算模态阻尼比
    zeta1 = (phi[:, 0].T @ C @ phi[:, 0]) / (2 * omega_n[0] * phi[:, 0].T @ M @ phi[:, 0])
    zeta2 = (phi[:, 1].T @ C @ phi[:, 1]) / (2 * omega_n[1] * phi[:, 1].T @ M @ phi[:, 1])
    zeta = [np.real(zeta1), np.real(zeta2)]

    # 计算幅频特性（频域法）
    theta1_amp = np.zeros_like(omega_range)
    theta2_amp = np.zeros_like(omega_range)
    F0 = np.array([params['M0'], 0])

    for i, omega in enumerate(omega_range):
        # 带阻尼动力学矩阵：-ω²M + jωC + K
        dynamic_matrix = -omega ** 2 * M + 1j * omega * C + K

        try:
            H = np.linalg.inv(dynamic_matrix)
            theta_amp = np.abs(H @ F0)
            theta1_amp[i] = theta_amp[0]
            theta2_amp[i] = theta_amp[1]
        except:
            theta1_amp[i] = 0
            theta2_amp[i] = 0

    return omega_n, zeta, theta1_amp, theta2_amp


# ====================== 3. 敏感性分析主函数 ======================
def analyze_sensitivity(base_params, param_name, param_min, param_max, param_step, omega_range):
    """
    分析单个参数的敏感性，生成对比图并输出量化结果
    """
    # 生成参数变化序列
    param_values = np.arange(param_min, param_max + param_step, param_step)
    if param_name in ['beta']:  # 浮点数步长修正
        param_values = np.round(param_values, 6)

    # 存储结果
    results = {
        'param_values': param_values,
        'omega_n1': [], 'omega_n2': [],
        'zeta1': [], 'zeta2': [],
        'max_theta1': [], 'max_theta2': [],
        'theta1_amp_list': [], 'theta2_amp_list': []
    }

    # 遍历参数值计算
    for val in param_values:
        current_params = base_params.copy()
        current_params[param_name] = val

        # 计算振动特性
        omega_n, zeta, theta1_amp, theta2_amp = calculate_vibration(current_params, omega_range)

        # 存储关键指标
        results['omega_n1'].append(omega_n[0])
        results['omega_n2'].append(omega_n[1])
        results['zeta1'].append(zeta[0])
        results['zeta2'].append(zeta[1])
        results['max_theta1'].append(np.max(theta1_amp))
        results['max_theta2'].append(np.max(theta2_amp))
        results['theta1_amp_list'].append(theta1_amp)
        results['theta2_amp_list'].append(theta2_amp)

    # ====================== 4. 生成敏感性分析图 ======================
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(param_values)))  # 渐变配色

    # 绘制θ1幅频曲线
    for i, val in enumerate(param_values):
        ax1.plot(omega_range, results['theta1_amp_list'][i], color=colors[i],
                 label=f'{param_name}={val}, ω_n1={results["omega_n1"][i]:.2f}, ζ1={results["zeta1"][i]:.4f}',
                 linewidth=2)
    ax1.set_title(f'{param_name}参数敏感性分析 - θ₁幅频特性', fontsize=14, fontweight='bold')
    ax1.set_ylabel('θ₁ 幅值 (rad)', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, np.max(results['max_theta1']) * 1.2)  # 自适应y轴

    # 绘制θ2幅频曲线
    for i, val in enumerate(param_values):
        ax2.plot(omega_range, results['theta2_amp_list'][i], color=colors[i],
                 label=f'{param_name}={val}, ω_n2={results["omega_n2"][i]:.2f}, ζ2={results["zeta2"][i]:.4f}',
                 linewidth=2)
    ax2.set_title(f'{param_name}参数敏感性分析 - θ₂幅频特性', fontsize=14, fontweight='bold')
    ax2.set_xlabel('激励频率 ω (rad/s)', fontsize=12)
    ax2.set_ylabel('θ₂ 幅值 (rad)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, np.max(results['max_theta2']) * 1.2)
    ax2.set_xlim(0, 70)

    # 保存图片
    plt.tight_layout()
    fig_name = f"{param_name}_sensitivity.png"
    plt.savefig(save_path + fig_name, dpi=300, bbox_inches='tight')
    plt.close()

    # ====================== 5. 输出量化敏感性结果 ======================
    print(f"\n===== {param_name} 参数敏感性分析结果 =====")
    print(f"参数变化范围：{param_min} → {param_max} (步长 {param_step})")
    # 固有频率变化
    delta_omega_n1 = results['omega_n1'][-1] - results['omega_n1'][0]
    delta_omega_n2 = results['omega_n2'][-1] - results['omega_n2'][0]
    print(
        f"一阶固有频率变化：{results['omega_n1'][0]:.2f} → {results['omega_n1'][-1]:.2f} (Δ={delta_omega_n1:.2f} rad/s)")
    print(
        f"二阶固有频率变化：{results['omega_n2'][0]:.2f} → {results['omega_n2'][-1]:.2f} (Δ={delta_omega_n2:.2f} rad/s)")
    # 阻尼比变化（仅阻尼参数有效）
    if param_name in ['alpha', 'beta']:
        delta_zeta1 = results['zeta1'][-1] - results['zeta1'][0]
        delta_zeta2 = results['zeta2'][-1] - results['zeta2'][0]
        print(f"一阶阻尼比变化：{results['zeta1'][0]:.4f} → {results['zeta1'][-1]:.4f} (Δ={delta_zeta1:.4f})")
        print(f"二阶阻尼比变化：{results['zeta2'][0]:.4f} → {results['zeta2'][-1]:.4f} (Δ={delta_zeta2:.4f})")
    # 最大幅值变化
    delta_theta1 = results['max_theta1'][-1] - results['max_theta1'][0]
    delta_theta2 = results['max_theta2'][-1] - results['max_theta2'][0]
    print(f"θ1最大幅值变化：{results['max_theta1'][0]:.4f} → {results['max_theta1'][-1]:.4f} (Δ={delta_theta1:.4f} rad)")
    print(f"θ2最大幅值变化：{results['max_theta2'][0]:.4f} → {results['max_theta2'][-1]:.4f} (Δ={delta_theta2:.4f} rad)")

    return results


# ====================== 6. 批量执行所有参数的敏感性分析 ======================
all_results = {}  # 存储所有参数的分析结果
for param_name, (min_val, max_val, step_val) in analysis_params.items():
    print(f"\n开始分析 {param_name} 参数...")
    results = analyze_sensitivity(base_params, param_name, min_val, max_val, step_val, omega_range)
    all_results[param_name] = results

# ====================== 7. 生成敏感性汇总表（可选） ======================
print("\n===== 所有参数敏感性汇总 =====")
print(f"{'参数名':<8} {'一阶固有频率Δ':<15} {'二阶固有频率Δ':<15} {'θ1幅值Δ':<15} {'θ2幅值Δ':<15}")
print("-" * 70)
for param_name, res in all_results.items():
    delta_omega_n1 = res['omega_n1'][-1] - res['omega_n1'][0]
    delta_omega_n2 = res['omega_n2'][-1] - res['omega_n2'][0]
    delta_theta1 = res['max_theta1'][-1] - res['max_theta1'][0]
    delta_theta2 = res['max_theta2'][-1] - res['max_theta2'][0]
    print(
        f"{param_name:<8} {delta_omega_n1:<15.2f} {delta_omega_n2:<15.2f} {delta_theta1:<15.4f} {delta_theta2:<15.4f}")

print(f"\n所有敏感性分析图已保存至：{save_path}")