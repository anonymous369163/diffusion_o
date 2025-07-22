#!/usr/bin/env python3
"""
测试增强版的gradient_assignment_vrp_KL_Sinkhorn_adpt函数
演示如何使用新的动量和自适应步长控制机制
"""

import numpy as np
import matplotlib.pyplot as plt
from optimizer_matrix_demo import gradient_assignment_vrp_KL_Sinkhorn_adpt_enhanced, compute_kl_divergence


def create_test_matrix(n=5, depot=0, seed=42):
    """创建测试矩阵"""
    np.random.seed(seed)
    
    # 创建一个有良好结构的初始矩阵
    C = np.zeros((n, n))
    
    # 为客户节点创建一个较好的循环结构
    clients = [i for i in range(n) if i != depot]
    for i, client in enumerate(clients):
        next_client = clients[(i + 1) % len(clients)]
        C[client, next_client] = 0.6 + 0.2 * np.random.rand()
        C[client, depot] = 0.3 + 0.1 * np.random.rand()
    
    # depot到客户
    for client in clients[:2]:  # 只有前两个客户从depot开始
        C[depot, client] = 0.4 + 0.2 * np.random.rand()
    
    # 添加一些噪声
    C += 0.05 * np.random.rand(n, n)
    C[depot, depot] = 0.0
    
    # 行归一化
    for i in range(n):
        row_sum = C[i, :].sum()
        if row_sum > 0:
            C[i, :] /= row_sum
    
    return C


def create_gradient_matrix(n=5, depot=0, seed=43):
    """创建梯度矩阵"""
    np.random.seed(seed)
    G = np.random.randn(n, n) * 0.5
    G[depot, depot] = 0.0
    return G


def test_enhanced_function():
    """测试增强版函数"""
    print("=== 测试增强版gradient_assignment_vrp_KL_Sinkhorn_adpt函数 ===\n")
    
    # 创建测试数据
    n = 6
    depot = 0
    K = 2
    
    C = create_test_matrix(n, depot)
    G = create_gradient_matrix(n, depot)
    
    print(f"初始矩阵C (n={n}, depot={depot}, K={K}):")
    print(np.round(C, 3))
    print(f"\n梯度矩阵G:")
    print(np.round(G, 3))
    
    # 测试多步优化，展示动量效果
    steps = 5
    momentum_term = None
    previous_C = None
    alpha_cc = 0.3
    alpha_depot = 0.5
    
    results = []
    
    for step in range(steps):
        print(f"\n--- Step {step + 1} ---")
        
        C_new, info = gradient_assignment_vrp_KL_Sinkhorn_adpt_enhanced(
            C, G,
            depot=depot,
            K=K,
            alpha_cc=alpha_cc,
            alpha_depot=alpha_depot,
            step_counter=step,
            previous_C=previous_C,
            momentum_term=momentum_term,
            preserve_initial=True,
            adaptive_alpha=True,
            warmup_steps=3,
            max_change_ratio=0.3,  # 限制变化率
            stability_threshold=0.01
        )
        
        print(f"KL散度: {info['kl_divergence']:.6f}")
        print(f"变化率: {info['change_ratio']:.6f}")
        print(f"初始质量: {info['initial_quality']:.6f}")
        print(f"当前alpha_cc: {info['current_alpha_cc']:.6f}")
        print(f"当前alpha_depot: {info['current_alpha_depot']:.6f}")
        
        if step < 3:
            print(f"预热期 - 使用较小步长")
        
        results.append({
            'step': step + 1,
            'kl_divergence': info['kl_divergence'],
            'change_ratio': info['change_ratio'],
            'initial_quality': info['initial_quality'],
            'alpha_cc': info['current_alpha_cc'],
            'alpha_depot': info['current_alpha_depot']
        })
        
        # 更新历史信息
        previous_C = C.copy()
        momentum_term = info['momentum_term']
        alpha_cc = info['alpha_cc']
        alpha_depot = info['alpha_depot']
        C = C_new
    
    print(f"\n最终矩阵C:")
    print(np.round(C, 3))
    
    # 可视化结果
    plot_results(results)
    
    return results


def plot_results(results):
    """绘制优化过程的结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    steps = [r['step'] for r in results]
    kl_divergences = [r['kl_divergence'] for r in results]
    change_ratios = [r['change_ratio'] for r in results]
    alpha_ccs = [r['alpha_cc'] for r in results]
    alpha_depots = [r['alpha_depot'] for r in results]
    
    # KL散度
    axes[0, 0].plot(steps, kl_divergences, 'b-o', linewidth=2, markersize=6)
    axes[0, 0].set_title('KL散度变化')
    axes[0, 0].set_xlabel('步数')
    axes[0, 0].set_ylabel('KL散度')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 变化率
    axes[0, 1].plot(steps, change_ratios, 'r-s', linewidth=2, markersize=6)
    axes[0, 1].set_title('变化率')
    axes[0, 1].set_xlabel('步数')
    axes[0, 1].set_ylabel('变化率')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Alpha值
    axes[1, 0].plot(steps, alpha_ccs, 'g-^', linewidth=2, markersize=6, label='alpha_cc')
    axes[1, 0].plot(steps, alpha_depots, 'm-v', linewidth=2, markersize=6, label='alpha_depot')
    axes[1, 0].set_title('自适应步长参数')
    axes[1, 0].set_xlabel('步数')
    axes[1, 0].set_ylabel('Alpha值')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 质量指标
    axes[1, 1].axhline(y=results[0]['initial_quality'], color='orange', 
                       linestyle='--', linewidth=2, label='初始质量')
    axes[1, 1].set_title('矩阵质量保护')
    axes[1, 1].set_xlabel('步数')
    axes[1, 1].set_ylabel('质量分数')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/yuepeng/codes/difusco_cross_pro/enhanced_gradient_assignment_test.png', 
                dpi=300, bbox_inches='tight')
    plt.show()


def comparison_test():
    """比较原版和增强版函数的性能"""
    print("\n=== 比较原版和增强版函数 ===\n")
    
    # 创建测试数据
    n = 6
    depot = 0
    K = 2
    
    C = create_test_matrix(n, depot)
    G = create_gradient_matrix(n, depot)
    
    print("原始矩阵:")
    print(np.round(C, 3))
    
    # 使用增强版函数
    C_enhanced, info_enhanced = gradient_assignment_vrp_KL_Sinkhorn_adpt_enhanced(
        C, G,
        depot=depot,
        K=K,
        preserve_initial=True,
        adaptive_alpha=True,
        max_change_ratio=0.2,
        stability_threshold=0.01
    )
    
    print(f"\n增强版结果:")
    print(f"KL散度: {info_enhanced['kl_divergence']:.6f}")
    print(f"变化率: {info_enhanced['change_ratio']:.6f}")
    print(f"初始质量: {info_enhanced['initial_quality']:.6f}")
    print(f"最终矩阵:")
    print(np.round(C_enhanced, 3))


if __name__ == "__main__":
    # 运行测试
    test_results = test_enhanced_function()
    
    # 运行比较测试
    comparison_test()
    
    print("\n=== 测试完成 ===")
    print("增强版函数的主要改进：")
    print("1. 动量机制：避免优化过程中的震荡")
    print("2. 多级步长控制：预热期使用小步长，逐步增大")
    print("3. 局部保护：对高质量区域使用更小的步长")
    print("4. 稳定性检查：质量显著下降时自动回退")
    print("5. 变化率限制：避免单步变化过大")