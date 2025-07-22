#!/usr/bin/env python3
"""
简化的增强版gradient_assignment_vrp_KL_Sinkhorn_adpt函数测试
"""

import numpy as np
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, '/home/yuepeng/codes/difusco_cross_pro')

def compute_kl_divergence(P, Q, eps=1e-12):
    """
    计算两个矩阵 P 和 Q 之间的 KL 散度。
    """
    assert P.shape == Q.shape, "输入的矩阵 P 和 Q 必须具有相同的形状"
    
    # 避免 log(0)
    P = np.clip(P, eps, 1.0)
    Q = np.clip(Q, eps, 1.0)
    
    # 计算 KL 散度
    kl_div = np.sum(P * np.log(P / Q))
    return kl_div


def sinkhorn_doubly_stochastic(S, mask=None, max_iter=100, tol=1e-9, eps=1e-12):
    """
    Sinkhorn算法实现双随机矩阵
    """
    S = np.maximum(S, eps)
    
    if mask is not None:
        S = np.where(mask, S, 0.0)
    
    for _ in range(max_iter):
        # 行归一化
        row_sums = S.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, eps)
        S = S / row_sums
        
        # 列归一化
        col_sums = S.sum(axis=0, keepdims=True)
        col_sums = np.maximum(col_sums, eps)
        S = S / col_sums
        
        if mask is not None:
            S = np.where(mask, S, 0.0)
    
    return S


def soft_topk_normalize(x, k=2, temperature=1.0, hard=False, eps=1e-12):
    """
    Soft top-k归一化
    """
    if hard:
        # 硬top-k
        top_k_indices = np.argpartition(x, -k)[-k:]
        result = np.zeros_like(x)
        result[top_k_indices] = x[top_k_indices]
        result = result / (result.sum() + eps)
    else:
        # 软top-k
        x_temp = x / temperature
        x_exp = np.exp(x_temp - np.max(x_temp))
        result = x_exp / (x_exp.sum() + eps)
    
    return result


def row_normalize(matrix, mask=None, eps=1e-12):
    """
    行归一化
    """
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, eps)
    normalized = matrix / row_sums
    
    if mask is not None:
        normalized = np.where(mask, normalized, 0.0)
    
    return normalized


def gradient_assignment_vrp_KL_Sinkhorn_adpt_enhanced(
    C, G,
    depot=0,
    K=2,
    eta_cc=1.0,
    alpha_cc=0.3,
    eta_depot=1.0,
    alpha_depot=0.5,
    sinkhorn_max_iter=100,
    forbid_self_loop=True,
    force_return_to_depot=True,
    min_return_mass=1e-3,
    depot_topk_hard=False,
    depot_topk_temperature=1.0,
    eps=1e-12,
    alpha_adjustment_factor=0.05,
    kl_threshold=0.03,
    preserve_initial=True,
    gradient_clip=1.5,
    adaptive_alpha=True,
    min_alpha=0.02,
    max_alpha=0.8,
    quality_threshold=0.1,
    momentum=0.9,
    history_weight=0.1,
    warmup_steps=3,
    stability_threshold=0.02,
    max_change_ratio=0.5,
    step_counter=0,
    previous_C=None,
    momentum_term=None
):
    """
    基于 KL + Sinkhorn 的 VRP 梯度指派更新（增强版）
    """
    C = np.asarray(C, dtype=float)
    G = np.asarray(G, dtype=float)
    N = C.shape[0]
    assert C.shape == (N, N)
    assert G.shape == (N, N)
    assert depot >= 0 and depot < N

    clients = [i for i in range(N) if i != depot]
    C_original = C.copy()
    
    # 增强的矩阵质量评估
    def evaluate_matrix_quality(matrix):
        """多维度矩阵质量评估"""
        # 计算行熵
        row_entropy = 0
        for i in range(matrix.shape[0]):
            row = matrix[i, :]
            row = row / (row.sum() + eps)
            row = np.maximum(row, eps)
            row_entropy += -np.sum(row * np.log(row))
        
        # 计算方差
        variance = np.var(matrix)
        
        # 综合评分
        normalized_entropy = row_entropy / (N * np.log(N))
        normalized_variance = min(variance, 1.0)
        
        quality_score = 0.5 * normalized_variance + 0.5 * (1 - normalized_entropy)
        
        return quality_score, {
            'entropy': normalized_entropy,
            'variance': normalized_variance
        }
    
    # 评估初始矩阵质量
    if preserve_initial:
        initial_quality, quality_details = evaluate_matrix_quality(C)
    else:
        initial_quality, quality_details = 0.0, {}
    
    # 动量项处理
    if momentum_term is None:
        momentum_term = np.zeros_like(C)
    
    # 计算历史变化
    if previous_C is not None:
        historical_change = C - previous_C
        momentum_term = momentum * momentum_term + (1 - momentum) * historical_change
    else:
        historical_change = np.zeros_like(C)
    
    # 自适应参数调整
    current_alpha_cc = alpha_cc
    current_alpha_depot = alpha_depot
    current_eta_cc = eta_cc
    current_eta_depot = eta_depot
    
    # 多级调整策略
    if step_counter < warmup_steps:
        # 预热期
        warmup_factor = 0.1 + 0.9 * (step_counter / warmup_steps)
        current_alpha_cc *= warmup_factor
        current_alpha_depot *= warmup_factor
        current_eta_cc *= warmup_factor
        current_eta_depot *= warmup_factor
    elif adaptive_alpha and initial_quality > quality_threshold:
        # 基于质量的自适应调整
        quality_factor = min(initial_quality / quality_threshold, 3.0)
        current_alpha_cc = max(alpha_cc / quality_factor, min_alpha)
        current_alpha_depot = max(alpha_depot / quality_factor, min_alpha)
        current_eta_cc = eta_cc / quality_factor
        current_eta_depot = eta_depot / quality_factor
    
    # 确保参数在合理范围内
    current_alpha_cc = np.clip(current_alpha_cc, min_alpha, max_alpha)
    current_alpha_depot = np.clip(current_alpha_depot, min_alpha, max_alpha)

    # Step 1: 客户-客户 KL + exp 梯度步
    Cc = C[np.ix_(clients, clients)]
    Gc = G[np.ix_(clients, clients)]
    
    # 梯度平滑
    if momentum_term is not None:
        Gc_momentum = momentum_term[np.ix_(clients, clients)]
        Gc = Gc + history_weight * Gc_momentum
    
    # 梯度裁剪
    if gradient_clip > 0:
        Gc = np.clip(Gc, -gradient_clip, gradient_clip)
    
    Sc = Cc * np.exp(-current_eta_cc * Gc)
    Sc = np.maximum(Sc, eps)

    # 禁自环
    if forbid_self_loop:
        mask_c = np.ones_like(Sc, dtype=bool)
        np.fill_diagonal(mask_c, False)
        Sc = np.where(mask_c, Sc, 0.0)
    else:
        mask_c = None

    # Sinkhorn算法
    Xc = sinkhorn_doubly_stochastic(
        Sc, mask=mask_c, max_iter=sinkhorn_max_iter, tol=1e-9, eps=eps
    )

    # Step 2: 混合客户子块
    X_full = C.copy()
    X_full[np.ix_(clients, clients)] = Xc
    
    # 局部保护
    if preserve_initial and initial_quality > quality_threshold:
        local_quality_mask = np.zeros_like(C, dtype=bool)
        for i in range(N):
            for j in range(N):
                if C[i, j] > np.percentile(C, 75):
                    local_quality_mask[i, j] = True
        
        local_alpha = current_alpha_cc * 0.5
        C_mid = np.where(local_quality_mask, 
                        (1 - local_alpha) * C + local_alpha * X_full,
                        (1 - current_alpha_cc) * C + current_alpha_cc * X_full)
    else:
        C_mid = (1 - current_alpha_cc) * C + current_alpha_cc * X_full

    # Step 3: depot处理
    depot_row = C_mid[depot, clients]
    depot_grad = G[depot, clients]
    
    # 梯度平滑
    if momentum_term is not None:
        depot_grad_momentum = momentum_term[depot, clients]
        depot_grad = depot_grad + history_weight * depot_grad_momentum
    
    # 梯度裁剪
    if gradient_clip > 0:
        depot_grad = np.clip(depot_grad, -gradient_clip, gradient_clip)
    
    # KL指数步
    depot_step = depot_row * np.exp(-current_eta_depot * depot_grad)
    depot_step = np.maximum(depot_step, eps)
    depot_step = depot_step / depot_step.sum()

    # soft top-k
    depot_step = soft_topk_normalize(
        depot_step, k=K, temperature=depot_topk_temperature, hard=depot_topk_hard, eps=eps
    )

    # 注入回depot行
    if preserve_initial and initial_quality > quality_threshold:
        depot_quality_mask = depot_row > np.percentile(depot_row, 75)
        local_alpha_depot = current_alpha_depot * 0.5
        depot_new_row = np.where(depot_quality_mask,
                               (1 - local_alpha_depot) * depot_row + local_alpha_depot * depot_step,
                               (1 - current_alpha_depot) * depot_row + current_alpha_depot * depot_step)
    else:
        depot_new_row = (1 - current_alpha_depot) * depot_row + current_alpha_depot * depot_step

    C_mid[depot, clients] = depot_new_row
    C_mid[depot, depot] = 0.0

    # Step 4: 客户行处理
    if force_return_to_depot:
        for i in clients:
            row_client = C_mid[i, :]
            to_clients = row_client[clients].sum()
            ret = 1.0 - to_clients
            if ret < min_return_mass:
                ret = min_return_mass
                scale = (1.0 - ret) / max(to_clients, eps)
                C_mid[i, clients] = row_client[clients] * scale
            C_mid[i, depot] = ret

        C_mid = row_normalize(C_mid, mask=None, eps=eps)
    else:
        C_mid = row_normalize(C_mid, eps=eps)

    # Step 5: 数值清理
    C_mid[depot, depot] = 0.0

    # 变化控制和稳定性检查
    change_ratio = np.linalg.norm(C_mid - C) / (np.linalg.norm(C) + eps)
    
    # 变化率限制
    if change_ratio > max_change_ratio:
        scale_factor = max_change_ratio / change_ratio
        C_mid = C + scale_factor * (C_mid - C)
    
    # 计算KL散度
    kl_divergence = compute_kl_divergence(C, C_mid)
    
    # 稳定性检查
    if preserve_initial:
        current_quality, _ = evaluate_matrix_quality(C_mid)
        quality_degradation = initial_quality - current_quality
        
        if quality_degradation > stability_threshold:
            conservative_factor = 0.3
            C_mid = C + conservative_factor * (C_mid - C)
            kl_divergence = compute_kl_divergence(C, C_mid)
    
    # 自适应步长调整
    if step_counter >= warmup_steps:
        if kl_divergence > kl_threshold:
            alpha_cc = max(alpha_cc - alpha_adjustment_factor, min_alpha)
            alpha_depot = max(alpha_depot - alpha_adjustment_factor, min_alpha)
        elif kl_divergence < kl_threshold * 0.5:
            alpha_cc = min(alpha_cc + alpha_adjustment_factor * 0.5, max_alpha)
            alpha_depot = min(alpha_depot + alpha_adjustment_factor * 0.5, max_alpha)

    # 返回结果
    return C_mid, {
        "Xc_client_block": Xc,
        "depot_row_new": depot_new_row,
        "num_active_starts_soft": float((depot_new_row > 1e-6).sum()),
        "kl_divergence": kl_divergence,
        "initial_quality": initial_quality,
        "quality_details": quality_details,
        "change_ratio": change_ratio,
        "current_alpha_cc": current_alpha_cc,
        "current_alpha_depot": current_alpha_depot,
        "momentum_term": momentum_term,
        "step_counter": step_counter + 1,
        "alpha_cc": alpha_cc,
        "alpha_depot": alpha_depot
    }


def create_test_matrix(n=5, depot=0, seed=42):
    """创建测试矩阵"""
    np.random.seed(seed)
    
    # 创建一个有良好结构的初始矩阵
    C = np.zeros((n, n))
    
    # 为客户节点创建循环结构
    clients = [i for i in range(n) if i != depot]
    for i, client in enumerate(clients):
        next_client = clients[(i + 1) % len(clients)]
        C[client, next_client] = 0.6 + 0.2 * np.random.rand()
        C[client, depot] = 0.3 + 0.1 * np.random.rand()
    
    # depot到客户
    for client in clients[:2]:
        C[depot, client] = 0.4 + 0.2 * np.random.rand()
    
    # 添加噪声
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
    
    # 测试多步优化
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
            max_change_ratio=0.3,
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
    
    return results


if __name__ == "__main__":
    # 运行测试
    test_results = test_enhanced_function()
    
    print("\n=== 测试完成 ===")
    print("增强版函数的主要改进：")
    print("1. 动量机制：避免优化过程中的震荡")
    print("2. 多级步长控制：预热期使用小步长，逐步增大")
    print("3. 局部保护：对高质量区域使用更小的步长")
    print("4. 稳定性检查：质量显著下降时自动回退")
    print("5. 变化率限制：避免单步变化过大")
    
    # 打印结果摘要
    print("\n=== 优化过程摘要 ===")
    for result in test_results:
        print(f"步骤 {result['step']}: KL散度={result['kl_divergence']:.4f}, "
              f"变化率={result['change_ratio']:.4f}, "
              f"α_cc={result['alpha_cc']:.4f}")