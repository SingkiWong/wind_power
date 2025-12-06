"""
Causal Discovery Module for Wind Power Prediction
因果发现模块 - 从相关性中剥离真实因果关系

核心算法：
1. PCMCI (Peter-Clark Momentary Conditional Independence)
2. LiNGAM (Linear Non-Gaussian Acyclic Model)
3. 物理约束因果发现

应用场景：
- 动态尾流拓扑图谱构建
- 故障传播路径分析
- 根因定位
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from collections import defaultdict
import warnings


class ConditionalIndependenceTest:
    """
    条件独立性检验基类
    
    用于判断两个变量在给定条件集下是否独立
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: Optional[np.ndarray] = None
    ) -> Tuple[float, bool]:
        """
        检验X和Y在给定Z条件下是否独立
        
        Returns:
            p_value: p值
            independent: 是否独立
        """
        raise NotImplementedError


class PartialCorrelationTest(ConditionalIndependenceTest):
    """
    偏相关检验
    
    通过计算偏相关系数来检验条件独立性
    """
    
    def test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: Optional[np.ndarray] = None
    ) -> Tuple[float, bool]:
        n = len(x)
        
        if z is None or len(z.shape) == 1 and z.size == 0:
            # 无条件相关
            r, p_value = stats.pearsonr(x, y)
        else:
            # 偏相关
            if len(z.shape) == 1:
                z = z.reshape(-1, 1)
            
            # 回归残差法计算偏相关
            # X对Z回归的残差
            z_with_const = np.column_stack([np.ones(n), z])
            try:
                beta_x = np.linalg.lstsq(z_with_const, x, rcond=None)[0]
                beta_y = np.linalg.lstsq(z_with_const, y, rcond=None)[0]
                res_x = x - z_with_const @ beta_x
                res_y = y - z_with_const @ beta_y
                r, p_value = stats.pearsonr(res_x, res_y)
            except:
                r, p_value = 0.0, 1.0
        
        return p_value, p_value > self.significance_level


class PCMCI:
    """
    PCMCI算法: Peter-Clark Momentary Conditional Independence
    
    专为高维、强自相关时间序列设计的因果发现算法
    
    两阶段过程：
    1. PC选择阶段：筛选潜在父节点，去除无关变量
    2. MCI检验阶段：瞬时条件独立性检验，去除虚假因果
    
    参考：Runge et al., "Detecting and quantifying causal associations 
          in large nonlinear time series datasets"
    """
    
    def __init__(
        self,
        max_lag: int = 5,
        significance_level: float = 0.05,
        max_conds_dim: int = 3
    ):
        """
        Args:
            max_lag: 最大滞后阶数
            significance_level: 显著性水平
            max_conds_dim: 条件集最大维度
        """
        self.max_lag = max_lag
        self.significance_level = significance_level
        self.max_conds_dim = max_conds_dim
        self.ci_test = PartialCorrelationTest(significance_level)
    
    def fit(self, data: np.ndarray) -> Dict:
        """
        执行PCMCI因果发现
        
        Args:
            data: [time_steps, num_variables] 时间序列数据
            
        Returns:
            results: 包含因果图、因果强度等信息的字典
        """
        T, N = data.shape
        
        # 构建滞后数据
        lagged_data = self._construct_lagged_data(data)
        
        # 阶段1: PC选择 - 筛选潜在父节点
        print("Phase 1: PC selection...")
        parents = self._pc_selection(lagged_data, N)
        
        # 阶段2: MCI检验 - 精确因果判断
        print("Phase 2: MCI test...")
        causal_graph, causal_strength = self._mci_test(lagged_data, parents, N)
        
        return {
            'causal_graph': causal_graph,
            'causal_strength': causal_strength,
            'parents': parents,
            'num_variables': N,
            'max_lag': self.max_lag
        }
    
    def _construct_lagged_data(self, data: np.ndarray) -> np.ndarray:
        """构建滞后数据矩阵"""
        T, N = data.shape
        
        # 为每个变量创建滞后版本
        lagged_vars = []
        for lag in range(self.max_lag + 1):
            if lag == 0:
                lagged_vars.append(data[self.max_lag:])
            else:
                lagged_vars.append(data[self.max_lag - lag:-lag])
        
        return np.concatenate(lagged_vars, axis=1)
    
    def _pc_selection(
        self,
        lagged_data: np.ndarray,
        num_vars: int
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        PC选择阶段：使用条件独立性检验筛选潜在父节点
        """
        T, total_vars = lagged_data.shape
        parents = {j: [] for j in range(num_vars)}
        
        # 对每个目标变量
        for j in range(num_vars):
            y = lagged_data[:, j]  # 当前时刻的目标变量
            
            # 检查所有可能的父节点（其他变量的滞后版本）
            candidates = []
            for i in range(num_vars):
                for lag in range(1, self.max_lag + 1):
                    idx = lag * num_vars + i
                    if idx < total_vars:
                        x = lagged_data[:, idx]
                        p_val, _ = self.ci_test.test(x, y)
                        if p_val < self.significance_level:
                            candidates.append((i, lag, p_val))
            
            # 按p值排序，选择最显著的
            candidates.sort(key=lambda x: x[2])
            
            # 条件独立性检验进行剪枝
            final_parents = []
            for (var, lag, _) in candidates[:self.max_conds_dim * 2]:
                idx = lag * num_vars + var
                x = lagged_data[:, idx]
                
                # 检查在已有父节点条件下是否仍然相关
                if len(final_parents) == 0:
                    final_parents.append((var, lag))
                else:
                    # 构建条件集
                    cond_indices = [l * num_vars + v for (v, l) in final_parents]
                    z = lagged_data[:, cond_indices]
                    
                    p_val, independent = self.ci_test.test(x, y, z)
                    if not independent:
                        final_parents.append((var, lag))
            
            parents[j] = final_parents
        
        return parents
    
    def _mci_test(
        self,
        lagged_data: np.ndarray,
        parents: Dict[int, List[Tuple[int, int]]],
        num_vars: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        MCI检验阶段：计算最终的因果关系强度
        """
        # 初始化因果图和强度矩阵
        # causal_graph[i, j, lag] = 1 表示变量i在lag滞后影响变量j
        causal_graph = np.zeros((num_vars, num_vars, self.max_lag + 1))
        causal_strength = np.zeros((num_vars, num_vars, self.max_lag + 1))
        
        T = lagged_data.shape[0]
        
        for j in range(num_vars):
            y = lagged_data[:, j]
            
            for (i, lag) in parents[j]:
                idx = lag * num_vars + i
                x = lagged_data[:, idx]
                
                # 构建条件集（MCI的关键：包含所有相关的滞后变量）
                cond_indices = []
                
                # 添加目标变量的自身滞后
                for l in range(1, self.max_lag + 1):
                    auto_idx = l * num_vars + j
                    if auto_idx < lagged_data.shape[1]:
                        cond_indices.append(auto_idx)
                
                # 添加其他父节点
                for (v, l) in parents[j]:
                    if (v, l) != (i, lag):
                        other_idx = l * num_vars + v
                        if other_idx not in cond_indices:
                            cond_indices.append(other_idx)
                
                # MCI检验
                if len(cond_indices) > 0:
                    z = lagged_data[:, cond_indices]
                    p_val, independent = self.ci_test.test(x, y, z)
                else:
                    p_val, independent = self.ci_test.test(x, y)
                
                if not independent:
                    causal_graph[i, j, lag] = 1
                    # 计算偏相关系数作为因果强度
                    if len(cond_indices) > 0:
                        z_with_const = np.column_stack([np.ones(T), lagged_data[:, cond_indices]])
                        beta_x = np.linalg.lstsq(z_with_const, x, rcond=None)[0]
                        beta_y = np.linalg.lstsq(z_with_const, y, rcond=None)[0]
                        res_x = x - z_with_const @ beta_x
                        res_y = y - z_with_const @ beta_y
                        r, _ = stats.pearsonr(res_x, res_y)
                    else:
                        r, _ = stats.pearsonr(x, y)
                    causal_strength[i, j, lag] = abs(r)
        
        return causal_graph, causal_strength


class LiNGAM:
    """
    LiNGAM: Linear Non-Gaussian Acyclic Model
    
    利用数据的非高斯性识别因果方向
    特别适用于风电数据（风速、湍流通常非高斯）
    
    核心思想：
    - 如果X->Y，且噪声非高斯，则Y对X回归的残差独立于X
    - 但X对Y回归的残差不独立于Y
    """
    
    def __init__(self, threshold: float = 0.1):
        """
        Args:
            threshold: 因果判断阈值
        """
        self.threshold = threshold
    
    def fit(self, data: np.ndarray) -> Dict:
        """
        执行LiNGAM因果发现
        
        Args:
            data: [samples, num_variables]
            
        Returns:
            results: 因果图和因果顺序
        """
        n_samples, n_vars = data.shape
        
        # 标准化数据
        data_std = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        
        # 使用ICA找到因果顺序
        causal_order = self._find_causal_order(data_std)
        
        # 计算因果系数
        causal_matrix = self._estimate_causal_matrix(data_std, causal_order)
        
        return {
            'causal_order': causal_order,
            'causal_matrix': causal_matrix,
            'adjacency_matrix': (np.abs(causal_matrix) > self.threshold).astype(int)
        }
    
    def _find_causal_order(self, data: np.ndarray) -> List[int]:
        """
        使用DirectLiNGAM算法找到因果顺序
        """
        n_vars = data.shape[1]
        remaining = list(range(n_vars))
        order = []
        residuals = data.copy()
        
        for _ in range(n_vars):
            # 找到当前的根节点（最外生的变量）
            root = self._find_root(residuals, remaining)
            order.append(root)
            remaining.remove(root)
            
            # 从其他变量中回归掉根节点的影响
            if len(remaining) > 0:
                x_root = residuals[:, root:root+1]
                for j in remaining:
                    y = residuals[:, j]
                    beta = np.linalg.lstsq(x_root, y, rcond=None)[0]
                    residuals[:, j] = y - x_root @ beta
        
        return order
    
    def _find_root(self, data: np.ndarray, candidates: List[int]) -> int:
        """
        找到当前的根节点（外生变量）
        
        使用独立性检验：如果X是根，则其他变量对X回归的残差应与X独立
        """
        if len(candidates) == 1:
            return candidates[0]
        
        best_score = -np.inf
        best_var = candidates[0]
        
        for i in candidates:
            score = 0
            x = data[:, i:i+1]
            
            for j in candidates:
                if i != j:
                    y = data[:, j]
                    # 回归
                    beta = np.linalg.lstsq(x, y, rcond=None)[0]
                    residual = y - x @ beta
                    
                    # 检验残差与x的独立性（使用非高斯性度量）
                    # 独立性越强，得分越高
                    ind_score = self._independence_score(x.flatten(), residual)
                    score += ind_score
            
            if score > best_score:
                best_score = score
                best_var = i
        
        return best_var
    
    def _independence_score(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        计算两个变量的独立性得分
        
        使用互信息的近似：独立时得分接近0
        """
        # 使用Hilbert-Schmidt Independence Criterion (HSIC)的简化版本
        n = len(x)
        
        # 标准化
        x = (x - x.mean()) / (x.std() + 1e-8)
        y = (y - y.mean()) / (y.std() + 1e-8)
        
        # 计算相关性（独立时相关性为0）
        corr = np.abs(np.corrcoef(x, y)[0, 1])
        
        # 非线性依赖检测
        x_sq = x ** 2
        y_sq = y ** 2
        corr_sq = np.abs(np.corrcoef(x_sq, y_sq)[0, 1])
        
        # 返回负的依赖性度量（越独立得分越高）
        return -corr - 0.5 * corr_sq
    
    def _estimate_causal_matrix(
        self,
        data: np.ndarray,
        causal_order: List[int]
    ) -> np.ndarray:
        """
        估计因果系数矩阵
        """
        n_vars = data.shape[1]
        B = np.zeros((n_vars, n_vars))
        
        for i, target in enumerate(causal_order[1:], 1):
            # target的父节点是在因果顺序中排在它前面的变量
            parents = causal_order[:i]
            
            if len(parents) > 0:
                X = data[:, parents]
                y = data[:, target]
                
                # 回归估计因果系数
                X_with_const = np.column_stack([np.ones(len(y)), X])
                beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                
                for j, parent in enumerate(parents):
                    B[parent, target] = beta[j + 1]
        
        return B


class DynamicWakeGraph:
    """
    动态尾流因果图谱
    
    结合PCMCI和物理约束，构建风电场的动态尾流拓扑
    
    应用：
    1. 无需物理坐标即可推断风机相对位置
    2. 故障传播路径追踪
    3. 尾流效应量化
    """
    
    def __init__(
        self,
        num_turbines: int,
        max_lag: int = 10,
        significance_level: float = 0.05
    ):
        self.num_turbines = num_turbines
        self.max_lag = max_lag
        self.pcmci = PCMCI(max_lag, significance_level)
        
        # 存储不同风向下的因果图
        self.causal_graphs = {}
    
    def build_graph(
        self,
        power_data: np.ndarray,
        wind_direction: float
    ) -> Dict:
        """
        为特定风向构建因果图
        
        Args:
            power_data: [time_steps, num_turbines] 功率时间序列
            wind_direction: 风向角度 (0-360)
            
        Returns:
            graph_info: 因果图信息
        """
        # 执行PCMCI
        results = self.pcmci.fit(power_data)
        
        # 提取主要因果关系（滞后1-3的累积）
        causal_strength = results['causal_strength']
        aggregated_strength = causal_strength[:, :, 1:4].sum(axis=2)
        
        # 构建边列表
        edges = []
        for i in range(self.num_turbines):
            for j in range(self.num_turbines):
                if i != j and aggregated_strength[i, j] > 0.1:
                    edges.append({
                        'source': i,
                        'target': j,
                        'strength': aggregated_strength[i, j],
                        'direction': 'upstream_to_downstream'
                    })
        
        # 存储结果
        wind_dir_key = int(wind_direction // 10) * 10  # 按10度分组
        self.causal_graphs[wind_dir_key] = {
            'edges': edges,
            'strength_matrix': aggregated_strength,
            'full_results': results
        }
        
        return self.causal_graphs[wind_dir_key]
    
    def get_upstream_turbines(
        self,
        turbine_id: int,
        wind_direction: float
    ) -> List[Tuple[int, float]]:
        """
        获取指定风机的上游风机列表
        
        Args:
            turbine_id: 目标风机ID
            wind_direction: 当前风向
            
        Returns:
            upstream: [(风机ID, 影响强度), ...]
        """
        wind_dir_key = int(wind_direction // 10) * 10
        
        if wind_dir_key not in self.causal_graphs:
            return []
        
        strength_matrix = self.causal_graphs[wind_dir_key]['strength_matrix']
        
        upstream = []
        for i in range(self.num_turbines):
            if strength_matrix[i, turbine_id] > 0.1:
                upstream.append((i, strength_matrix[i, turbine_id]))
        
        # 按影响强度排序
        upstream.sort(key=lambda x: -x[1])
        return upstream
    
    def trace_fault_propagation(
        self,
        fault_turbine: int,
        wind_direction: float
    ) -> List[Dict]:
        """
        追踪故障传播路径
        
        Args:
            fault_turbine: 故障风机ID
            wind_direction: 当前风向
            
        Returns:
            propagation_path: 故障传播路径
        """
        wind_dir_key = int(wind_direction // 10) * 10
        
        if wind_dir_key not in self.causal_graphs:
            return []
        
        strength_matrix = self.causal_graphs[wind_dir_key]['strength_matrix']
        
        # BFS遍历下游风机
        visited = {fault_turbine}
        path = []
        queue = [(fault_turbine, 0)]  # (turbine_id, depth)
        
        while queue:
            current, depth = queue.pop(0)
            
            # 找到当前风机影响的下游风机
            for j in range(self.num_turbines):
                if j not in visited and strength_matrix[current, j] > 0.1:
                    visited.add(j)
                    queue.append((j, depth + 1))
                    path.append({
                        'from': current,
                        'to': j,
                        'strength': strength_matrix[current, j],
                        'depth': depth + 1
                    })
        
        return path


class PhysicsConstrainedCausalDiscovery:
    """
    物理约束因果发现
    
    结合物理模型（Jensen尾流模型）和数据驱动方法
    
    流程：
    1. 根据物理模型生成骨架图（先验知识）
    2. 使用PCMCI在骨架图基础上进行因果发现
    3. 验证结果的物理一致性
    """
    
    def __init__(
        self,
        turbine_positions: np.ndarray,  # [num_turbines, 2] x,y坐标
        rotor_diameter: float = 126.0,  # 叶轮直径(m)
        wake_decay_constant: float = 0.04  # 尾流衰减常数
    ):
        self.positions = turbine_positions
        self.num_turbines = len(turbine_positions)
        self.rotor_diameter = rotor_diameter
        self.wake_decay = wake_decay_constant
    
    def generate_physics_skeleton(
        self,
        wind_direction: float,
        wind_speed: float = 10.0
    ) -> np.ndarray:
        """
        根据Jensen尾流模型生成物理骨架图
        
        Args:
            wind_direction: 风向（度，0=北，顺时针）
            wind_speed: 风速(m/s)
            
        Returns:
            skeleton: [num_turbines, num_turbines] 骨架邻接矩阵
        """
        # 转换风向为弧度
        theta = np.radians(wind_direction)
        
        # 风向单位向量（从上游指向下游）
        wind_vec = np.array([np.sin(theta), np.cos(theta)])
        
        skeleton = np.zeros((self.num_turbines, self.num_turbines))
        
        for i in range(self.num_turbines):
            for j in range(self.num_turbines):
                if i == j:
                    continue
                
                # 计算从i到j的向量
                vec_ij = self.positions[j] - self.positions[i]
                
                # 沿风向的距离（正值表示j在i的下游）
                downstream_dist = np.dot(vec_ij, wind_vec)
                
                # 垂直于风向的距离
                cross_dist = np.abs(np.cross(wind_vec, vec_ij))
                
                if downstream_dist > 0:  # j在i的下游
                    # Jensen尾流模型：尾流宽度随距离线性扩展
                    wake_radius = self.rotor_diameter / 2 + self.wake_decay * downstream_dist
                    
                    # 如果j在尾流区内
                    if cross_dist < wake_radius:
                        # 计算影响系数（基于距离衰减）
                        influence = 1.0 / (1 + self.wake_decay * downstream_dist / self.rotor_diameter)
                        skeleton[i, j] = influence
        
        return skeleton
    
    def discover_with_physics(
        self,
        power_data: np.ndarray,
        wind_direction: float,
        wind_speed: float = 10.0
    ) -> Dict:
        """
        结合物理先验的因果发现
        
        Args:
            power_data: [time_steps, num_turbines] 功率数据
            wind_direction: 风向
            wind_speed: 风速
            
        Returns:
            results: 因果发现结果
        """
        # 1. 生成物理骨架
        physics_skeleton = self.generate_physics_skeleton(wind_direction, wind_speed)
        
        # 2. 使用PCMCI进行数据驱动因果发现
        pcmci = PCMCI(max_lag=10, significance_level=0.05)
        data_results = pcmci.fit(power_data)
        
        # 3. 融合物理先验和数据驱动结果
        data_strength = data_results['causal_strength'][:, :, 1:4].sum(axis=2)
        
        # 物理一致性检验
        consistent_edges = []
        inconsistent_edges = []
        
        for i in range(self.num_turbines):
            for j in range(self.num_turbines):
                if i == j:
                    continue
                
                physics_edge = physics_skeleton[i, j] > 0.1
                data_edge = data_strength[i, j] > 0.1
                
                if physics_edge and data_edge:
                    # 物理和数据都支持
                    consistent_edges.append({
                        'source': i,
                        'target': j,
                        'physics_strength': physics_skeleton[i, j],
                        'data_strength': data_strength[i, j],
                        'status': 'confirmed'
                    })
                elif physics_edge and not data_edge:
                    # 物理预测有，但数据未发现
                    consistent_edges.append({
                        'source': i,
                        'target': j,
                        'physics_strength': physics_skeleton[i, j],
                        'data_strength': 0,
                        'status': 'physics_only'
                    })
                elif not physics_edge and data_edge:
                    # 数据发现但物理不支持（可能是复杂耦合）
                    inconsistent_edges.append({
                        'source': i,
                        'target': j,
                        'physics_strength': 0,
                        'data_strength': data_strength[i, j],
                        'status': 'data_only_needs_verification'
                    })
        
        return {
            'physics_skeleton': physics_skeleton,
            'data_causal_graph': data_strength,
            'consistent_edges': consistent_edges,
            'inconsistent_edges': inconsistent_edges,
            'full_pcmci_results': data_results
        }


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("Causal Discovery Module for Wind Power Prediction")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 生成模拟数据
    print("\n1. Testing PCMCI:")
    T = 500  # 时间步
    N = 5    # 变量数（风机数）
    
    # 模拟带有因果关系的数据
    # X0 -> X1 (lag 1), X1 -> X2 (lag 2)
    data = np.random.randn(T, N)
    for t in range(2, T):
        data[t, 1] += 0.6 * data[t-1, 0]  # X0 -> X1
        data[t, 2] += 0.5 * data[t-2, 1]  # X1 -> X2
        data[t, 3] += 0.4 * data[t-1, 0] + 0.3 * data[t-1, 1]  # X0,X1 -> X3
    
    pcmci = PCMCI(max_lag=5, significance_level=0.05)
    results = pcmci.fit(data)
    
    print(f"   Discovered parents:")
    for var, parents in results['parents'].items():
        if parents:
            print(f"   Variable {var} <- {parents}")
    
    # 测试LiNGAM
    print("\n2. Testing LiNGAM:")
    # 生成简单的结构数据
    n = 500
    X0 = np.random.laplace(size=n)  # 非高斯
    X1 = 0.8 * X0 + np.random.laplace(size=n) * 0.3
    X2 = 0.6 * X1 + np.random.laplace(size=n) * 0.3
    simple_data = np.column_stack([X0, X1, X2])
    
    lingam = LiNGAM(threshold=0.1)
    lingam_results = lingam.fit(simple_data)
    print(f"   Causal order: {lingam_results['causal_order']}")
    print(f"   Expected order: [0, 1, 2]")
    
    # 测试动态尾流图
    print("\n3. Testing Dynamic Wake Graph:")
    wake_graph = DynamicWakeGraph(num_turbines=5, max_lag=5)
    graph_info = wake_graph.build_graph(data, wind_direction=270)
    print(f"   Found {len(graph_info['edges'])} causal edges")
    
    # 测试物理约束因果发现
    print("\n4. Testing Physics-Constrained Causal Discovery:")
    # 模拟风机位置（简单直线排列）
    positions = np.array([
        [0, 0],
        [500, 0],
        [1000, 0],
        [1500, 0],
        [2000, 0]
    ], dtype=float)
    
    physics_cd = PhysicsConstrainedCausalDiscovery(
        turbine_positions=positions,
        rotor_diameter=126
    )
    
    # 西风（270度），风从左向右吹
    physics_results = physics_cd.discover_with_physics(
        power_data=data,
        wind_direction=270
    )
    
    print(f"   Consistent edges: {len(physics_results['consistent_edges'])}")
    print(f"   Inconsistent edges: {len(physics_results['inconsistent_edges'])}")
    
    print("\n" + "=" * 60)
    print("Causal Discovery Module Test Complete!")
    print("=" * 60)
