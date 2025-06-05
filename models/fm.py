import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch

from framework import BaseRecommender

class FMRecommender(BaseRecommender):
    """
    Factorization Machines (FM) 模型
    
    FM模型能够自动学习特征之间的交互关系，通过隐向量内积捕捉二阶特征交叉，
    相比于传统的线性模型（如LR）具有更强的特征组合能力，尤其适用于稀疏数据场景。
    """
    
    def __init__(self, config, num_factors=4, learning_rate=0.001, 
                 regularization=0.01, num_epochs=10, batch_size=256):
        """初始化FM模型"""
        super().__init__(config)
        self.num_factors = num_factors       # 隐向量维度（降低以提高稳定性）
        self.learning_rate = learning_rate   # 学习率
        self.regularization = regularization # 正则化系数
        self.num_epochs = num_epochs         # 迭代次数
        self.batch_size = batch_size         # 批处理大小
        
        # FM参数
        self.w0 = 0.          # 全局偏置
        self.w = None         # 一阶特征权重
        self.v = None         # 特征隐向量
        
        # 特征处理
        self.user_feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.item_feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.id_scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
        self.feature_names = []
        self.num_features = 0
        
        self.user_stats = {}
        self.item_stats = {}
        self.y_mean = None
        self.y_std = None
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _create_feature_matrix(self, users_data, items_data):
        """创建训练特征矩阵，并应用适当的特征工程和缩放"""
        print("创建FM特征矩阵...")
        
        # 计算用户和物品的统计特征
        for user_id, ratings in users_data.items():
            if ratings:
                user_ratings = [r for _, r in ratings]
                self.user_stats[user_id] = {
                    'mean': np.mean(user_ratings),
                    'count': len(user_ratings),
                    'std': np.std(user_ratings) if len(user_ratings) > 1 else 0
                }
            else:
                self.user_stats[user_id] = {'mean': self.global_mean, 'count': 0, 'std': 0}
        
        for item_id, users in items_data.items():
            if users:
                item_ratings = [r for _, r in users]
                self.item_stats[item_id] = {
                    'mean': np.mean(item_ratings),
                    'count': len(item_ratings),
                    'std': np.std(item_ratings) if len(item_ratings) > 1 else 0
                }
            else:
                self.item_stats[item_id] = {'mean': self.global_mean, 'count': 0, 'std': 0}
        
        # 创建特征矩阵和目标向量
        data = []
        
        # 收集训练数据
        for user_id, ratings in users_data.items():
            for item_id, rating in ratings:
                # 提取特征
                user_mean = self.user_stats[user_id]['mean']
                user_count = self.user_stats[user_id]['count']
                user_std = self.user_stats[user_id]['std']
                
                item_mean = self.item_stats.get(item_id, {'mean': self.global_mean})['mean']
                item_count = self.item_stats.get(item_id, {'count': 0})['count']
                item_std = self.item_stats.get(item_id, {'std': 0})['std']
                
                # 打包特征
                data.append([
                    user_id, item_id, 
                    user_mean, user_count, user_std,
                    item_mean, item_count, item_std,
                    rating
                ])
        
        # 创建DataFrame
        df = pd.DataFrame(data, columns=[
            'user_id', 'item_id', 
            'user_mean', 'user_count', 'user_std',
            'item_mean', 'item_count', 'item_std',
            'rating'
        ])
        
        # 特征和目标分离
        X = df.drop('rating', axis=1)
        y = df['rating'].values
        
        # 记录特征名称
        self.feature_names = list(X.columns)
        self.num_features = len(self.feature_names)
        
        # 对ID特征进行缩放 - 降低数值范围
        X_ids = X[['user_id', 'item_id']].values.astype(float)
        X_ids = self.id_scaler.fit_transform(X_ids)
        
        # 对用户特征进行缩放
        user_features = X[['user_mean', 'user_count', 'user_std']].values
        # 对count特征进行对数变换，减少数值大小差异
        user_features[:, 1] = np.log1p(user_features[:, 1])
        user_features = self.user_feature_scaler.fit_transform(user_features)
        
        # 对物品特征进行缩放
        item_features = X[['item_mean', 'item_count', 'item_std']].values
        # 对count特征进行对数变换，减少数值大小差异
        item_features[:, 1] = np.log1p(item_features[:, 1])
        item_features = self.item_feature_scaler.fit_transform(item_features)
        
        # 合并所有特征
        X_array = np.hstack([X_ids, user_features, item_features]).astype(np.float32)
        
        return X_array, y
    
    def _fm_prediction(self, x):
        """
        FM模型预测函数 - 使用NumPy实现FM计算
        
        优化公式: y = w0 + ∑_i w_i * x_i + 0.5 * ∑_f [(∑_i v_{i,f} * x_i)^2 - ∑_i (v_{i,f} * x_i)^2]
        """
        # 线性项
        linear_term = self.w0 + np.dot(self.w, x)
        
        # 计算特征交互项，使用FM优化公式
        # 非零特征优化
        non_zero_indices = np.where(x != 0)[0]
        
        # 初始化累加项
        sum_square_terms = np.zeros(self.num_factors, dtype=np.float32)
        square_sum_terms = np.zeros(self.num_factors, dtype=np.float32)
        
        # 计算两个累加项
        for f in range(self.num_factors):
            # 只计算非零特征
            vx = self.v[non_zero_indices, f] * x[non_zero_indices]
            sum_square_terms[f] = np.sum(vx) ** 2
            square_sum_terms[f] = np.sum(vx ** 2)
        
        # 计算交互项: 1/2 * ∑_f [(∑_i v_i,f*x_i)^2 - ∑_i (v_i,f*x_i)^2]
        interaction_term = 0.5 * np.sum(sum_square_terms - square_sum_terms)
        
        # 返回最终预测值
        return float(linear_term + interaction_term)
    
    def fit(self, train_users: Dict, train_items: Dict) -> None:
        """训练FM模型，使用PyTorch优化器"""
        print("训练FM模型...")
        super().fit(train_users, train_items)
        
        # 创建特征矩阵
        X, y = self._create_feature_matrix(train_users, train_items)
        
        if len(X) == 0:
            print("警告：特征矩阵为空，无法训练模型")
            return
            
        print(f"特征矩阵形状: {X.shape}, 目标向量长度: {len(y)}")
        
        # 标准化目标值，使其接近0均值、单位方差
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        y_normalized = (y - self.y_mean) / self.y_std
        
        # 初始化FM模型参数
        torch.manual_seed(42)  # 设置随机种子确保可复现
        self.w0 = 0.0
        self.w = np.random.normal(0, 0.01, self.num_features).astype(np.float32)
        self.v = np.random.normal(0, 0.01, (self.num_features, self.num_factors)).astype(np.float32)
        
        # 将模型参数转换为PyTorch张量
        w0_tensor = torch.tensor(self.w0, dtype=torch.float32, requires_grad=True, device=self.device)
        w_tensor = torch.tensor(self.w, dtype=torch.float32, requires_grad=True, device=self.device)
        v_tensor = torch.tensor(self.v, dtype=torch.float32, requires_grad=True, device=self.device)
        
        # 创建Adam优化器
        optimizer = torch.optim.Adam([w0_tensor, w_tensor, v_tensor], lr=self.learning_rate)
        
        # 将数据转换为PyTorch张量
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_normalized, dtype=torch.float32, device=self.device)
        
        # 训练模型
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        for epoch in range(self.num_epochs):
            # 打乱数据顺序
            np.random.shuffle(indices)
            X_shuffled = X_tensor[torch.tensor(indices, device=self.device)]
            y_shuffled = y_tensor[torch.tensor(indices, device=self.device)]
            
            # 批量训练
            epoch_loss = 0.0
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # 清零梯度
                optimizer.zero_grad()
                
                # 前向传播
                batch_preds = []
                for i in range(len(X_batch)):
                    x = X_batch[i]
                    # 线性项
                    linear_term = w0_tensor + torch.sum(w_tensor * x)
                    
                    # 交互项
                    non_zero_indices = torch.nonzero(x).squeeze()
                    vx = torch.index_select(v_tensor, 0, non_zero_indices) * torch.index_select(x, 0, non_zero_indices).unsqueeze(1)
                    
                    sum_square_term = torch.sum(vx, dim=0) ** 2
                    square_sum_term = torch.sum(vx ** 2, dim=0)
                    
                    interaction_term = 0.5 * torch.sum(sum_square_term - square_sum_term)
                    
                    pred = linear_term + interaction_term
                    batch_preds.append(pred)
                
                batch_preds = torch.stack(batch_preds)
                
                # 计算损失
                loss = torch.mean((batch_preds - y_batch) ** 2)
                
                # 添加L2正则化
                l2_reg = self.regularization * (torch.sum(w_tensor ** 2) + torch.sum(v_tensor ** 2))
                loss += l2_reg
                
                # 反向传播和优化
                loss.backward()
                
                # 应用梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_([w0_tensor, w_tensor, v_tensor], max_norm=1.0)
                
                optimizer.step()
                
                # 累计批次损失
                epoch_loss += loss.item() * len(X_batch)
            
            # 计算平均损失
            avg_loss = epoch_loss / n_samples
            
            # 打印训练进度
            if (epoch + 1) % max(1, self.num_epochs // 10) == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.6f}")
        
        # 保存最终训练好的参数
        self.w0 = w0_tensor.item()
        self.w = w_tensor.detach().cpu().numpy()
        self.v = v_tensor.detach().cpu().numpy()
        
        print("FM模型训练完成")
    
    def _extract_features(self, user_id: int, item_id: int) -> np.ndarray:
        """为单个用户-物品对提取特征"""
        try:
            # 获取用户和物品统计特征
            user_mean = self.user_stats.get(user_id, {'mean': self.global_mean})['mean']
            user_count = self.user_stats.get(user_id, {'count': 0})['count']
            user_std = self.user_stats.get(user_id, {'std': 0})['std']
            
            item_mean = self.item_stats.get(item_id, {'mean': self.global_mean})['mean']
            item_count = self.item_stats.get(item_id, {'count': 0})['count']
            item_std = self.item_stats.get(item_id, {'std': 0})['std']
            
            # 构建特征向量
            user_item_ids = np.array([[user_id, item_id]], dtype=float)
            user_item_ids = self.id_scaler.transform(user_item_ids).flatten()
            
            # 用户特征处理
            user_features = np.array([[user_mean, np.log1p(user_count), user_std]], dtype=float)
            user_features = self.user_feature_scaler.transform(user_features).flatten()
            
            # 物品特征处理
            item_features = np.array([[item_mean, np.log1p(item_count), item_std]], dtype=float)
            item_features = self.item_feature_scaler.transform(item_features).flatten()
            
            # 合并所有特征
            features = np.concatenate([user_item_ids, user_features, item_features]).astype(np.float32)
            
            return features
        except Exception as e:
            print(f"特征提取错误: {e}")
            # 返回零向量作为回退策略
            return np.zeros(self.num_features, dtype=np.float32)
    
    def predict(self, user_id: int, item_id: int) -> float:
        """预测用户对物品的评分"""
        if self.w is None or self.v is None:
            # 如果模型未训练，则回退到基线方法
            return self.global_mean
        
        try:
            # 提取特征
            x = self._extract_features(user_id, item_id)
            
            # FM预测 - 得到归一化值
            normalized_pred = self._fm_prediction(x)
            
            # 反标准化预测值
            pred = normalized_pred * self.y_std + self.y_mean
            
            # 确保预测值在有效范围内
            return max(self.config.rating_min, min(self.config.rating_max, pred))
        except Exception as e:
            # 错误处理
            print(f"预测错误: {e}")
            # 回退到基线预测
            return self.item_means.get(item_id, self.global_mean)
