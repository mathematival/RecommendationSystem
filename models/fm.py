import numpy as np
import pandas as pd
from typing import Dict
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

from framework import BaseRecommender

class FMRecommender(BaseRecommender):
    """
    Factorization Machines (FM) 模型
    
    FM模型能够自动学习特征之间的交互关系，通过隐向量内积捕捉二阶特征交叉，
    相比于传统的线性模型（如LR）具有更强的特征组合能力，尤其适用于稀疏数据场景。
    """
    
    def __init__(self, config, num_factors=10, learning_rate=0.002, 
                 regularization=0.005, num_epochs=20, batch_size=1024):
        """初始化FM模型"""
        super().__init__(config)
        self.num_factors = num_factors       # 隐向量维度
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
        item_features[:, 1] = np.log1p(item_features[:, 1])
        item_features = self.item_feature_scaler.fit_transform(item_features)
        
        # 合并所有特征
        X_array = np.hstack([X_ids, user_features, item_features]).astype(np.float32)
        
        return X_array, y
    
    def fm_forward(self, X_batch):
        """向量化的FM前向传播计算 - 整批处理"""
        # 线性项：w0 + w^T x
        linear_term = self.w0 + torch.matmul(X_batch, self.w)
        
        # 交互项计算：1/2 * sum_f [ (sum_i v_i,f*x_i)^2 - sum_i (v_i,f*x_i)^2 ]
        # 第一项: (sum_i v_i,f * x_i)^2 对所有f求和
        sum_vx = torch.matmul(X_batch.unsqueeze(1), self.v)
        sum_vx_square = torch.pow(sum_vx, 2).squeeze(1)
        
        # 第二项: sum_i (v_i,f * x_i)^2 对所有f求和
        X_batch_square = torch.pow(X_batch, 2)
        v_square = torch.pow(self.v, 2)
        vx_square_sum = torch.matmul(X_batch_square, v_square)
        
        # 1/2 * (sum_vx_square - vx_square_sum)
        interaction_term = 0.5 * torch.sum(sum_vx_square - vx_square_sum, dim=1)
        
        # 模型输出 = 线性项 + 交互项
        return linear_term + interaction_term
    
    def fit(self, train_users: Dict, train_items: Dict) -> None:
        """训练FM模型"""
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
        self.w0 = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=self.device)
        self.w = torch.tensor(
            np.random.normal(0, 0.01, self.num_features).astype(np.float32), 
            requires_grad=True, 
            device=self.device
        )
        self.v = torch.tensor(
            np.random.normal(0, 0.01, (self.num_features, self.num_factors)).astype(np.float32), 
            requires_grad=True, 
            device=self.device
        )
        
        # 创建Adam优化器
        optimizer = torch.optim.Adam([self.w0, self.w, self.v], lr=self.learning_rate)
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        
        # 创建数据集和数据加载器
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_normalized, dtype=torch.float32, device=self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 训练模型
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            
            # 批量训练
            for X_batch, y_batch in data_loader:
                # 清零梯度
                optimizer.zero_grad()
                
                # 前向传播 - 一次计算整个批次
                pred = self.fm_forward(X_batch)
                
                # 计算损失 - 使用均方误差
                loss = torch.mean((pred - y_batch) ** 2)
                
                # 添加L2正则化
                l2_reg = self.regularization * (torch.sum(self.w ** 2) + torch.sum(self.v ** 2))
                loss += l2_reg
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_([self.w0, self.w, self.v], max_norm=1.0)
                
                # 更新参数
                optimizer.step()
                
                epoch_loss += loss.item() * X_batch.size(0)
            
            # 计算平均损失
            avg_loss = epoch_loss / len(dataset)
            
            # 更新学习率
            scheduler.step(avg_loss)
            
            # 打印训练进度
            if (epoch + 1) % max(1, self.num_epochs // 10) == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.6f}")
        
        # 保存最终训练好的参数 - 转换为NumPy数组
        self.w0 = self.w0.item()
        self.w = self.w.detach().cpu().numpy()
        self.v = self.v.detach().cpu().numpy()
        
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
    
    def _fm_prediction(self, x):
        """
        FM模型预测函数 - 使用NumPy实现FM计算
        
        优化公式: y = w0 + ∑_i w_i * x_i + 0.5 * ∑_f [(∑_i v_{i,f} * x_i)^2 - ∑_i (v_{i,f} * x_i)^2]
        """
        # 线性项
        linear_term = self.w0 + np.dot(self.w, x)
        
        # 使用向量化操作计算交互项
        # 第一项: (sum_i v_i,f*x_i)^2
        sum_vx = np.zeros(self.num_factors, dtype=np.float32)
        for f in range(self.num_factors):
            sum_vx[f] = np.dot(self.v[:, f], x)
        sum_square = np.sum(sum_vx ** 2)
        
        # 第二项: sum_i (v_i,f*x_i)^2
        square_sum = 0
        for i in range(self.num_features):
            if x[i] != 0:  # 稀疏优化
                square_sum += np.sum((self.v[i] * x[i]) ** 2)
        
        # 计算交互项: 1/2 * (sum_square - square_sum)
        interaction_term = 0.5 * (sum_square - square_sum)
        
        # 返回最终预测值
        return float(linear_term + interaction_term)
    
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
