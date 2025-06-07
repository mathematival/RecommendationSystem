import numpy as np
import random
import math
from typing import Dict, List, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from framework import BaseRecommender, ExperimentConfig


class SVDPlusPlusNet(nn.Module):
    """
    PyTorch版本的SVD++神经网络模型
    
    使用PyTorch构建的矩阵分解模型，包含用户和物品的嵌入层、偏置项以及隐式反馈
    SVD++公式: r_ui = μ + b_u + b_i + q_i^T * (p_u + |N(u)|^(-0.5) * Σ_{j∈N(u)} y_j)
    """
    
    def __init__(self, num_users: int, num_items: int, latent_factors: int = 50, 
                 user_id_map: Dict = None, item_id_map: Dict = None,
                 user_items: Dict = None):
        super(SVDPlusPlusNet, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.latent_factors = latent_factors
        self.user_id_map = user_id_map or {}
        self.item_id_map = item_id_map or {}
        self.user_items = user_items or {}
        
        # 用户和物品嵌入层
        self.user_embedding = nn.Embedding(num_users, latent_factors)
        self.item_embedding = nn.Embedding(num_items, latent_factors)
        
        # SVD++特有：隐式反馈嵌入层
        self.implicit_feedback = nn.Embedding(num_items, latent_factors)
        
        # 用户和物品偏置
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        
        # 全局偏置
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        # 使用标准正态分布初始化嵌入层
        init_std = 0.1
        nn.init.normal_(self.user_embedding.weight, std=init_std)
        nn.init.normal_(self.item_embedding.weight, std=init_std)
        nn.init.normal_(self.implicit_feedback.weight, std=init_std)
        
        # 偏置初始化为0
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor, 
                user_items_list: List[List[int]] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            user_ids: 用户ID张量 [batch_size]
            item_ids: 物品ID张量 [batch_size]
            user_items_list: 每个用户的历史交互物品列表
            
        Returns:
            predictions: 预测评分 [batch_size]
        """
        # 获取嵌入向量
        user_embed = self.user_embedding(user_ids)  # [batch_size, latent_factors]
        item_embed = self.item_embedding(item_ids)  # [batch_size, latent_factors]
        
        # 获取偏置
        user_bias = self.user_bias(user_ids).squeeze()  # [batch_size]
        item_bias = self.item_bias(item_ids).squeeze()  # [batch_size]
        
        # SVD++核心：计算隐式反馈
        if user_items_list is not None:
            device = user_embed.device
            implicit_sum = torch.zeros_like(user_embed)  # [batch_size, latent_factors]
            
            for i, user_item_list in enumerate(user_items_list):
                if user_item_list:
                    # 获取用户历史交互物品的隐式反馈向量
                    item_indices = torch.tensor(user_item_list, dtype=torch.long, device=device)
                    implicit_vectors = self.implicit_feedback(item_indices)  # [num_items, latent_factors]
                    
                    # 计算归一化系数 |N(u)|^(-0.5)
                    norm_factor = 1.0 / math.sqrt(len(user_item_list))
                    
                    # 累加隐式反馈向量
                    implicit_sum[i] = norm_factor * implicit_vectors.sum(dim=0)
            
            # 增强用户嵌入：p_u + |N(u)|^(-0.5) * Σ_{j∈N(u)} y_j
            enhanced_user_embed = user_embed + implicit_sum
        else:
            enhanced_user_embed = user_embed
        
        # 计算点积
        dot_product = (enhanced_user_embed * item_embed).sum(dim=1)  # [batch_size]
          # 计算最终预测：μ + b_u + b_i + q_i^T * (p_u + |N(u)|^(-0.5) * Σ_{j∈N(u)} y_j)
        predictions = self.global_bias + user_bias + item_bias + dot_product
        
        return predictions
    
    def predict_single(self, user_id: int, item_id: int) -> float:
        """预测单个用户-物品对的评分"""
        self.eval()
        with torch.no_grad():
            # 映射ID
            user_idx = self.user_id_map.get(user_id, 0)
            item_idx = self.item_id_map.get(item_id, 0)
            
            # 创建tensors并移动到正确的设备
            device = next(self.parameters()).device
            user_tensor = torch.tensor([user_idx], dtype=torch.long, device=device)
            item_tensor = torch.tensor([item_idx], dtype=torch.long, device=device)
            
            # 获取用户历史交互物品列表
            user_items_list = None
            if user_id in self.user_items:
                user_item_ids = list(self.user_items[user_id])
                # 将原始物品ID转换为索引
                user_item_indices = [self.item_id_map.get(iid, 0) for iid in user_item_ids]
                user_items_list = [user_item_indices]
            
            prediction = self.forward(user_tensor, item_tensor, user_items_list)
            return prediction.item()


class SVDPlusPlusRecommender(BaseRecommender):
    """
    使用PyTorch实现的SVD++推荐算法
    
    采用深度学习框架构建，使用均方误差损失函数和Adam优化器，
    并引入隐式反馈机制提升推荐效果
    """
    
    def __init__(self, config: ExperimentConfig,
                 latent_factors: int = 100,
                 learning_rate: float = 0.001,
                 regularization: float = 0.01,
                 max_epochs: int = 1000,
                 batch_size: int = 1024,
                 early_stopping_patience: int = 10,
                 device: str = None):
        """初始化PyTorch SVD++推荐器"""
        super().__init__(config)
        
        self.latent_factors = latent_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        
        # 设备选择
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 模型相关
        self.model = None
        self.user_id_map = {}
        self.item_id_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        
        # 训练记录
        self.training_losses = []
        self.validation_losses = []
        
        # 设置随机种子
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)
        
    def _create_id_mappings(self):
        """创建用户和物品ID到连续索引的映射"""
        # 创建用户映射
        unique_users = sorted(self.all_users)
        self.user_id_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.reverse_user_map = {idx: user_id for user_id, idx in self.user_id_map.items()}
        
        # 创建物品映射
        unique_items = sorted(self.all_items)
        self.item_id_map = {item_id: idx for idx, item_id in enumerate(unique_items)}
        self.reverse_item_map = {idx: item_id for item_id, idx in self.item_id_map.items()}
        
    def _prepare_training_data(self):
        """准备训练数据"""
        user_ids = []
        item_ids = []
        ratings = []
        
        for user_id, user_ratings in self.train_users.items():
            for item_id, rating in user_ratings:
                user_ids.append(self.user_id_map[user_id])
                item_ids.append(self.item_id_map[item_id])
                ratings.append(rating)
        
        return (torch.tensor(user_ids, dtype=torch.long),
                torch.tensor(item_ids, dtype=torch.long),
                torch.tensor(ratings, dtype=torch.float32))
    
    def _create_data_loader(self, user_ids, item_ids, ratings, shuffle=True):
        """创建数据加载器"""
        dataset = torch.utils.data.TensorDataset(user_ids, item_ids, ratings)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle
        )
    
    def _create_user_items_dict(self):
        """创建用户历史交互物品字典，用于隐式反馈"""
        user_items = {}
        for user_id, ratings in self.train_users.items():
            user_items[user_id] = {item_id for item_id, _ in ratings}
        return user_items
    
    def fit(self, train_users: Dict, train_items: Dict) -> None:
        """训练PyTorch SVD++模型"""
        super().fit(train_users, train_items)
        
        print(f"开始训练PyTorch SVD++模型...")
        print(f"设备: {self.device}")
        print(f"隐向量维度: {self.latent_factors}")
        print(f"学习率: {self.learning_rate}")
        print(f"正则化系数: {self.regularization}")
        print(f"最大轮数: {self.max_epochs}")
        print(f"批次大小: {self.batch_size}")
        
        # 创建ID映射
        self._create_id_mappings()
        
        # 创建模型
        num_users = len(self.user_id_map)
        num_items = len(self.item_id_map)
        self.model = SVDPlusPlusNet(
            num_users=num_users,
            num_items=num_items,
            latent_factors=self.latent_factors,
            user_id_map=self.user_id_map,
            item_id_map=self.item_id_map,
            user_items=self._create_user_items_dict()
        ).to(self.device)
        
        print(f"用户数量: {num_users}, 物品数量: {num_items}")
        
        # 准备训练数据
        user_ids, item_ids, ratings = self._prepare_training_data()
        print(f"训练样本数量: {len(user_ids)}")
          # 设置全局偏置为全局平均值
        with torch.no_grad():
            self.model.global_bias.data = torch.tensor([self.global_mean], dtype=torch.float32, device=self.device)
        
        # 创建数据加载器
        train_loader = self._create_data_loader(user_ids, item_ids, ratings)
        
        # 设置损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.regularization)
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=5
        )
        
        # 早停机制
        best_loss = float('inf')
        patience_counter = 0        # 训练循环
        self.model.train()
        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_users, batch_items, batch_ratings in train_loader:
                # 移动到设备
                batch_users = batch_users.to(self.device)
                batch_items = batch_items.to(self.device)
                batch_ratings = batch_ratings.to(self.device)
                
                # 为SVD++准备用户历史交互物品列表
                user_items_list = []
                for user_idx in batch_users.cpu().numpy():
                    # 从索引映射回原始用户ID
                    original_user_id = self.reverse_user_map[user_idx]
                    if original_user_id in self.model.user_items:
                        # 获取用户历史交互物品，并转换为索引
                        user_item_ids = list(self.model.user_items[original_user_id])
                        user_item_indices = [self.item_id_map.get(iid, 0) for iid in user_item_ids]
                        user_items_list.append(user_item_indices)
                    else:
                        user_items_list.append([])
                
                # 前向传播
                predictions = self.model(batch_users, batch_items, user_items_list)
                
                # 计算损失
                loss = criterion(predictions, batch_ratings)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # 计算平均损失
            avg_loss = epoch_loss / num_batches
            self.training_losses.append(avg_loss)
            
            # 学习率调度
            scheduler.step(avg_loss)
            
            # 早停检查
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_patience:
                print(f"早停触发，在第 {epoch+1} 轮停止训练")
                break
            
            # 每轮都显示训练进度
            print(f"Epoch {epoch+1}/{self.max_epochs}, 平均损失: {avg_loss:.6f}, 学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        print(f"\nPyTorch SVD++训练完成！最终损失: {self.training_losses[-1]:.6f}")
        
        # 设置为评估模式
        self.model.eval()
    
    def predict(self, user_id: int, item_id: int) -> float:
        """预测用户对物品的评分"""
        if self.model is None:
            return self.global_mean
            
        # 只负责已知用户和物品的预测
        # 冷启动情况将由父类的predict_for_user处理
        prediction = self.model.predict_single(user_id, item_id)
        
        # 不需要进行范围限制，因为父类的predict_for_user会做这件事
        return prediction
