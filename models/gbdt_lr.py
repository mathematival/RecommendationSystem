import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Any
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

from framework import BaseRecommender

class GBDTLRRecommender(BaseRecommender):
    """
    GBDT+LR混合模型
    
    该模型使用GBDT自动进行特征筛选和组合，生成新的离散特征向量，
    再把该特征向量当做LR模型的输入，来产生最后的预测结果。
    """
    
    def __init__(self, config, num_trees=100, max_depth=6, learning_rate=0.05):
        """
        初始化GBDT+LR模型
        
        Parameters:
        -----------
        config : ExperimentConfig
            实验配置对象
        num_trees : int, default=100
            GBDT模型中树的数量
        max_depth : int, default=6
            GBDT模型中树的最大深度
        learning_rate : float, default=0.05
            GBDT模型的学习率
        """
        super().__init__(config)
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.gbdt_model = None
        self.lr_model = None
        self.one_hot_encoder = None
        
        # 存储用户和物品的映射信息
        self.user_index_map = {}  # 用户ID -> 索引
        self.item_index_map = {}  # 物品ID -> 索引
        self.features = None
        self.user_stats = {}
        self.item_stats = {}
        self.user_count = 0
        self.item_count = 0
    
    def _create_feature_matrix(self, users_data, items_data):
        """
        创建训练特征矩阵
        
        Parameters:
        -----------
        users_data : Dict
            用户-物品评分数据
        items_data : Dict
            物品-用户评分数据
            
        Returns:
        --------
        X : numpy.ndarray
            特征矩阵
        y : numpy.ndarray
            目标评分值
        """
        print("创建特征矩阵...")
        
        # 1. 创建用户和物品的索引映射
        self.user_index_map = {user_id: idx for idx, user_id in enumerate(sorted(users_data.keys()))}
        self.item_index_map = {item_id: idx for idx, item_id in enumerate(sorted(items_data.keys()))}
        self.user_count = len(self.user_index_map)
        self.item_count = len(self.item_index_map)
        
        # 2. 计算用户和物品的统计特征
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
        
        # 3. 创建特征矩阵和目标向量
        data = []
        
        # 收集数据
        for user_id, ratings in users_data.items():
            for item_id, rating in ratings:
                # 基本特征
                user_idx = self.user_index_map[user_id]
                item_idx = self.item_index_map.get(item_id, -1)
                
                if item_idx == -1:
                    continue
                
                # 添加统计特征
                user_mean = self.user_stats[user_id]['mean']
                user_count = self.user_stats[user_id]['count']
                user_std = self.user_stats[user_id]['std']
                
                item_mean = self.item_stats[item_id]['mean']
                item_count = self.item_stats[item_id]['count']
                item_std = self.item_stats[item_id]['std']
                
                # 每一行是一个(user_id, item_id, user_mean, user_count, user_std, item_mean, item_count, item_std, rating)
                data.append([user_id, item_id, user_mean, user_count, user_std, 
                             item_mean, item_count, item_std, rating])
        
        # 创建DataFrame
        df = pd.DataFrame(data, columns=['user_id', 'item_id', 'user_mean', 'user_count', 'user_std', 
                                        'item_mean', 'item_count', 'item_std', 'rating'])
        
        # 特征和目标分离
        X = df.drop('rating', axis=1)
        y = df['rating'].values
        
        self.features = X
        
        return X, y
        
    def fit(self, train_users: Dict, train_items: Dict) -> None:
        """
        训练GBDT+LR模型
        
        Parameters:
        -----------
        train_users : Dict
            用户-物品评分数据
        train_items : Dict
            物品-用户评分数据
        """
        print("训练GBDT+LR模型...")
        super().fit(train_users, train_items)
        
        # 1. 创建特征矩阵
        X, y = self._create_feature_matrix(train_users, train_items)
        
        if len(X) == 0:
            print("警告：特征矩阵为空，无法训练模型")
            return
            
        print(f"特征矩阵形状: {X.shape}, 目标向量长度: {len(y)}")
        
        # 2. 训练GBDT模型
        print("训练GBDT模型...")
        self.gbdt_model = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=self.num_trees,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=31,
            verbose=-1
        )
        
        self.gbdt_model.fit(X, y)
        
        # 3. 获取叶子节点索引，进行特征转换
        print("特征转换...")
        leaf_indices = self.gbdt_model.predict(X, pred_leaf=True)
        
        # 4. 独热编码GBDT输出的叶子节点索引
        print("独热编码...")
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        gbdt_features = self.one_hot_encoder.fit_transform(leaf_indices)
        
        # 将原始特征和GBDT生成的特征结合
        print("合并特征...")
        # 将X转换为numpy数组
        X_array = X.values
        
        # 5. 训练线性回归模型
        print("训练线性回归模型...")
        self.lr_model = LinearRegression()
        self.lr_model.fit(gbdt_features, y)
        
        print("GBDT+LR模型训练完成")
        
    def _extract_features(self, user_id: int, item_id: int) -> pd.DataFrame:
        """
        为单个用户-物品对提取特征
        
        Parameters:
        -----------
        user_id : int
            用户ID
        item_id : int
            物品ID
            
        Returns:
        --------
        features : pd.DataFrame
            特征向量
        """
        # 基本特征
        user_mean = self.user_stats.get(user_id, {'mean': self.global_mean})['mean']
        user_count = self.user_stats.get(user_id, {'count': 0})['count']
        user_std = self.user_stats.get(user_id, {'std': 0})['std']
        
        item_mean = self.item_stats.get(item_id, {'mean': self.global_mean})['mean']
        item_count = self.item_stats.get(item_id, {'count': 0})['count']
        item_std = self.item_stats.get(item_id, {'std': 0})['std']
        
        # 创建特征向量
        features = pd.DataFrame([[user_id, item_id, user_mean, user_count, user_std, 
                                item_mean, item_count, item_std]], 
                               columns=['user_id', 'item_id', 'user_mean', 'user_count', 'user_std', 
                                       'item_mean', 'item_count', 'item_std'])
        
        return features
        
    def predict(self, user_id: int, item_id: int) -> float:
        """
        预测用户对物品的评分
        
        Parameters:
        -----------
        user_id : int
            用户ID
        item_id : int
            物品ID
            
        Returns:
        --------
        rating : float
            预测评分
        """
        if not self.gbdt_model or not self.lr_model:
            # 如果模型未训练，则回退到基线方法
            return self.global_mean
        
        # 1. 提取特征
        features = self._extract_features(user_id, item_id)
        
        # 2. 使用GBDT模型进行特征转换
        leaf_indices = self.gbdt_model.predict(features, pred_leaf=True)
        
        # 3. 独热编码
        try:
            gbdt_features = self.one_hot_encoder.transform(leaf_indices)
        except Exception as e:
            # 如果独热编码失败，可能是遇到了训练集中未见过的叶子节点
            print(f"独热编码失败: {e}")
            # 回退到基线方法
            return self.item_means.get(item_id, self.global_mean)
        
        # 4. 使用LR模型预测
        pred = self.lr_model.predict(gbdt_features)[0]
        
        # 确保预测值在有效范围内
        return max(self.config.rating_min, min(self.config.rating_max, pred))
