import numpy as np
# 在顶部添加 ↓
from collections import defaultdict
import math

from typing import Dict
from framework import BaseRecommender, ExperimentConfig
from tqdm import tqdm

class UserCFRecommender(BaseRecommender):
    """基于用户的协同过滤推荐算法"""
    
    def __init__(self, config: ExperimentConfig, k=20):
        super().__init__(config)
        self.k = k  # 相似用户数量
        self.user_sim = {}  # 用户相似度字典
        self.user_ratings = {}  # 用户评分缓存

    def fit(self, train_users: Dict, train_items: Dict) -> None:
        super().fit(train_users, train_items)
        
        # 转换数据结构为 {user: {item: rating}}
        self.user_ratings = {
            user: {item: rating for item, rating in ratings}
            for user, ratings in train_users.items()
        }
        
        # 新增倒排表优化 ↓
        print("构建物品-用户倒排表...")
        item_users = defaultdict(set)
        for user, items in tqdm(self.user_ratings.items(), desc='倒排表'):
            for item in items:
                item_users[item].add(user)
        
        # 优化相似度计算 ↓
        print("计算用户相似度...")
        user_sim_matrix = defaultdict(lambda: defaultdict(float))
        user_item_count = defaultdict(int)
        
        # 基于倒排表计算
        for _, users in tqdm(item_users.items(), desc='处理物品'):
            for u in users:
                user_item_count[u] += 1
                for v in users:
                    if u != v:
                        user_sim_matrix[u][v] += 1
        
        # 计算余弦相似度
        for u, related_users in tqdm(user_sim_matrix.items(), desc='标准化'):
            for v, count in related_users.items():
                denominator = math.sqrt(user_item_count[u] * user_item_count[v])
                user_sim_matrix[u][v] = count / denominator if denominator != 0 else 0
                
        self.user_sim = user_sim_matrix

    def predict(self, user_id: int, item_id: int) -> float:
        if user_id not in self.user_ratings:
            return super().predict_for_user(user_id, item_id)
            
        # 获取topK相似用户
        sim_users = sorted(self.user_sim.get(user_id, {}).items(), 
                         key=lambda x: x[1], reverse=True)[:self.k]
        
        numerator = 0
        denominator = 0
        for sim_user, similarity in sim_users:
            if item_id in self.user_ratings[sim_user]:
                user_mean = np.mean(list(self.user_ratings[sim_user].values()))
                numerator += similarity * (self.user_ratings[sim_user][item_id] - user_mean)
                denominator += abs(similarity)
        
        if denominator == 0:
            return self.user_means.get(user_id, self.global_mean)
            
        return np.mean(list(self.user_ratings[user_id].values())) + (numerator / denominator)
