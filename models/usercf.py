import numpy as np
from collections import defaultdict
import math
from typing import Dict, List, Tuple, Set, Any
from tqdm import tqdm

from framework import BaseRecommender, ExperimentConfig

class UserCFRecommender(BaseRecommender):
    """基于用户的协同过滤推荐算法"""
    
    def __init__(self, config: ExperimentConfig, k=15):
        super().__init__(config)
        self.k = k  # 相似用户数量
        self.user_sim = {}  # 用户相似度字典
        self.user_ratings = {}  # 用户评分缓存

    def fit(self, train_users: Dict, train_items: Dict) -> None:
        """训练用户协同过滤模型"""
        # 调用父类的fit方法计算全局均值和各类均值
        super().fit(train_users, train_items)
        
        # 转换数据结构为 {user: {item: rating}}
        self.user_ratings = {
            user: {item: rating for item, rating in ratings}
            for user, ratings in train_users.items()
        }
        
        # 构建物品-用户倒排表
        print("构建物品-用户倒排表...")
        item_users = defaultdict(set)
        for user, items in tqdm(self.user_ratings.items(), desc='倒排表'):
            for item in items:
                item_users[item].add(user)
        
        # 优化相似度计算
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
        """预测用户对物品的评分"""
        if user_id not in self.user_ratings:
            return super().predict_for_user(user_id, item_id)
            
        # 用户已经评分过该物品，直接返回该评分
        if item_id in self.user_ratings[user_id]:
            return self.user_ratings[user_id][item_id]
            
        # 获取topK相似用户
        sim_users = sorted(self.user_sim.get(user_id, {}).items(), 
                         key=lambda x: x[1], reverse=True)[:self.k]
        
        numerator = 0
        denominator = 0
        for sim_user, similarity in sim_users:
            if sim_user in self.user_ratings and item_id in self.user_ratings[sim_user]:
                user_mean = np.mean(list(self.user_ratings[sim_user].values()))
                numerator += similarity * (self.user_ratings[sim_user][item_id] - user_mean)
                denominator += abs(similarity)
        
        if denominator == 0:
            return self.user_means.get(user_id, self.global_mean)
            
        # 使用协同过滤公式计算评分预测
        user_mean = np.mean(list(self.user_ratings[user_id].values())) if user_id in self.user_ratings else self.global_mean
        predicted_rating = user_mean + (numerator / denominator)
        
        # 确保预测值在配置的评分范围内
        return max(self.config.rating_min, min(self.config.rating_max, predicted_rating))
    
    def predict_all(self, test_pairs: List[Tuple[int, int]]) -> List[Tuple[int, int, float]]:
        """批量预测多个用户-物品对的评分"""
        predictions = []
        for user_id, item_id in tqdm(test_pairs, desc="UserCF预测"):
            # 使用predict_for_user确保冷启动策略被应用
            pred_score = self.predict_for_user(user_id, item_id)
            predictions.append((user_id, item_id, pred_score))
        return predictions
        
    def recommend_for_user(self, user_id: int, n=10, exclude_rated=True) -> List[Tuple[int, float]]:
        """为指定用户推荐n个物品"""
        if user_id not in self.user_ratings:
            # 对于新用户，返回热门物品
            popular_items = [(item, len(users)) for item, users in self.train_items.items()]
            return [(item, score) for item, score in 
                    sorted(popular_items, key=lambda x: x[1], reverse=True)[:n]]
        
        # 用户已评分物品
        user_rated_items = set(self.user_ratings[user_id].keys()) if exclude_rated else set()
        
        # 候选物品评分
        candidate_scores = defaultdict(float)
        
        # 获取相似用户
        sim_users = sorted(self.user_sim.get(user_id, {}).items(), 
                         key=lambda x: x[1], reverse=True)[:self.k]
        
        # 汇总相似用户评分
        for sim_user, similarity in sim_users:
            if sim_user in self.user_ratings:
                for item, rating in self.user_ratings[sim_user].items():
                    if item not in user_rated_items:
                        candidate_scores[item] += similarity * rating
        
        # 返回评分最高的n个物品
        return sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:n]
