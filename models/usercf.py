import math
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm

from framework import BaseRecommender, ExperimentConfig

class UserCFRecommender(BaseRecommender):
    """
    基于用户的协同过滤推荐算法
    
    支持的相似度计算方法:
    1. 余弦相似度 (cosine)
    2. 皮尔逊相关系数 (pearson)
    """
    
    def __init__(self, config: ExperimentConfig, 
                 similarity_top_k: int = 20,
                 similarity_method: str = 'cosine',
                 use_mean_centering: bool = True):
        """初始化UserCF推荐器"""
        super().__init__(config)
        self.similarity_top_k = similarity_top_k     # 相似用户数量
        self.similarity_method = similarity_method   # 相似度计算方法
        self.use_mean_centering = use_mean_centering # 是否使用均值中心化
        
        # 模型内部数据结构
        self.user_similarity = defaultdict(dict)  # 用户相似度矩阵 {user_id: {similar_user_id: similarity}}
        self.user_ratings = {}                    # 用户评分字典 {user_id: {item_id: rating}}
        self.item_users = defaultdict(set)        # 物品-用户倒排表 {item_id: set(user_ids)}
    
    def fit(self, train_users: Dict, train_items: Dict) -> None:
        """训练UserCF模型，构建用户相似度矩阵"""
        # 调用父类的fit方法初始化基础数据
        super().fit(train_users, train_items)
        
        # 构建用户评分字典和物品-用户倒排表
        print("构建用户评分和倒排表...")
        for user_id, items in tqdm(train_users.items(), desc="处理用户数据"):
            # 转换评分数据为字典格式 {item_id: rating}
            user_rating_dict = {item_id: rating for item_id, rating in items}
            self.user_ratings[user_id] = user_rating_dict
            
            # 构建物品-用户倒排表
            for item_id in user_rating_dict:
                self.item_users[item_id].add(user_id)
        
        # 计算用户相似度矩阵
        self._calculate_user_similarity()
        
        print(f"UserCF模型训练完成: {len(self.user_similarity)}个用户, {len(self.item_users)}个物品")
    
    def _calculate_user_similarity(self) -> None:
        """计算用户相似度矩阵（余弦相似度和皮尔逊相关系数）"""
        print("计算用户相似度矩阵...")
        
        # 使用倒排表优化用户相似度计算
        print("使用倒排表计算用户共现矩阵...")
        user_co_rating = defaultdict(lambda: defaultdict(list))  # {user_i: {user_j: [(rating_i, rating_j), ...]}}
        user_co_count = defaultdict(lambda: defaultdict(int))    # {user_i: {user_j: 共同评分物品数}}
        
        # 通过物品的倒排表计算用户共现情况
        for item_id, users in tqdm(self.item_users.items(), desc="处理物品倒排表"):
            # 仅当物品有足够多用户评分时才考虑
            if len(users) < 2:
                continue
                
            # 遍历评分该物品的每对用户组合
            users_list = list(users)
            for i in range(len(users_list)):
                user_i = users_list[i]
                rating_i = self.user_ratings[user_i][item_id]
                
                for j in range(i + 1, len(users_list)):
                    user_j = users_list[j]
                    rating_j = self.user_ratings[user_j][item_id]
                    
                    # 增加共同评分物品数量计数
                    user_co_count[user_i][user_j] += 1
                    user_co_count[user_j][user_i] += 1
                    
                    # 保存评分对，用于计算皮尔逊相关系数
                    user_co_rating[user_i][user_j].append((rating_i, rating_j))
                    user_co_rating[user_j][user_i].append((rating_j, rating_i))
        
        # 计算用户相似度
        print("计算用户相似度...")
        for user_i, related_users in tqdm(user_co_count.items(), desc="计算相似度"):
            for user_j, count in related_users.items():
                # 根据选择的相似度方法计算相似度
                if self.similarity_method == 'cosine':
                    # 余弦相似度
                    denominator = math.sqrt(len(self.user_ratings[user_i]) * len(self.user_ratings[user_j]))
                    similarity = count / denominator if denominator > 0 else 0
                
                elif self.similarity_method == 'pearson':
                    # 皮尔逊相关系数
                    if count < 2:  # 至少需要两个共同评分
                        similarity = 0
                        continue
                        
                    ratings_i = [r[0] for r in user_co_rating[user_i][user_j]]
                    ratings_j = [r[1] for r in user_co_rating[user_i][user_j]]
                    
                    mean_i = np.mean(ratings_i)
                    mean_j = np.mean(ratings_j)
                    
                    numerator = sum((ri - mean_i) * (rj - mean_j) for ri, rj in zip(ratings_i, ratings_j))
                    denominator_i = math.sqrt(sum((ri - mean_i) ** 2 for ri in ratings_i))
                    denominator_j = math.sqrt(sum((rj - mean_j) ** 2 for rj in ratings_j))
                    denominator = denominator_i * denominator_j
                    
                    similarity = numerator / denominator if denominator > 0 else 0
                
                else:
                    # 默认使用余弦相似度
                    denominator = math.sqrt(len(self.user_ratings[user_i]) * len(self.user_ratings[user_j]))
                    similarity = count / denominator if denominator > 0 else 0
                
                # 只保存正相似度
                if similarity > 0:
                    self.user_similarity[user_i][user_j] = similarity
    
    def predict(self, user_id: int, item_id: int) -> float:
        """预测用户对物品的评分"""
        # 用户已评分过该物品，直接返回评分
        if user_id in self.user_ratings and item_id in self.user_ratings[user_id]:
            return self.user_ratings[user_id][item_id]
            
        # 获取与目标用户最相似的K个用户
        similar_users = []
        if user_id in self.user_similarity:
            similar_users = sorted(
                self.user_similarity[user_id].items(),
                key=lambda x: x[1],
                reverse=True
            )[:self.similarity_top_k]
        
        # 用于累积加权评分
        weighted_sum = 0
        similarity_sum = 0
        
        # 目标用户评分均值（用于中心化）
        user_mean = self.user_means.get(user_id, self.global_mean)
        
        # 遍历相似用户
        for sim_user_id, similarity in similar_users:
            # 只考虑评分过目标物品的相似用户
            if item_id in self.user_ratings[sim_user_id]:
                sim_user_rating = self.user_ratings[sim_user_id][item_id]
                
                # 使用评分中心化处理评分偏差
                if self.use_mean_centering:
                    sim_user_mean = self.user_means.get(sim_user_id, self.global_mean)
                    weighted_sum += similarity * (sim_user_rating - sim_user_mean)
                else:
                    weighted_sum += similarity * sim_user_rating
                    
                similarity_sum += similarity
        
        # 如果没有足够的数据做出预测
        if similarity_sum == 0:
            return self.item_means.get(item_id, self.global_mean)
            
        # 计算最终预测评分，加入均值补偿
        if self.use_mean_centering:
            prediction = user_mean + (weighted_sum / similarity_sum)
        else:
            prediction = weighted_sum / similarity_sum
            
        return prediction
    
    def predict_all(self, test_pairs: List[Tuple[int, int]]) -> List[Tuple[int, int, float]]:
        """批量预测多个用户-物品对的评分"""
        predictions = []
        for user_id, item_id in tqdm(test_pairs, desc="UserCF预测"):
            # 使用predict_for_user确保冷启动处理和评分范围控制
            pred_score = self.predict_for_user(user_id, item_id)
            predictions.append((user_id, item_id, pred_score))
        
        return predictions
