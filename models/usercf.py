import math
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any, Optional
from tqdm import tqdm

from framework import BaseRecommender, ExperimentConfig

class UserCFRecommender(BaseRecommender):
    """
    基于用户的协同过滤推荐算法（改进版）
    
    主要特点：
    1. 实现了多种相似度计算方法（余弦相似度、皮尔逊相关系数和Jaccard）
    2. 支持惩罚活跃用户对相似度的影响
    3. 使用倒排表优化相似度计算，显著提升大规模数据集上的性能
    4. 实现用户均值中心化，降低评分偏差影响
    5. 提供对冷启动用户的特殊处理
    """
    
    def __init__(self, config: ExperimentConfig, 
                 similarity_top_k: int = 20, 
                 recommend_top_n: int = 10,
                 similarity_method: str = 'pearson',
                 penalize_active_users: bool = True,
                 use_mean_centering: bool = True):
        """
        初始化UserCF推荐器
        
        参数:
            config: 实验配置对象
            similarity_top_k: 预测评分时考虑的相似用户数量
            recommend_top_n: 推荐结果数量
            similarity_method: 相似度计算方法，支持'cosine', 'pearson', 'jaccard'
            penalize_active_users: 是否惩罚活跃用户的相似度贡献
            use_mean_centering: 是否使用均值中心化处理评分
        """
        super().__init__(config)
        self.similarity_top_k = similarity_top_k
        self.recommend_top_n = recommend_top_n
        self.similarity_method = similarity_method
        self.penalize_active_users = penalize_active_users
        self.use_mean_centering = use_mean_centering
        
        # 模型内部数据结构
        self.user_similarity = defaultdict(dict)  # 用户相似度矩阵 {user_id: {similar_user_id: similarity}}
        self.user_popular = defaultdict(int)      # 用户活跃度 {user_id: 交互过多少物品}
        self.user_ratings = {}                    # 用户评分字典 {user_id: {item_id: rating}}
        self.item_users = defaultdict(set)        # 物品-用户倒排表 {item_id: set(user_ids)}
    
    def fit(self, train_users: Dict, train_items: Dict) -> None:
        """
        训练UserCF模型，构建用户相似度矩阵
        
        参数:
            train_users: 用户-物品评分字典 {user_id: [(item_id, rating), ...]}
            train_items: 物品-用户评分字典 {item_id: [(user_id, rating), ...]}
        """
        # 调用父类的fit方法初始化基础数据
        super().fit(train_users, train_items)
        
        # 构建用户评分字典和物品-用户倒排表
        print("构建用户评分和倒排表...")
        for user_id, items in tqdm(train_users.items(), desc="处理用户数据"):
            # 转换评分数据为字典格式 {item_id: rating}
            user_rating_dict = {item_id: rating for item_id, rating in items}
            self.user_ratings[user_id] = user_rating_dict
            self.user_popular[user_id] = len(user_rating_dict)
            
            # 构建物品-用户倒排表
            for item_id in user_rating_dict:
                self.item_users[item_id].add(user_id)
        
        # 计算用户相似度矩阵
        self._calculate_user_similarity()
        
        print(f"UserCF模型训练完成: {len(self.user_similarity)}个用户, {len(self.item_users)}个物品")
    
    def _calculate_user_similarity(self) -> None:
        """计算用户相似度矩阵（多种方法，支持倒排表优化）"""
        print("计算用户相似度矩阵...")
        
        # 使用倒排表优化用户相似度计算
        print("1. 使用倒排表计算用户共现矩阵...")
        user_co_rating = defaultdict(lambda: defaultdict(list))  # {user_i: {user_j: [(rating_i, rating_j), ...]}}
        user_co_count = defaultdict(lambda: defaultdict(int))    # {user_i: {user_j: 共同评分物品数}}
        
        # 通过物品的倒排表计算用户共现情况
        for item_id, users in tqdm(self.item_users.items(), desc="处理物品倒排表"):
            # 仅当物品有足够多用户评分时才考虑（可选筛选）
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
        print("2. 计算用户相似度...")
        for user_i, related_users in tqdm(user_co_count.items(), desc="计算相似度"):
            for user_j, count in related_users.items():
                # 根据选择的相似度方法计算相似度
                if self.similarity_method == 'cosine':
                    # 余弦相似度
                    if self.penalize_active_users:
                        # 惩罚活跃用户
                        denominator = math.sqrt(self.user_popular[user_i] * self.user_popular[user_j])
                    else:
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
                    
                elif self.similarity_method == 'jaccard':
                    # 杰卡德相似系数
                    union_count = self.user_popular[user_i] + self.user_popular[user_j] - count
                    similarity = count / union_count if union_count > 0 else 0
                    
                else:
                    # 默认使用余弦相似度
                    denominator = math.sqrt(self.user_popular[user_i] * self.user_popular[user_j])
                    similarity = count / denominator if denominator > 0 else 0
                
                # 只保存正相似度
                if similarity > 0:
                    self.user_similarity[user_i][user_j] = similarity
    
    def recommend(self, user_id: int, n: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        为指定用户生成推荐列表
        
        参数:
            user_id: 用户ID
            n: 推荐物品数量，不指定则使用默认值
            
        返回:
            推荐物品列表，每项为(item_id, score)元组
        """
        if n is None:
            n = self.recommend_top_n
            
        # 检查是否是新用户
        if user_id not in self.user_ratings:
            return self._recommend_for_new_user(n)
        
        # 获取用户的历史评分物品
        user_hist_items = set(self.user_ratings[user_id].keys())
        
        # 获取与目标用户最相似的K个用户
        similar_users = sorted(
            self.user_similarity.get(user_id, {}).items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.similarity_top_k]
        
        # 候选物品及其评分
        candidate_items = defaultdict(float)
        
        # 用户均值，用于评分中心化
        user_mean = self.user_means.get(user_id, self.global_mean) if self.use_mean_centering else 0
        
        # 对每个相似用户，考虑他们评分过但目标用户未评分的物品
        for sim_user_id, similarity in similar_users:
            # 相似用户的均值，用于评分中心化
            sim_user_mean = self.user_means.get(sim_user_id, self.global_mean) if self.use_mean_centering else 0
            
            # 遍历相似用户评分过的物品
            for item_id, rating in self.user_ratings[sim_user_id].items():
                # 排除目标用户已评分物品
                if item_id not in user_hist_items:
                    # 如果使用均值中心化，考虑评分偏差
                    if self.use_mean_centering:
                        adjusted_rating = similarity * (rating - sim_user_mean)
                    else:
                        adjusted_rating = similarity * rating
                        
                    candidate_items[item_id] += adjusted_rating
        
        # 使用用户均值调整评分（如果启用均值中心化）
        if self.use_mean_centering:
            for item_id in candidate_items:
                candidate_items[item_id] = user_mean + candidate_items[item_id]
        
        # 对候选物品进行排序并计算最终得分
        # 此处还可以添加额外的排名因素，比如物品流行度或多样性考量
        
        # 排序并截取前n个物品作为推荐结果
        recommendations = sorted(
            candidate_items.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:n]
        
        return recommendations
    
    def _recommend_for_new_user(self, n: int) -> List[Tuple[int, float]]:
        """为新用户推荐热门物品"""
        # 根据物品被评分的用户数量计算流行度
        item_popularity = [(item_id, len(users)) for item_id, users in self.item_users.items()]
        
        # 按流行度排序并返回前n个物品
        return sorted(item_popularity, key=lambda x: x[1], reverse=True)[:n]
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        预测用户对物品的评分
        
        参数:
            user_id: 用户ID
            item_id: 物品ID
            
        返回:
            预测评分
        """
        # 冷启动处理
        if user_id not in self.user_ratings:
            return super().predict_for_user(user_id, item_id)
            
        # 用户已评分过该物品，直接返回评分
        if item_id in self.user_ratings[user_id]:
            return self.user_ratings[user_id][item_id]
            
        # 获取与目标用户最相似的K个用户
        similar_users = sorted(
            self.user_similarity.get(user_id, {}).items(),
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
            
        # 确保预测评分在有效范围内
        return max(self.config.rating_min, min(self.config.rating_max, prediction))
    
    def predict_all(self, test_pairs: List[Tuple[int, int]]) -> List[Tuple[int, int, float]]:
        """
        批量预测多个用户-物品对的评分
        
        参数:
            test_pairs: 测试集用户-物品对列表 [(user_id, item_id), ...]
            
        返回:
            预测结果列表 [(user_id, item_id, predicted_rating), ...]
        """
        predictions = []
        for user_id, item_id in tqdm(test_pairs, desc="UserCF预测"):
            # 使用predict_for_user确保冷启动处理和评分范围控制
            pred_score = self.predict_for_user(user_id, item_id)
            predictions.append((user_id, item_id, pred_score))
        
        return predictions
