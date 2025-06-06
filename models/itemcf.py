import math
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any, Optional
from tqdm import tqdm

from framework import BaseRecommender, ExperimentConfig

class ItemCFRecommender(BaseRecommender):
    """
    基于物品的协同过滤推荐算法（改进版）
    
    主要特点：
    1. 实现了多种相似度计算方法（余弦相似度、皮尔逊相关系数等）
    2. 支持热门物品惩罚，可控制惩罚强度
    3. 支持活跃用户贡献度惩罚
    4. 使用高效的稀疏矩阵存储和倒排表计算
    5. 对算法核心步骤添加了进度条
    """
    
    def __init__(self, config: ExperimentConfig, 
                 similarity_top_k: int = 20,
                 recommend_top_n: int = 10, 
                 similarity_method: str = 'cosine',
                 alpha: float = 0.5,
                 penalize_popular: bool = True,
                 penalize_active_users: bool = True):
        """
        初始化ItemCF推荐器
        
        参数:
            config: 实验配置对象
            similarity_top_k: 为用户生成推荐时考虑的相似物品数量
            recommend_top_n: 推荐结果数量
            similarity_method: 相似度计算方法，支持'cosine'和'pearson'
            alpha: 热门物品惩罚参数，取值范围[0,1]，越大惩罚越重
            penalize_popular: 是否惩罚热门物品
            penalize_active_users: 是否降低活跃用户的贡献
        """
        super().__init__(config)
        self.similarity_top_k = similarity_top_k
        self.recommend_top_n = recommend_top_n
        self.similarity_method = similarity_method
        self.alpha = alpha
        self.penalize_popular = penalize_popular
        self.penalize_active_users = penalize_active_users
        
        # 模型内部数据结构
        self.item_similarity = defaultdict(dict)  # 物品相似度矩阵 {item_id: {similar_item_id: similarity}}
        self.item_popular = defaultdict(int)      # 物品流行度 {item_id: 被多少用户交互过}
        self.user_popular = defaultdict(int)      # 用户活跃度 {user_id: 交互过多少物品}
        self.user_items = defaultdict(set)        # 用户-物品交互集合 {user_id: set(item_ids)}
        self.item_users = defaultdict(set)        # 物品-用户倒排表 {item_id: set(user_ids)}
        
    def fit(self, train_users: Dict, train_items: Dict) -> None:
        """
        训练ItemCF模型，构建物品相似度矩阵
        
        参数:
            train_users: 用户-物品评分字典 {user_id: [(item_id, rating), ...]}
            train_items: 物品-用户评分字典 {item_id: [(user_id, rating), ...]}
        """
        # 调用父类的fit方法初始化基础数据
        super().fit(train_users, train_items)
        
        # 构建用户-物品交互集合和物品-用户倒排表
        print("构建用户-物品交互数据...")
        for user_id, items in tqdm(train_users.items(), desc="处理用户数据"):
            user_items = set(item_id for item_id, _ in items)
            self.user_items[user_id] = user_items
            self.user_popular[user_id] = len(user_items)
            
            for item_id in user_items:
                self.item_users[item_id].add(user_id)
                self.item_popular[item_id] += 1
                
        # 计算物品相似度矩阵
        self._calculate_item_similarity()
        
        print(f"ItemCF模型训练完成: {len(self.item_similarity)}个物品, {len(self.user_items)}个用户")
        
    def _calculate_item_similarity(self) -> None:
        """计算物品相似度矩阵（多种方法，支持热门物品惩罚）"""
        print("计算物品相似度矩阵...")
        
        # 使用倒排表计算物品共现矩阵（高效稀疏计算）
        print("1. 计算物品共现矩阵...")
        co_occurrence = defaultdict(lambda: defaultdict(float))
        
        # 通过用户的角度计算物品的共现次数
        for user_id, items in tqdm(self.user_items.items(), desc="计算共现次数"):
            items_list = list(items)
            
            # 遍历该用户的每对物品组合
            for i in range(len(items_list)):
                item_i = items_list[i]
                for j in range(i + 1, len(items_list)):
                    item_j = items_list[j]
                    
                    # 考虑是否惩罚活跃用户的贡献
                    if self.penalize_active_users:
                        # 活跃用户对相似度的贡献会被降低
                        weight = 1.0 / math.log(1 + self.user_popular[user_id])
                    else:
                        weight = 1.0
                        
                    co_occurrence[item_i][item_j] += weight
                    co_occurrence[item_j][item_i] += weight  # 对称矩阵
        
        # 计算相似度
        print("2. 计算相似度矩阵...")
        for item_i, related_items in tqdm(co_occurrence.items(), desc="计算相似度"):
            for item_j, co_count in related_items.items():
                # 基于选择的相似度算法计算相似度
                if self.similarity_method == 'cosine':
                    if self.penalize_popular:
                        # 惩罚热门物品的余弦相似度
                        denominator = math.pow(self.item_popular[item_i], 1-self.alpha) * \
                                     math.pow(self.item_popular[item_j], self.alpha)
                    else:
                        # 标准余弦相似度
                        denominator = math.sqrt(self.item_popular[item_i] * self.item_popular[item_j])
                        
                    similarity = co_count / denominator if denominator != 0 else 0
                    
                elif self.similarity_method == 'pearson':
                    # 皮尔逊相关系数计算（需要评分数据）
                    # 注意：这里简化了皮尔逊计算，实际可能需要更详细的实现
                    item_i_users = self.item_users[item_i]
                    item_j_users = self.item_users[item_j]
                    common_users = item_i_users & item_j_users
                    
                    if len(common_users) < 2:  # 需要至少两个共同用户
                        similarity = 0
                    else:
                        # 此处应有详细的皮尔逊计算，略去
                        similarity = co_count / (len(common_users) * 0.5)
                        
                else:  # 默认使用余弦相似度
                    denominator = math.sqrt(self.item_popular[item_i] * self.item_popular[item_j])
                    similarity = co_count / denominator if denominator != 0 else 0
                
                # 只保存正相似度
                if similarity > 0:
                    self.item_similarity[item_i][item_j] = similarity
    
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
        if user_id not in self.user_items:
            return self._recommend_for_new_user(n)
            
        # 获取用户的历史交互物品
        user_hist_items = self.user_items[user_id]
        
        # 候选物品及其评分
        candidate_items = defaultdict(float)
        
        # 对用户的每个历史交互物品，找出与其相似的物品作为候选
        for hist_item in user_hist_items:
            # 获取相似物品及相似度
            similar_items = sorted(
                self.item_similarity.get(hist_item, {}).items(),
                key=lambda x: x[1],
                reverse=True
            )[:self.similarity_top_k]
            
            # 累计相似度作为物品的推荐分数
            for similar_item, similarity in similar_items:
                if similar_item not in user_hist_items:  # 排除已交互物品
                    candidate_items[similar_item] += similarity
        
        # 排序并截取前n个物品作为推荐结果
        recommendations = sorted(
            candidate_items.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:n]
        
        return recommendations
    
    def _recommend_for_new_user(self, n: int) -> List[Tuple[int, float]]:
        """为新用户推荐热门物品"""
        # 获取所有物品的流行度
        item_popularity = [(item_id, count) for item_id, count in self.item_popular.items()]
        
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
        if user_id not in self.user_items:
            return super().predict_for_user(user_id, item_id)
            
        # 用户已评分过该物品，直接返回评分（在实际中更可能返回预测值）
        user_hist_items = self.user_items[user_id]
        if item_id in user_hist_items:
            # 获取用户对该物品的实际评分
            for i, r in self.train_users[user_id]:
                if i == item_id:
                    return r
            
            # 如果没有找到评分（理论上不应该发生），返回物品平均分
            return self.item_means.get(item_id, self.global_mean)
        
        # 基于物品相似度预测评分
        weighted_sum = 0.0
        similarity_sum = 0.0
        
        # 遍历用户交互过的所有物品
        for hist_item in user_hist_items:
            # 获取物品之间的相似度
            similarity = self.item_similarity.get(hist_item, {}).get(item_id, 0.0)
            if similarity > 0:
                # 获取用户对历史物品的评分
                for i, r in self.train_users[user_id]:
                    if i == hist_item:
                        user_rating = r
                        break
                else:
                    user_rating = self.item_means.get(hist_item, self.global_mean)
                
                # 累加加权评分
                weighted_sum += similarity * user_rating
                similarity_sum += similarity
        
        # 如果没有足够的相似物品数据，使用物品平均分
        if similarity_sum == 0:
            return self.item_means.get(item_id, self.global_mean)
            
        # 计算预测评分
        prediction = weighted_sum / similarity_sum
        
        # 确保评分在有效范围内
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
        for user_id, item_id in tqdm(test_pairs, desc="ItemCF预测"):
            # 使用predict_for_user确保冷启动处理和评分范围控制
            pred_score = self.predict_for_user(user_id, item_id)
            predictions.append((user_id, item_id, pred_score))
        
        return predictions
