import math
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm

from framework import BaseRecommender, ExperimentConfig

class ItemCFRecommender(BaseRecommender):
    """
    基于物品的协同过滤推荐算法
    
    支持的相似度计算方法:
    1. 余弦相似度 (cosine)
    2. 皮尔逊相关系数 (pearson)
    3. 改进的相似度计算 (improved)
    """
    
    def __init__(self, config: ExperimentConfig, 
                 similarity_top_k: int = 20,
                 similarity_method: str = 'improved',
                 alpha: float = 0.5,
                 penalize_active_users: bool = True):  # 是否惩罚活跃用户
        """初始化ItemCF推荐器"""
        super().__init__(config)
        self.similarity_top_k = similarity_top_k            # 相似物品数量
        self.similarity_method = similarity_method          # 相似度计算方法
        self.alpha = alpha                                  # 控制对热门物品的惩罚力度
        self.penalize_active_users = penalize_active_users  # 是否惩罚活跃用户
        
        # 模型内部数据结构
        self.item_similarity = defaultdict(dict)  # 物品相似度矩阵 {item_id: {similar_item_id: similarity}}
        self.item_popular = defaultdict(int)      # 物品流行度 {item_id: 被多少用户交互过}
        self.user_items = defaultdict(set)        # 用户-物品交互集合 {user_id: set(item_ids)}
        self.item_users = defaultdict(set)        # 物品-用户倒排表 {item_id: set(user_ids)}
        self.user_counts = defaultdict(int)       # 用户活跃度 {user_id: 交互物品数量}
        
    def fit(self, train_users: Dict, train_items: Dict) -> None:
        """训练ItemCF模型，构建物品相似度矩阵"""
        # 调用父类的fit方法初始化基础数据
        super().fit(train_users, train_items)
        
        # 构建用户-物品交互集合、物品-用户倒排表和用户活跃度
        print("构建用户-物品交互数据...")
        for user_id, items in tqdm(train_users.items(), desc="处理用户数据"):
            user_items = set(item_id for item_id, _ in items)
            self.user_items[user_id] = user_items
            self.user_counts[user_id] = len(user_items)  # 记录用户活跃度
            
            for item_id in user_items:
                self.item_users[item_id].add(user_id)
                self.item_popular[item_id] += 1
                
        # 计算物品相似度矩阵
        self._calculate_item_similarity()
        
        print(f"ItemCF模型训练完成: {len(self.item_similarity)}个物品, {len(self.user_items)}个用户")
        
    def _calculate_item_similarity(self) -> None:
        """计算物品相似度矩阵（根据选择的方法）"""
        print("计算物品相似度矩阵...")
        
        if self.similarity_method == 'cosine':
            self._calculate_item_similarity_cosine()
        elif self.similarity_method == 'pearson':
            self._calculate_item_similarity_pearson()
        elif self.similarity_method == 'improved':
            self._calculate_item_similarity_improved()
        else:
            print(f"不支持的相似度方法: {self.similarity_method}，使用改进的相似度计算")
            self._calculate_item_similarity_improved()

    def _calculate_item_similarity_cosine(self) -> None:
        """使用标准余弦相似度计算物品相似度"""
        print("计算物品余弦相似度...")
        
        # 初始化相似度矩阵
        similarity_matrix = defaultdict(lambda: defaultdict(float))
        
        # 遍历所有物品对
        with tqdm(total=len(self.item_users), desc="计算物品相似度") as pbar:
            for item_i in self.item_users:
                users_i = self.item_users[item_i]
                
                for item_j in self.item_users:
                    if item_i >= item_j:  # 避免重复计算
                        continue
                    
                    users_j = self.item_users[item_j]
                    
                    # 找到同时喜欢物品i和j的用户集合
                    common_users = users_i & users_j
                    
                    if not common_users:
                        continue
                    
                    # 标准余弦相似度: |N(i) ∩ N(j)| / sqrt(|N(i)| * |N(j)|)
                    denominator = math.sqrt(len(users_i) * len(users_j))
                    sim_ij = len(common_users) / denominator if denominator > 0 else 0
                    
                    # 保存正相似度（对称的）
                    if sim_ij > 0:
                        similarity_matrix[item_i][item_j] = sim_ij
                        similarity_matrix[item_j][item_i] = sim_ij
                
                pbar.update(1)
        
        # 将计算结果保存到模型
        self.item_similarity = similarity_matrix

    def _calculate_item_similarity_improved(self) -> None:
        """使用改进的方法计算物品相似度"""
        print("使用改进的相似度计算方法...")
        
        # 初始化相似度矩阵
        similarity_matrix = defaultdict(lambda: defaultdict(float))
        
        # 遍历所有物品对
        with tqdm(total=len(self.item_users), desc="计算物品相似度") as pbar:
            for item_i in self.item_users:
                users_i = self.item_users[item_i]
                
                for item_j in self.item_users:
                    if item_i >= item_j:  # 避免重复计算
                        continue
                    
                    users_j = self.item_users[item_j]
                    
                    # 找到同时喜欢物品i和j的用户集合
                    common_users = users_i & users_j
                    
                    if not common_users:
                        continue
                    
                    # 使用改进公式: |N(i) ∩ N(j)| / (|N(i)|^(1-α) * |N(j)|^α)
                    denominator = (len(users_i) ** (1 - self.alpha)) * (len(users_j) ** self.alpha)
                    
                    if self.penalize_active_users:
                        # 对活跃用户进行惩罚
                        numerator = 0.0
                        for user in common_users:
                            # 用户贡献与活跃度成反比
                            user_weight = 1.0 / math.log(1.0 + self.user_counts[user])
                            numerator += user_weight
                    else:
                        numerator = len(common_users)
                        
                    sim_ij = numerator / denominator if denominator > 0 else 0
                    
                    # 保存正相似度（对称的）
                    if sim_ij > 0:
                        similarity_matrix[item_i][item_j] = sim_ij
                        similarity_matrix[item_j][item_i] = sim_ij
                
                pbar.update(1)
        
        # 将计算结果保存到模型
        self.item_similarity = similarity_matrix

    def _calculate_item_similarity_pearson(self) -> None:
        """使用皮尔逊相关系数计算物品相似度"""
        print("计算物品皮尔逊相关系数...")
        
        # 遍历所有物品对
        with tqdm(total=len(self.item_users), desc="计算物品相似度") as pbar:
            for item_i in self.item_users:
                users_i = self.item_users[item_i]
                
                for item_j in self.item_users:
                    if item_i >= item_j:  # 避免重复计算
                        continue
                    
                    users_j = self.item_users[item_j]
                    common_users = users_i & users_j
                    
                    if len(common_users) < 2:  # 需要至少两个共同用户
                        continue
                    
                    # 计算每个物品的用户评分
                    i_ratings = []
                    j_ratings = []
                    
                    for user in common_users:
                        # 获取用户对两个物品的评分
                        for u, r in self.train_items[item_i]:
                            if u == user:
                                i_ratings.append(r)
                                break
                        
                        for u, r in self.train_items[item_j]:
                            if u == user:
                                j_ratings.append(r)
                                break
                    
                    # 计算皮尔逊相关系数
                    if len(i_ratings) >= 2:
                        mean_i = np.mean(i_ratings)
                        mean_j = np.mean(j_ratings)
                        
                        numerator = sum((ri - mean_i) * (rj - mean_j) for ri, rj in zip(i_ratings, j_ratings))
                        denominator_i = math.sqrt(sum((r - mean_i) ** 2 for r in i_ratings))
                        denominator_j = math.sqrt(sum((r - mean_j) ** 2 for r in j_ratings))
                        denominator = denominator_i * denominator_j
                        
                        sim_ij = numerator / denominator if denominator != 0 else 0
                        
                        # 保存正相似度（对称的）
                        if sim_ij > 0:
                            self.item_similarity[item_i][item_j] = sim_ij
                            self.item_similarity[item_j][item_i] = sim_ij
                
                pbar.update(1)
    
    def predict(self, user_id: int, item_id: int) -> float:
        """预测用户对物品的评分"""
        # 用户已评分过该物品，直接返回评分
        if user_id in self.train_users:
            user_rated_items = self.user_items.get(user_id, set())
            if item_id in user_rated_items:
                # 获取用户对该物品的实际评分
                for i, r in self.train_users[user_id]:
                    if i == item_id:
                        return r
            
        # 基于物品相似度预测评分
        weighted_sum = 0.0
        similarity_sum = 0.0
        
        # 遍历用户交互过的所有物品
        if user_id in self.user_items:
            user_hist_items = self.user_items[user_id]
            
            # 获取与目标物品最相似的物品及其相似度
            item_similarities = []
            for hist_item in user_hist_items:
                similarity = self.item_similarity.get(hist_item, {}).get(item_id, 0.0)
                if similarity > 0:
                    item_similarities.append((hist_item, similarity))
            
            # 根据相似度排序，选择前K个最相似物品
            item_similarities.sort(key=lambda x: x[1], reverse=True)
            top_items = item_similarities[:self.similarity_top_k]
            
            # 基于相似物品的评分加权预测
            for hist_item, similarity in top_items:
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
        
        # 如果没有足够的相似物品数据，使用物品平均分或全局平均分
        if similarity_sum == 0:
            return self.item_means.get(item_id, self.global_mean)
            
        # 计算预测评分
        prediction = weighted_sum / similarity_sum
        
        return prediction
    
    def predict_all(self, test_pairs: List[Tuple[int, int]]) -> List[Tuple[int, int, float]]:
        """批量预测多个用户-物品对的评分"""
        predictions = []
        for user_id, item_id in tqdm(test_pairs, desc="ItemCF预测"):
            # 使用predict_for_user确保冷启动处理和评分范围控制
            pred_score = self.predict_for_user(user_id, item_id)
            predictions.append((user_id, item_id, pred_score))
        
        return predictions
