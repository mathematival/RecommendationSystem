import math
from collections import defaultdict
from tqdm import tqdm

from framework import BaseRecommender

class ItemCFRecommender(BaseRecommender):
    def __init__(self, config, similarity_top_k=20, recommend_top_n=10):
        """ItemCF 推荐模型初始化"""
        super().__init__(config)
        self.similarity_top_k = similarity_top_k  # K参数（相似物品数）
        self.recommend_top_n = recommend_top_n    # N参数（推荐数量）
        
        # 模型内部数据结构
        self.item_similarity = defaultdict(dict)  # 物品相似度矩阵 {item: {similar_item: score}}
        self.item_popular = defaultdict(int)       # 物品流行度（被多少用户交互过）
        self.trn_user_items = None                # 训练数据（用户-物品交互字典）

    def fit(self, train_users, train_items=None):
        """训练模型（构建物品相似度矩阵）"""
        # 调用父类的fit方法计算全局均值和各类均值
        super().fit(train_users, train_items)
        
        # 构建用户-物品交互字典
        self.trn_user_items = {
            user: {item for item, _ in items} 
            for user, items in train_users.items()
        }
        
        self._build_item_popularity()  # 统计物品流行度
        self._calculate_item_similarity()  # 计算物品相似度矩阵

    def _build_item_popularity(self):
        """统计每个物品被多少用户交互过（流行度）"""
        for user, items in self.trn_user_items.items():
            for item in items:
                self.item_popular[item] += 1

    def _calculate_item_similarity(self):
        """计算物品相似度矩阵（优化版，带进度条）"""
        # 构建物品-用户倒排表（物品到交互用户的集合）
        item_users = defaultdict(set)
        for user, items in self.trn_user_items.items():
            for item in items:
                item_users[item].add(user)

        # 优化：通过用户交互的物品集合统计共同用户数（原双重物品循环改为用户物品对循环）
        sim_matrix = defaultdict(lambda: defaultdict(int))
        # 遍历用户并添加进度条（关键优化点1）
        for user, items in tqdm(self.trn_user_items.items(), desc="计算物品共同用户数"):
            items_list = list(items)  # 转为列表方便索引
            # 仅遍历用户交互物品的两两组合（i < j 避免重复计算）
            for i in range(len(items_list)):
                item_i = items_list[i]
                for j in range(i + 1, len(items_list)):
                    item_j = items_list[j]
                    sim_matrix[item_i][item_j] += 1  # 共同用户数+1
                    sim_matrix[item_j][item_i] += 1  # 对称矩阵

        # 归一化为余弦相似度并添加进度条（关键优化点2）
        for item_i in tqdm(sim_matrix.keys(), desc="计算物品相似度"):
            related_items = sim_matrix[item_i]
            for item_j, common_count in related_items.items():
                denominator = math.sqrt(self.item_popular[item_i] * self.item_popular[item_j])
                self.item_similarity[item_i][item_j] = common_count / denominator if denominator != 0 else 0

    def recommend(self, user_id):
        """为指定用户生成推荐（返回带分数的列表）"""
        if self.trn_user_items is None:
            raise ValueError("模型未训练，请先调用 fit() 方法")
        
        # 对于新用户，返回空列表
        if user_id not in self.trn_user_items:
            return []

        user_hist_items = self.trn_user_items[user_id]
        candidate_scores = defaultdict(float)

        for hist_item in user_hist_items:
            sorted_similar = sorted(
                self.item_similarity[hist_item].items(),
                key=lambda x: x[1],
                reverse=True
            )[:self.similarity_top_k]

            for sim_item, score in sorted_similar:
                if sim_item not in user_hist_items:
                    candidate_scores[sim_item] += score

        # 保留分数，返回 [(item, score)] 格式
        return sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:self.recommend_top_n]

    def predict(self, user_id, item_id):
        """符合框架接口的预测方法"""
        if user_id not in self.trn_user_items:
            # 对于框架中的冷启动处理，使用父类的预测
            return super().predict_for_user(user_id, item_id)
        
        # 如果用户已经评分过该物品，直接返回该物品的平均评分
        if user_id in self.trn_user_items and item_id in self.trn_user_items[user_id]:
            return self.item_means.get(item_id, self.global_mean)
            
        # 否则使用协同过滤进行预测
        user_recommendations = self.recommend(user_id)
        for rec_item, score in user_recommendations:
            if rec_item == item_id:
                # 将相似度分数缩放到评分范围
                rating_range = self.config.rating_max - self.config.rating_min
                pred = self.config.rating_min + score * rating_range
                # 确保预测值在配置的评分范围内
                return max(self.config.rating_min, min(self.config.rating_max, pred))
                
        # 如果无法做出预测，使用物品平均值
        return self.item_means.get(item_id, self.global_mean)

    def predict_all(self, test_pairs):
        """框架要求的批量预测方法（返回三元组列表）"""
        predictions = []
        for user_id, item_id in tqdm(test_pairs, desc="ItemCF预测"):
            # 使用predict_for_user确保评分范围和冷启动处理完全符合框架标准
            pred_score = self.predict_for_user(user_id, item_id)
            predictions.append((user_id, item_id, pred_score))
        return predictions
