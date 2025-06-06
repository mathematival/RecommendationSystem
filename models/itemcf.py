import math
from collections import defaultdict
from tqdm import tqdm  # 新增：进度条库

class ItemCFRecommender:
    def __init__(self, config, similarity_top_k=20, recommend_top_n=10):
        """
        ItemCF 推荐模型初始化
        Args:
            config: 全局配置对象（框架要求）
            similarity_top_k: 计算相似物品时取的前K个相似物品
            recommend_top_n: 最终推荐的物品数量
        """
        self.config = config
        self.similarity_top_k = similarity_top_k  # K参数（相似物品数）
        self.recommend_top_n = recommend_top_n    # N参数（推荐数量）
        
        # 模型内部数据结构
        self.item_similarity = defaultdict(dict)  # 物品相似度矩阵 {item: {similar_item: score}}
        self.item_popular = defaultdict(int)       # 物品流行度（被多少用户交互过）
        self.trn_user_items = None                # 训练数据（用户-物品交互字典）

    def fit(self, trn_user_items, train_items=None):
        """
        训练模型（构建物品相似度矩阵）
        Args:
            trn_user_items (dict): 训练数据，格式 {user_id: {item_id1, item_id2, ...}}
            train_items: 框架保留参数（未使用）
        """
        self.trn_user_items = trn_user_items
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
        """
        为指定用户生成推荐（返回带分数的列表）
        Args:
            user_id: 需要推荐的用户ID
        Returns:
            list: 推荐的物品及分数列表，格式 [(item_id, score), ...]（按分数降序）
        """
        if self.trn_user_items is None:
            raise ValueError("模型未训练，请先调用 fit() 方法")
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

        # 保留分数，返回 [(item, score)] 格式（原仅返回 item 列表）
        return sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:self.recommend_top_n]

    def predict_all(self, test_users):
        """
        框架要求的批量预测方法（返回三元组列表）
        Args:
            test_users (list): 需要预测的用户ID列表（如 [user1, user2, ...]）
        Returns:
            list: 预测结果，格式 [(user_id, item_id, score), ...]
        """
        predictions = []
        for user_id in test_users:
            # 获取带分数的推荐列表 [(item, score), ...]
            user_recommendations = self.recommend(user_id)
            # 转换为框架需要的三元组
            for item_id, score in user_recommendations:
                predictions.append((user_id, item_id, score))
        return predictions

    def predict(self, user_id, item_id=None):
        """框架兼容方法（示例返回推荐列表）"""
        return self.recommend(user_id)
