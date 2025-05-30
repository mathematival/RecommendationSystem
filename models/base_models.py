import numpy as np
from typing import Dict, List, Tuple, Set, Callable, Any, Optional, Union

from framework import BaseRecommender

# 基线模型实现
class GlobalMeanRecommender(BaseRecommender):
    """全局平均评分基线模型"""
    
    def predict(self, user_id: int, item_id: int) -> float:
        """使用全局平均评分进行预测"""
        return self.global_mean


class UserMeanRecommender(BaseRecommender):
    """用户平均评分基线模型"""
    
    def predict(self, user_id: int, item_id: int) -> float:
        """使用用户平均评分进行预测"""
        return self.user_means.get(user_id, self.global_mean)


class ItemMeanRecommender(BaseRecommender):
    """物品平均评分基线模型"""
    
    def predict(self, user_id: int, item_id: int) -> float:
        """使用物品平均评分进行预测"""
        return self.item_means.get(item_id, self.global_mean)


class BiasedBaselineRecommender(BaseRecommender):
    """结合全局均值、用户偏置和物品偏置的基线模型"""
    
    def fit(self, train_users: Dict, train_items: Dict) -> None:
        """训练模型，计算用户和物品偏置"""
        super().fit(train_users, train_items)
        
        # 计算用户和物品的偏置
        self.user_biases = {}
        self.item_biases = {}
        
        for user_id, ratings in train_users.items():
            if ratings:
                self.user_biases[user_id] = self.user_means[user_id] - self.global_mean
            else:
                self.user_biases[user_id] = 0
                
        for item_id, users in train_items.items():
            if users:
                # 先减去全局均值和用户偏置后的平均残差
                residuals = []
                for user_id, rating in users:
                    if user_id in self.user_biases:
                        residuals.append(rating - self.global_mean - self.user_biases[user_id])
                    else:
                        residuals.append(rating - self.global_mean)
                
                if residuals:
                    self.item_biases[item_id] = np.mean(residuals)
                else:
                    self.item_biases[item_id] = 0
            else:
                self.item_biases[item_id] = 0
    
    def predict(self, user_id: int, item_id: int) -> float:
        """使用全局均值+用户偏置+物品偏置进行预测"""
        user_bias = self.user_biases.get(user_id, 0)
        item_bias = self.item_biases.get(item_id, 0)
        
        return self.global_mean + user_bias + item_bias
