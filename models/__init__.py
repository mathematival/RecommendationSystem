"""
推荐系统模型包

此包包含所有可用的推荐系统模型，包括基础模型和高级模型。
"""

# 导入所有基线模型
from models.base_models import (
    GlobalMeanRecommender,
    UserMeanRecommender,
    ItemMeanRecommender,
    BiasedBaselineRecommender
)

# 导入协同过滤模型
from models.itemcf import ItemCFRecommender
from models.usercf import UserCFRecommender

# 导入进阶模型
from models.gbdt_lr import GBDTLRRecommender
from models.fm import FMRecommender

# 导入矩阵分解模型
from models.bias_svd import BiasSVDRecommender

# 导出所有模型，这样就可以通过 from models import * 导入所有模型
__all__ = [
    # 基线模型
    'GlobalMeanRecommender',
    'UserMeanRecommender',
    'ItemMeanRecommender',
    'BiasedBaselineRecommender',
    
    # 协同过滤模型
    'ItemCFRecommender',
    'UserCFRecommender',
    
    # 进阶模型
    'GBDTLRRecommender',
    'FMRecommender',
    
    # 矩阵分解模型
    'BiasSVDRecommender',
]
