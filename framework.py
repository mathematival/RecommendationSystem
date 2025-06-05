import numpy as np
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Callable, Any, Optional, Union
from tqdm import tqdm

# 复用已有的数据加载函数
from data_analysis import load_training_data, load_test_data

class ExperimentConfig:
    """统一管理实验配置"""
    def __init__(self, 
                 random_seed: int = 42,
                 train_path: str = "./data/train.txt",
                 test_path: str = "./data/test.txt",
                 result_path: str = "./results/",
                 result_filename: str = "predictions.txt",
                 rating_min: float = 10.0,
                 rating_max: float = 100.0,
                 normalize_ratings: bool = False,
                 metrics: List[str] = ["rmse", "mae"],
                 cold_start_strategy: str = "item_mean"):
        
        self.random_seed = random_seed
        self.train_path = train_path
        self.test_path = test_path
        self.result_path = result_path
        self.result_filename = result_filename
        self.rating_min = rating_min
        self.rating_max = rating_max
        self.normalize_ratings = normalize_ratings
        self.metrics = metrics
        self.cold_start_strategy = cold_start_strategy
        
        # 设置随机种子确保结果可复现
        np.random.seed(self.random_seed)
        
        # 确保结果目录存在
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
    
    @property
    def result_file_path(self) -> str:
        return os.path.join(self.result_path, self.result_filename)


class BaseRecommender:
    """所有推荐模型的基类，定义统一接口"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.global_mean = None
        self.user_means = {}
        self.item_means = {}
        self.train_users = None
        self.train_items = None
        self.all_users = set()
        self.all_items = set()
        
    def fit(self, train_users: Dict, train_items: Dict) -> None:
        """训练模型"""
        self.train_users = train_users
        self.train_items = train_items
        
        # 计算全局平均值
        all_ratings = []
        for user_id, ratings in train_users.items():
            all_ratings.extend([r for _, r in ratings])
        self.global_mean = np.mean(all_ratings) if all_ratings else 0
        
        # 计算用户平均评分
        for user_id, ratings in train_users.items():
            if ratings:
                self.user_means[user_id] = np.mean([r for _, r in ratings])
            else:
                self.user_means[user_id] = self.global_mean
        
        # 计算物品平均评分
        for item_id, users in train_items.items():
            if users:
                self.item_means[item_id] = np.mean([r for _, r in users])
            else:
                self.item_means[item_id] = self.global_mean
                
        # 收集所有用户和物品ID
        self.all_users = set(train_users.keys())
        self.all_items = set(train_items.keys())
        
    def predict(self, user_id: int, item_id: int) -> float:
        """预测用户对物品的评分，需要在子类中实现具体算法"""
        raise NotImplementedError("在子类中实现具体预测算法")
    
    def predict_for_user(self, user_id: int, item_id: int) -> float:
        """带有冷启动处理的统一预测函数"""
        # 确保评分被限制在有效范围内
        if user_id in self.all_users:
            # 正常预测
            pred = self.predict(user_id, item_id)
        else:
            # 冷启动用户处理
            if self.config.cold_start_strategy == "item_mean":
                pred = self.item_means.get(item_id, self.global_mean)
            elif self.config.cold_start_strategy == "global_mean":
                pred = self.global_mean
            elif self.config.cold_start_strategy == "popular_items":
                # 根据物品热度推荐，热度越高评分越接近该物品平均分
                if item_id in self.train_items:
                    popularity = len(self.train_items[item_id])
                    max_pop = max(len(users) for users in self.train_items.values())
                    weight = popularity / max_pop
                    pred = weight * self.item_means.get(item_id, self.global_mean) + (1 - weight) * self.global_mean
                else:
                    pred = self.global_mean
            else:
                pred = self.global_mean
        
        # 确保预测值在评分范围内
        return max(self.config.rating_min, min(self.config.rating_max, pred))
    
    def predict_all(self, test_pairs: List[Tuple[int, int]]) -> List[Tuple[int, int, float]]:
        """预测所有测试集中的评分"""
        return [(user, item, self.predict_for_user(user, item)) 
                for user, item in tqdm(test_pairs, desc="预测评分", unit="对")]


class DataProcessor:
    """统一的数据加载和预处理模块"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
    def load_data(self) -> Tuple[Dict, Dict, List[Tuple[int, int]], Set[int], Set[int]]:
        """加载训练和测试数据"""
        print("正在加载训练数据...")
        train_users, train_items = load_training_data(self.config.train_path)
        
        print("正在加载测试数据...")
        test_pairs = load_test_data(self.config.test_path)
        
        # 收集所有用户和物品ID
        all_users = set(train_users.keys()) | set(user for user, _ in test_pairs)
        all_items = set(train_items.keys()) | set(item for _, item in test_pairs)
        
        return train_users, train_items, test_pairs, all_users, all_items
    
    def normalize_ratings(self, train_users: Dict) -> Dict:
        """对用户评分进行归一化处理"""
        if not self.config.normalize_ratings:
            return train_users
            
        normalized_users = {}
        
        for user_id, ratings in train_users.items():
            if not ratings:
                normalized_users[user_id] = []
                continue
                
            # 获取该用户的评分范围
            user_ratings = [r for _, r in ratings]
            min_rating = min(user_ratings)
            max_rating = max(user_ratings)
            rating_range = max_rating - min_rating
            
            if rating_range == 0:
                # 如果用户所有评分相同，则保持不变
                normalized_users[user_id] = ratings.copy()
            else:
                # 归一化到[rating_min, rating_max]区间
                normalized = []
                for item_id, rating in ratings:
                    norm_rating = ((rating - min_rating) / rating_range) * \
                                 (self.config.rating_max - self.config.rating_min) + \
                                 self.config.rating_min
                    normalized.append((item_id, norm_rating))
                normalized_users[user_id] = normalized
                
        return normalized_users
    
    def convert_to_item_ratings(self, user_ratings: Dict) -> Dict:
        """将用户评分转换为物品评分格式"""
        item_ratings = {}
        
        for user_id, ratings in user_ratings.items():
            for item_id, rating in ratings:
                if item_id not in item_ratings:
                    item_ratings[item_id] = []
                item_ratings[item_id].append((user_id, rating))
                
        return item_ratings


class Evaluator:
    """统一的模型评估模块"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
    def calculate_rmse(self, true_ratings: List[Tuple[int, int, float]], 
                      pred_ratings: List[Tuple[int, int, float]]) -> float:
        """计算均方根误差(RMSE)"""
        # 将预测结果转换为字典以便查找
        pred_dict = {(user, item): rating for user, item, rating in pred_ratings}
        
        squared_errors = []
        for user, item, true_rating in true_ratings:
            if (user, item) in pred_dict:
                error = true_rating - pred_dict[(user, item)]
                squared_errors.append(error ** 2)
        
        if not squared_errors:
            return float('inf')
            
        return np.sqrt(np.mean(squared_errors))
    
    def calculate_mae(self, true_ratings: List[Tuple[int, int, float]], 
                     pred_ratings: List[Tuple[int, int, float]]) -> float:
        """计算平均绝对误差(MAE)"""
        # 将预测结果转换为字典以便查找
        pred_dict = {(user, item): rating for user, item, rating in pred_ratings}
        
        abs_errors = []
        for user, item, true_rating in true_ratings:
            if (user, item) in pred_dict:
                error = abs(true_rating - pred_dict[(user, item)])
                abs_errors.append(error)
        
        if not abs_errors:
            return float('inf')
            
        return np.mean(abs_errors)
    
    def evaluate_model(self, model: BaseRecommender, test_data: List[Tuple[int, int, float]]) -> Dict[str, float]:
        """评估模型性能"""
        # 分离测试数据中的用户-物品对和真实评分
        test_pairs = [(user, item) for user, item, _ in test_data]
        test_ratings = test_data
        
        # 获取预测评分
        pred_ratings = model.predict_all(test_pairs)
        
        # 计算各项评估指标
        results = {}
        if "rmse" in self.config.metrics:
            results["RMSE"] = self.calculate_rmse(test_ratings, pred_ratings)
        if "mae" in self.config.metrics:
            results["MAE"] = self.calculate_mae(test_ratings, pred_ratings)
        # 后续可以添加更多指标，暂时删除覆盖率和新颖度指标

        return results


def save_predictions(predictions: List[Tuple[int, int, float]], output_path: str) -> None:
    """按照ResultForm.txt的格式保存预测结果"""
    # 按用户分组预测结果
    user_predictions = defaultdict(list)
    for user_id, item_id, pred_rating in predictions:
        user_predictions[user_id].append((item_id, pred_rating))
    
    with open(output_path, 'w') as f:
        for user_id, items in user_predictions.items():
            # 写入用户行
            f.write(f"{user_id}|{len(items)}\n")
            # 写入物品评分行
            for item_id, rating in items:
                f.write(f"{item_id}  {rating:.1f}\n")


class ExperimentRunner:
    """实验运行管理器"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data_processor = DataProcessor(config)
        self.evaluator = Evaluator(config)
        
    def run_experiment(self, model_class, model_params=None) -> Dict:
        """运行单个实验"""
        if model_params is None:
            model_params = {}
            
        start_time = time.time()
        
        # 加载数据
        train_users, train_items, test_pairs, all_users, all_items = self.data_processor.load_data()
        
        # 如果需要，进行评分归一化
        if self.config.normalize_ratings:
            train_users = self.data_processor.normalize_ratings(train_users)
            # 重新构建物品评分数据
            train_items = self.data_processor.convert_to_item_ratings(train_users)
        
        # 创建并训练模型
        model = model_class(self.config, **model_params)
        print(f"开始训练 {model.__class__.__name__}...")
        model.fit(train_users, train_items)
        
        # 预测测试集
        print("开始预测...")
        predictions = model.predict_all(test_pairs)
        
        # 保存预测结果
        save_predictions(predictions, self.config.result_file_path)
        
        # 计算运行时间
        runtime = time.time() - start_time
        
        return {
            "model_name": model.__class__.__name__,
            "runtime": runtime,
            "num_predictions": len(predictions),
            "result_file": self.config.result_file_path
        }
        
    def run_experiments(self, models_config: List[Dict]) -> List[Dict]:
        """运行多个实验并比较结果"""
        results = []
        
        for model_config in models_config:
            model_class = model_config["class"]
            model_params = model_config.get("params", {})
            config_override = model_config.get("config_override", {})
            
            # 创建特定于此模型的配置
            model_specific_config = ExperimentConfig(**{**self.config.__dict__, **config_override})
            
            # 创建一个实验运行器
            runner = ExperimentRunner(model_specific_config)
            
            # 运行实验
            print(f"\n{'-'*50}")
            print(f"运行模型: {model_class.__name__}")
            print(f"{'-'*50}")
            
            result = runner.run_experiment(model_class, model_params)
            results.append(result)
            
            print(f"完成! 结果保存在: {result['result_file']}")
            print(f"运行时间: {result['runtime']:.2f} 秒")
            
        return results
