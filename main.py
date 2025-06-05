import numpy as np
import argparse

from framework import *
from models import *

def train_and_predict(config, models_to_run):
    """训练模型并生成预测结果"""
    print("\n" + "="*50)
    print("训练测试阶段")
    print("="*50)
    
    # 创建实验运行器并执行实验
    runner = ExperimentRunner(config)
    results = runner.run_experiments(models_to_run)
    
    return results

def evaluate_models(config, results, models_to_run):
    """评估模型性能"""
    print("\n" + "="*50)
    print("模型评估阶段")
    print("="*50)
    
    # 加载数据进行评估
    processor = DataProcessor(config)
    train_users, train_items, test_pairs, all_users, all_items = processor.load_data()
    
    evaluator = Evaluator(config)
    
    # 划分一部分训练数据作为验证集(10%)
    np.random.seed(config.random_seed)
    validation_data = []
    for user_id, ratings in train_users.items():
        if len(ratings) > 10:  # 确保用户有足够的评分数据
            validation_indices = np.random.choice(len(ratings), max(1, int(len(ratings) * 0.1)), replace=False)
            for idx in validation_indices:
                item_id, rating = ratings[idx]
                validation_data.append((user_id, item_id, rating))
    
    print("\n验证集评估结果:")
    eval_results = []
    
    for result in results:
        model_name = result['model_name']
        for model_config in models_to_run:
            if model_config["class"].__name__ == model_name:
                # 创建并训练模型
                model = model_config["class"](config, **(model_config.get("params", {})))
                model.fit(train_users, train_items)
                
                # 在验证集上评估
                validation_metrics = evaluator.evaluate_model(model, validation_data)
                
                print(f"模型: {model_name}")
                for metric, value in validation_metrics.items():
                    print(f"  {metric}: {value:.4f}")
                
                eval_results.append({
                    "model_name": model_name,
                    "metrics": validation_metrics
                })
    
    return eval_results

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='推荐系统实验运行器')
    
    parser.add_argument('--model', type=str, default='all',
                      choices=['all', 'global_mean', 'user_mean', 'item_mean', 
                              'biased_baseline', 'gbdt_lr'],
                      help='选择要运行的推荐模型')
    
    parser.add_argument('--train_path', type=str, default='./data/train.txt',
                      help='训练数据路径')
    
    parser.add_argument('--test_path', type=str, default='./data/test.txt',
                      help='测试数据路径')
    
    parser.add_argument('--result_path', type=str, default='./results/',
                      help='结果保存路径')
    
    parser.add_argument('--seed', type=int, default=42,
                      help='随机数种子')
    
    return parser.parse_args()

def get_model_config(model_name):
    """根据模型名称返回模型配置"""
    model_configs = {
        'global_mean': {
            "class": GlobalMeanRecommender,
            "config_override": {"result_filename": "global_mean_predictions.txt"}
        },
        'user_mean': {
            "class": UserMeanRecommender,
            "config_override": {"result_filename": "user_mean_predictions.txt"}
        },
        'item_mean': {
            "class": ItemMeanRecommender,
            "config_override": {"result_filename": "item_mean_predictions.txt"}
        },
        'biased_baseline': {
            "class": BiasedBaselineRecommender,
            "config_override": {"result_filename": "biased_baseline_predictions.txt"}
        },
        'gbdt_lr': {
            "class": GBDTLRRecommender,
            "config_override": {"result_filename": "gbdt_lr_predictions.txt"}
        }
    }
    return model_configs.get(model_name)

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 创建配置
    config = ExperimentConfig(
        random_seed=args.seed,
        train_path=args.train_path,
        test_path=args.test_path,
        result_path=args.result_path,
        normalize_ratings=True,  # 启用评分归一化
        cold_start_strategy="item_mean",  # 冷启动策略
        metrics=["rmse", "mae"]  # 评估指标
    )
    
    # 根据命令行参数选择要运行的模型
    if args.model == 'all':
        models_to_run = [
            get_model_config(model_name) 
            for model_name in ['global_mean', 'user_mean', 'item_mean', 
                             'biased_baseline', 'gbdt_lr']
        ]
    else:
        models_to_run = [get_model_config(args.model)]
    
    # 训练和预测阶段
    training_results = train_and_predict(config, models_to_run)
    
    # 评估阶段
    evaluation_results = evaluate_models(config, training_results, models_to_run)
    
    # 显示实验结果摘要
    print("\n" + "="*50)
    print("实验结果总结")
    print("="*50)
    
    for result in evaluation_results:
        print(f"模型: {result['model_name']}")
        for metric, value in result['metrics'].items():
            print(f"  {metric}: {value:.4f}")
        print("-" * 30)

if __name__ == "__main__":
    main()
