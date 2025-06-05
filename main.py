import numpy as np

from framework import *
from models.base_models import *
# 导入GBDT+LR模型
from models.gbdt_lr import GBDTLRRecommender

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
                    "metrics": validation_metrics,
                    "runtime": result['runtime']
                })
    
    return eval_results

def main():
    # 创建默认配置
    config = ExperimentConfig(
        random_seed=42,
        train_path="./data/train.txt",
        test_path="./data/test.txt",
        result_path="./results/",
        normalize_ratings=True,  # 启用评分归一化
        cold_start_strategy="item_mean",  # 冷启动策略
        metrics=["rmse", "mae"]  # 评估指标
    )
    
    # 定义要运行的模型
    models_to_run = [
        {
            "class": GlobalMeanRecommender,
            "config_override": {"result_filename": "global_mean_predictions.txt"}
        },
        {
            "class": UserMeanRecommender,
            "config_override": {"result_filename": "user_mean_predictions.txt"}
        },
        {
            "class": ItemMeanRecommender, 
            "config_override": {"result_filename": "item_mean_predictions.txt"}
        },
        {
            "class": BiasedBaselineRecommender,
            "config_override": {"result_filename": "biased_baseline_predictions.txt"}
        },
        {
            "class": GBDTLRRecommender,
            "config_override": {"result_filename": "gbdt_lr_predictions.txt"}
        }
    ]
    
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
        print(f"  运行时间: {result['runtime']:.2f}秒")
        print("-" * 30)

if __name__ == "__main__":
    main()
