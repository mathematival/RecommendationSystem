import numpy as np
import argparse

# 在顶部添加引用
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

def evaluate_models(config, models_to_run):
    """评估模型性能"""
    print("\n" + "="*50)
    print("模型评估阶段")
    print("="*50)
    
    # 加载数据进行评估
    processor = DataProcessor(config)
    train_users, train_items, test_pairs, all_users, all_items = processor.load_data()
    
    evaluator = Evaluator(config)
    
    # 划分一部分训练数据作为验证集(10%)并从训练集中移除这部分数据
    np.random.seed(config.random_seed)
    validation_data = []
    train_users_holdout = {user_id: [] for user_id in train_users}
    
    for user_id, ratings in train_users.items():
        if len(ratings) > 10:  # 确保用户有足够的评分数据
            # 随机选择10%的数据作为验证集
            validation_indices = np.random.choice(len(ratings), max(1, int(len(ratings) * 0.1)), replace=False)
            validation_indices_set = set(validation_indices)
            
            # 将选中的数据加入验证集
            for idx in validation_indices:
                item_id, rating = ratings[idx]
                validation_data.append((user_id, item_id, rating))
            
            # 将剩余的数据保留在训练集中
            for idx in range(len(ratings)):
                if idx not in validation_indices_set:
                    train_users_holdout[user_id].append(ratings[idx])
        else:
            # 如果用户评分较少，全部保留在训练集中
            train_users_holdout[user_id] = ratings.copy()
    
    # 重新构建物品-用户评分字典
    train_items_holdout = {}
    for user_id, ratings in train_users_holdout.items():
        for item_id, rating in ratings:
            if item_id not in train_items_holdout:
                train_items_holdout[item_id] = []
            train_items_holdout[item_id].append((user_id, rating))
    
    print(f"创建了验证集: {len(validation_data)}条评分")
    print(f"保留训练集: {sum(len(ratings) for ratings in train_users_holdout.values())}条评分")
    
    print("\n验证集评估结果:")
    eval_results = []
    
    for model_config in models_to_run:
        model_class = model_config["class"]
        model_params = model_config.get("params", {})
        model_name = model_class.__name__
        
        print(f"正在评估模型: {model_name}")
        
        # 使用评估器训练和评估模型
        results = evaluator.train_and_evaluate_model(
            model_class, train_users_holdout, train_items_holdout, validation_data, model_params
        )
        
        print(f"模型: {model_name}")
        print(f"  准确性指标:")
        for metric in ["RMSE", "MAE"]:
            if metric in results:
                print(f"    {metric}: {results[metric]:.4f}")
        
        print(f"  性能指标:")
        print(f"    训练时间: {results.get('training_time', 0):.2f}秒")
        print(f"    内存消耗: {results.get('memory_usage', 0):.2f}MB (初始:{results.get('initial_memory', 0):.1f}MB -> 最大:{results.get('max_memory', 0):.1f}MB)")
        
        eval_results.append({
            "model_name": model_name,
            "metrics": results
        })
        
        print("-" * 50)
    
    return eval_results

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='推荐系统实验运行器')
    
    parser.add_argument('--mode', type=str, default='all',
                      choices=['train', 'evaluate', 'all'],
                      help='选择运行模式: train(训练测试), evaluate(模型评估), all(全部)')
    
    parser.add_argument('--model', type=str, default='all',
                      choices=['all', 'global_mean', 'user_mean', 'item_mean', 
                              'biased_baseline', 'usercf', 'itemcf', 'gbdt_lr', 'fm'],
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
        'usercf': {
            "class": UserCFRecommender,
            "config_override": {"result_filename": "usercf_predictions.txt"}
        },
        'itemcf': {
            "class": ItemCFRecommender,
            "config_override": {"result_filename": "itemcf_predictions.txt"}
        },
        'gbdt_lr': {
            "class": GBDTLRRecommender,
            "config_override": {"result_filename": "gbdt_lr_predictions.txt"}
        },
        'fm': {
            "class": FMRecommender,
            "config_override": {"result_filename": "fm_predictions.txt"}
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
        cold_start_strategy="item_mean",  # 冷启动策略
        metrics=["rmse", "mae"]  # 评估指标
    )
    
    # 根据命令行参数选择要运行的模型
    if args.model == 'all':
        models_to_run = [
            get_model_config(model_name) 
            for model_name in ['global_mean', 'user_mean', 'item_mean', 
                             'biased_baseline', 'usercf', 'itemcf', 'gbdt_lr', 'fm']
        ]
    else:
        models_to_run = [get_model_config(args.model)]
    
    # 根据模式选择执行的阶段
    if args.mode in ['train', 'all']:
        # 训练和预测阶段
        print("执行训练测试阶段...")
        training_results = train_and_predict(config, models_to_run)
        
        print("\n训练测试完成！")
    
    if args.mode in ['evaluate', 'all']:
        # 评估阶段
        print("执行模型评估阶段...")
        evaluation_results = evaluate_models(config, models_to_run)
        
        # 显示实验结果摘要
        print("\n" + "="*50)
        print("实验结果总结")
        print("="*50)
        
        for result in evaluation_results:
            print(f"\n模型: {result['model_name']}")
            metrics = result['metrics']
            
            print(f"准确性指标:")
            for metric in ["RMSE", "MAE"]:
                if metric in metrics:
                    print(f"  {metric}: {metrics[metric]:.4f}")
            
            print(f"性能指标:")
            print(f"  训练时间: {metrics.get('training_time', 0):.2f}秒")
            print(f"  内存消耗: {metrics.get('memory_usage', 0):.2f}MB (初始:{metrics.get('initial_memory', 0):.1f}MB -> 最大:{metrics.get('max_memory', 0):.1f}MB)")
            print("-" * 50)

if __name__ == "__main__":
    main()
