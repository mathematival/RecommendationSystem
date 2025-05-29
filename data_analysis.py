import numpy as np
from collections import defaultdict

def load_training_data(file_path):
    """
    加载训练数据集，并返回用户评分和物品评分信息
    """
    users_data = {}  # 存储用户评分信息
    items_data = {}  # 存储物品被评分信息
    
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
            i = 0
            while i < len(lines):
                # 解析用户行
                user_info = lines[i].strip().split('|')
                user_id = int(user_info[0])
                num_ratings = int(user_info[1])
                
                if user_id not in users_data:
                    users_data[user_id] = []
                
                # 读取该用户的所有评分
                for j in range(1, num_ratings + 1):
                    if i + j >= len(lines):
                        break
                        
                    item_data = lines[i + j].strip().split()
                    item_id = int(float(item_data[0]))
                    rating = float(item_data[1])
                    
                    # 添加到用户评分列表
                    users_data[user_id].append((item_id, rating))
                    
                    # 添加到物品被评分列表
                    if item_id not in items_data:
                        items_data[item_id] = []
                    items_data[item_id].append((user_id, rating))
                
                i += num_ratings + 1
                
        return users_data, items_data
    
    except Exception as e:
        print(f"读取数据时出错: {e}")
        return {}, {}

def load_test_data(file_path):
    """
    加载测试数据集，返回待预测的用户-物品对
    """
    test_pairs = []
    
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
            i = 0
            while i < len(lines):
                # 解析用户行
                user_info = lines[i].strip().split('|')
                user_id = int(user_info[0])
                num_items = int(user_info[1])
                
                # 读取该用户的所有待预测物品
                for j in range(1, num_items + 1):
                    if i + j >= len(lines):
                        break
                        
                    item_id = int(float(lines[i + j].strip()))
                    test_pairs.append((user_id, item_id))
                
                i += num_items + 1
                
        return test_pairs
    
    except Exception as e:
        print(f"读取测试数据时出错: {e}")
        return []

def compute_statistics(train_users, train_items, test_pairs):
    """
    计算数据集的基本统计信息
    """
    # 训练集统计
    num_users = len(train_users)
    num_items = len(train_items)
    
    all_ratings = []
    for user_id, ratings in train_users.items():
        all_ratings.extend([r for _, r in ratings])
    
    total_ratings = len(all_ratings)
    avg_rating = np.mean(all_ratings) if all_ratings else 0
    min_rating = min(all_ratings) if all_ratings else 0
    max_rating = max(all_ratings) if all_ratings else 0
    
    # 用户评分情况
    ratings_per_user = [len(ratings) for ratings in train_users.values()]
    avg_ratings_per_user = np.mean(ratings_per_user)
    min_ratings_per_user = min(ratings_per_user) if ratings_per_user else 0
    max_ratings_per_user = max(ratings_per_user) if ratings_per_user else 0
    
    # 物品被评分情况
    ratings_per_item = [len(users) for users in train_items.values()]
    avg_ratings_per_item = np.mean(ratings_per_item)
    min_ratings_per_item = min(ratings_per_item) if ratings_per_item else 0
    max_ratings_per_item = max(ratings_per_item) if ratings_per_item else 0
    
    # 测试集统计
    test_users = set([user for user, _ in test_pairs])
    test_items = set([item for _, item in test_pairs])
    
    # 数据集稀疏度计算
    sparsity = 1.0 - (total_ratings / (num_users * num_items))
    
    return {
        "训练集用户数": num_users,
        "训练集物品数": num_items,
        "总评分数": total_ratings,
        "平均评分": avg_rating,
        "最低评分": min_rating,
        "最高评分": max_rating,
        "每用户平均评分数": avg_ratings_per_user,
        "单用户最少评分数": min_ratings_per_user,
        "单用户最多评分数": max_ratings_per_user,
        "每物品平均被评分数": avg_ratings_per_item,
        "单物品最少被评分数": min_ratings_per_item,
        "单物品最多被评分数": max_ratings_per_item,
        "测试集用户数": len(test_users),
        "测试集物品数": len(test_items),
        "测试集待预测数": len(test_pairs),
        "数据集稀疏度": sparsity
    }

def print_statistics(stats):
    """
    打印统计信息
    """
    print("=" * 50)
    print("推荐系统数据集基本统计信息")
    print("=" * 50)
    
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

def main():
    train_path = "./data/train.txt"
    test_path = "./data/test.txt"
    
    print("正在加载训练数据...")
    train_users, train_items = load_training_data(train_path)
    
    print("正在加载测试数据...")
    test_pairs = load_test_data(test_path)
    
    print("计算统计信息...")
    stats = compute_statistics(train_users, train_items, test_pairs)
    
    print_statistics(stats)

if __name__ == "__main__":
    main()
