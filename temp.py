import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from data_analysis import load_training_data, load_test_data, compute_statistics, print_statistics

# 设置中文字体以避免字体警告
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def analyze_rating_distribution(train_users):
    """
    分析评分分布特征
    """
    # 收集所有评分
    all_ratings = []
    for user_id, ratings in train_users.items():
        all_ratings.extend([r for _, r in ratings])
    
    all_ratings = np.array(all_ratings)
    
    print("\n" + "="*50)
    print("评分分布分析")
    print("="*50)
    
    # 基本统计
    print(f"评分总数: {len(all_ratings)}")
    print(f"平均评分: {np.mean(all_ratings):.4f}")
    print(f"标准差: {np.std(all_ratings):.4f}")
    print(f"最小值: {np.min(all_ratings):.4f}")
    print(f"最大值: {np.max(all_ratings):.4f}")
    print(f"偏度: {stats.skew(all_ratings):.4f}")
    print(f"峰度: {stats.kurtosis(all_ratings):.4f}")
    
    # 评分分布统计
    unique_ratings, counts = np.unique(all_ratings, return_counts=True)
    print(f"\n评分值分布:")
    for rating, count in zip(unique_ratings, counts):
        percentage = count / len(all_ratings) * 100
        print(f"评分 {rating}: {count} 次 ({percentage:.2f}%)")
    
    return all_ratings, unique_ratings, counts

def analyze_user_item_patterns(train_users, train_items):
    """
    分析用户和物品的评分模式
    """
    # 用户评分数分布
    user_rating_counts = [len(ratings) for ratings in train_users.values()]
    
    # 物品被评分数分布
    item_rating_counts = [len(users) for users in train_items.values()]
    
    print("\n" + "="*50)
    print("用户和物品评分模式分析")
    print("="*50)
    
    print(f"用户评分数统计:")
    print(f"  平均每用户评分数: {np.mean(user_rating_counts):.2f}")
    print(f"  标准差: {np.std(user_rating_counts):.2f}")
    print(f"  最少评分用户: {np.min(user_rating_counts)} 个评分")
    print(f"  最多评分用户: {np.max(user_rating_counts)} 个评分")
    
    print(f"\n物品被评分数统计:")
    print(f"  平均每物品被评分数: {np.mean(item_rating_counts):.2f}")
    print(f"  标准差: {np.std(item_rating_counts):.2f}")
    print(f"  最少被评分物品: {np.min(item_rating_counts)} 次")
    print(f"  最多被评分物品: {np.max(item_rating_counts)} 次")
    
    return user_rating_counts, item_rating_counts

def plot_distributions(all_ratings, user_rating_counts, item_rating_counts):
    """
    可视化各种分布
    """
    plt.figure(figsize=(16, 12))
    
    # 评分分布
    plt.subplot(2, 3, 1)
    plt.hist(all_ratings, bins=20, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Rating Distribution')
    plt.xlabel('Rating Value')
    plt.ylabel('Density')
    
    # 评分分布条形图
    plt.subplot(2, 3, 2)
    unique_ratings, counts = np.unique(all_ratings, return_counts=True)
    plt.bar(unique_ratings, counts / len(all_ratings), alpha=0.7, color='lightgreen')
    plt.title('Rating Value Distribution')
    plt.xlabel('Rating Value')
    plt.ylabel('Proportion')
    
    # 用户评分数分布
    plt.subplot(2, 3, 3)
    plt.hist(user_rating_counts, bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.title('User Rating Count Distribution')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Users')
    plt.yscale('log')
    
    # 物品被评分数分布
    plt.subplot(2, 3, 4)
    plt.hist(item_rating_counts, bins=30, alpha=0.7, color='red', edgecolor='black')
    plt.title('Item Rating Count Distribution')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Items')
    plt.yscale('log')
    
    # 用户评分数分布（对数尺度）
    plt.subplot(2, 3, 5)
    plt.hist(user_rating_counts, bins=50, alpha=0.7, color='purple', edgecolor='black')
    plt.title('User Activity (Log-Log)')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Users')
    plt.xscale('log')
    plt.yscale('log')
    
    # 物品被评分数分布（对数尺度）
    plt.subplot(2, 3, 6)
    plt.hist(item_rating_counts, bins=50, alpha=0.7, color='brown', edgecolor='black')
    plt.title('Item Popularity (Log-Log)')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Items')
    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()

def test_power_law_distribution(data, name):
    """
    测试是否符合幂律分布
    """
    print(f"\n{name}幂律分布测试:")
    
    # 计算度分布
    unique_vals, counts = np.unique(data, return_counts=True)
    
    # 过滤掉count为0的值
    mask = counts > 0
    unique_vals = unique_vals[mask]
    counts = counts[mask]
    
    # 对数变换测试线性关系
    log_vals = np.log(unique_vals)
    log_counts = np.log(counts)
    
    # 线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_vals, log_counts)
    
    print(f"  线性回归结果 (log-log):")
    print(f"    斜率 (α): {-slope:.4f}")
    print(f"    R²: {r_value**2:.4f}")
    print(f"    p值: {p_value:.4f}")
    
    # 如果R²>0.8且p<0.05，可能符合幂律分布
    if r_value**2 > 0.8 and p_value < 0.05:
        print(f"  可能符合幂律分布，指数α ≈ {-slope:.2f}")
    else:
        print(f"  不太符合幂律分布")

def analyze_distribution_characteristics(all_ratings, user_rating_counts, item_rating_counts):
    """
    深入分析分布特征
    """
    print("\n" + "="*50)
    print("数据集分布特征详细分析")
    print("="*50)
    
    # 1. 评分分布分析
    print("1. 评分分布特征:")
    unique_ratings = np.unique(all_ratings)
    print(f"   - 评分范围: {np.min(all_ratings):.0f} - {np.max(all_ratings):.0f}")
    print(f"   - 评分值: {[int(x) for x in sorted(unique_ratings)]}")
    print(f"   - 评分类型: 离散型（10个等级，10-100分制)")
    
    # 检查评分分布的对称性
    rating_skew = stats.skew(all_ratings)
    if rating_skew < -0.5:
        print(f"   - 分布形状: 左偏分布（负偏度 = {rating_skew:.4f}）")
        print(f"   - 特征: 用户倾向于给出较高评分")
    elif rating_skew > 0.5:
        print(f"   - 分布形状: 右偏分布（正偏度 = {rating_skew:.4f}）")
        print(f"   - 特征: 用户倾向于给出较低评分")
    else:
        print(f"   - 分布形状: 近似对称分布（偏度 = {rating_skew:.4f}）")
    
    # 峰度分析
    rating_kurtosis = stats.kurtosis(all_ratings)
    if rating_kurtosis > 0:
        print(f"   - 峰度特征: 尖峰分布（峰度 = {rating_kurtosis:.4f}）")
    else:
        print(f"   - 峰度特征: 平坦分布（峰度 = {rating_kurtosis:.4f}）")
    
    # 2. 用户活跃度分布
    print(f"\n2. 用户活跃度分布:")
    user_skew = stats.skew(user_rating_counts)
    print(f"   - 偏度: {user_skew:.4f}")
    print(f"   - 分布类型: 高度右偏分布（长尾分布）")
    print(f"   - 80-20法则验证:")
    
    sorted_users = sorted(user_rating_counts, reverse=True)
    total_ratings = sum(sorted_users)
    top_20_percent = int(0.2 * len(sorted_users))
    top_20_ratings = sum(sorted_users[:top_20_percent])
    percentage = (top_20_ratings / total_ratings) * 100
    print(f"     * 最活跃的20%用户贡献了 {percentage:.1f}% 的评分")
    
    # 3. 物品流行度分布
    print(f"\n3. 物品流行度分布:")
    item_skew = stats.skew(item_rating_counts)
    print(f"   - 偏度: {item_skew:.4f}")
    print(f"   - 分布类型: 高度右偏分布（长尾分布）")
    print(f"   - 流行度集中度:")
    
    sorted_items = sorted(item_rating_counts, reverse=True)
    total_item_ratings = sum(sorted_items)
    top_20_percent_items = int(0.2 * len(sorted_items))
    top_20_item_ratings = sum(sorted_items[:top_20_percent_items])
    item_percentage = (top_20_item_ratings / total_item_ratings) * 100
    print(f"     * 最受欢迎的20%物品获得了 {item_percentage:.1f}% 的评分")
    
    # 4. 数据集稀疏性分析
    print(f"\n4. 数据集稀疏性:")
    num_users = len(user_rating_counts)
    num_items = len(item_rating_counts)
    total_possible = num_users * num_items
    actual_ratings = sum(user_rating_counts)
    sparsity = 1 - (actual_ratings / total_possible)
    print(f"   - 稀疏度: {sparsity:.4f} ({sparsity*100:.2f}%)")
    print(f"   - 实际评分数: {actual_ratings:,}")
    print(f"   - 可能的最大评分数: {total_possible:,}")

def main():
    train_path = "./data/train.txt"
    test_path = "./data/test.txt"
    
    print("正在加载数据...")
    train_users, train_items = load_training_data(train_path)
    test_pairs = load_test_data(test_path)
    
    # 基本统计
    basic_stats = compute_statistics(train_users, train_items, test_pairs)
    print_statistics(basic_stats)
    
    # 评分分布分析
    all_ratings, unique_ratings, counts = analyze_rating_distribution(train_users)
    
    # 用户物品模式分析
    user_rating_counts, item_rating_counts = analyze_user_item_patterns(train_users, train_items)
    
    # 可视化
    plot_distributions(all_ratings, user_rating_counts, item_rating_counts)
    
    # 幂律分布测试
    test_power_law_distribution(user_rating_counts, "用户评分数")
    test_power_law_distribution(item_rating_counts, "物品被评分数")
    
    # 详细分布特征分析
    analyze_distribution_characteristics(all_ratings, user_rating_counts, item_rating_counts)

if __name__ == "__main__":
    main()