## 项目结构

1. 数据集分析：data_analysis.py

2. 实验框架，包括模型基类、评估类等的定义：framework.py

3. 主函数，包括模型训练测试、模型评估等：main.py
(可以通过命令行参数来控制运行模式和模型)

4. 模型实现都放在：/models目录下

## 运行方法

可以通过以下方式使用命令行参数来运行程序：

1. **运行所有阶段和所有模型（默认）**：
```bash
python main.py
```

2. **只执行训练测试阶段**：
```bash
python main.py --mode train
```

3. **只执行模型评估阶段**：
```bash
python main.py --mode evaluate
```

4. **运行特定模型**：
```bash
python main.py --model usercf
python main.py --model gbdt_lr
python main.py --model fm
```

5. **组合模式和模型选择**：
```bash
# 只训练特定模型
python main.py --mode train --model itemcf

# 只评估特定模型
python main.py --mode evaluate --model biased_baseline
```

6. **指定数据路径和其他参数**：
```bash
python main.py --train_path data/my_train.txt --test_path data/my_test.txt --seed 42
```

7. **完整参数示例**：
```bash
python main.py --mode all --model gbdt_lr --train_path ./data/train.txt --test_path ./data/test.txt --result_path ./my_results/ --seed 123
```

## 参考结果

| 模型名称 | RMSE | MAE | 训练时间(秒) | 内存消耗(MB) | 初始内存(MB) | 最大内存(MB) |
|---------|------|-----|-------------|-------------|-------------|-------------|
| GlobalMeanRecommender | 20.4626 | 16.2001 | 0.07 | 0.31 | 1513.3 | 1513.6 |
| UserMeanRecommender | 18.5947 | 14.4972 | 0.07 | 0.00 | 1515.4 | 1515.4 |
| ItemMeanRecommender | 19.1592 | 14.7852 | 0.07 | 0.00 | 1515.4 | 1515.4 |
| BiasedBaselineRecommender | 17.6669 | 13.4715 | 0.13 | 0.00 | 1515.4 | 1515.4 |
| UserCFRecommender | 18.6357 | 14.1562 | 2.46 | 297.45 | 1516.0 | 1813.4 |
| ItemCFRecommender | 17.0674 | 12.9842 | 17.63 | 1050.63 | 1571.9 | 2622.5 |
| GBDTLRRecommender | 17.6893 | 13.3432 | 29.21 | 45.75 | 1531.5 | 1577.3 |
| FMRecommender | 17.5162 | 13.4052 | 14.22 | 0.70 | 1572.6 | 1573.3 |
| BiasSVDRecommender | 17.8221 | 13.8988 | 199.66 | 10.43 | 1573.3 | 1583.7 |
| SVDPlusPlusRecommender | 19.9107 | 15.9565 | 180.89 | 2.39 | 1583.7 | 1586.1 |


