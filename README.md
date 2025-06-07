## 项目结构

1. 数据集分析：data_analysis.py

2. 实验框架，包括模型基类、评估类等的定义：framework.py

3. 主函数，包括模型训练测试、模型评估等：main.py
(可以通过命令行参数来控制运行模式和模型)

4. 后续实验的模型实现都放在：/models目录下

5. 暂时的指导手册：temp.pdf

## 备注

这次实验好像不需要单文件，我打算统一数据处理、模型训练测试、模型评估过程；不同算法实现只有模型本身的区别

训练测试老师提供了数据集和相应的格式要求，但是不能确定测试结果好不好；模型评估是将训练集划分一些出来，做评估，所以也不能保证准不准确，仅供参考

不能保证代码的正确性，所以你得检查检查😂

可供参考：

https://datawhalechina.github.io/fun-rec/#/

lxm学长的代码报告

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

| 模型名称                  | RMSE    | MAE     | 训练时间(秒) | 内存消耗(MB) | 初始内存(MB) | 最大内存(MB) |
|--------------------------|---------|---------|-------------|-------------|-------------|-------------|
| GlobalMeanRecommender    | 20.4626 | 16.2001 | 0.07        | 2.09        | 541.8       | 543.9       |
| UserMeanRecommender      | 18.5947 | 14.4972 | 0.07        | 0.06        | 546.2       | 546.2       |
| ItemMeanRecommender      | 19.1592 | 14.7852 | 0.06        | 0.00        | 546.2       | 546.2       |
| BiasedBaselineRecommender| 17.6669 | 13.4715 | 0.12        | 0.00        | 547.0       | 547.0       |
| UserCFRecommender        | 18.6357 | 14.1562 | 2.49        | 276.07      | 547.3       | 823.4       |
| ItemCFRecommender        | 17.0674 | 12.9842 | 17.52       | 1054.65     | 604.0       | 1658.6      |
| GBDTLRRecommender        | 17.6893 | 13.3432 | 33.08       | 57.75       | 587.9       | 645.6       |
| FMRecommender            | 17.5162 | 13.4052 | 15.42       | 680.07      | 636.9       | 1317.0      |
| BiasSVDRecommender       | 17.8221 | 13.8988 | 176.93      | 198.16      | 1317.1      | 1515.3      |


