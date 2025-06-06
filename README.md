## 项目结构

1. 数据集分析：data_analysis.py

2. 实验框架，包括模型基类、评估类等的定义：framework.py

3. 主函数，包括模型训练测试、模型评估等：main.py
（这里是统一测试评估了所有模型，如果只想测试单个模型，可以查看framework.py中的run_experiment函数，注意不是run_experiments函数）

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

1. 运行所有模型（默认）：
```bash
python main.py
```

2. 只运行特定模型：
```bash
python main.py --model gbdt_lr
```

3. 指定其他参数：
```bash
python main.py --model gbdt_lr --train_path data/my_train.txt --test_path data/my_test.txt --seed 42
```

## 参考结果

| 模型 | RMSE | MAE | 训练时间(秒) |
|------|------|-----|--------------|
| GlobalMeanRecommender | 20.5654 | 16.2754 | 0.19 |
| UserMeanRecommender | 18.4579 | 14.3839 | 0.19 |
| ItemMeanRecommender | 17.3269 | 13.1534 | 0.19 |
| BiasedBaselineRecommender | 15.8856 | 11.8685 | 0.27 |
| GBDTLRRecommender | 14.7330 | 11.0346 | 34.94 |
| FMRecommender | 15.9564 | 12.1132 | 17.19 |


