import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np

from .base_recommender import BaseRecommender # 假设您的 BaseRecommender 在同级目录的 base_recommender.py 中

class BiasSVDRecommender(BaseRecommender):
    def __init__(self, config, num_users, num_items, num_factors=50, learning_rate=0.005, regularization=0.02, num_epochs=20, batch_size=1024, device=None):
        super().__init__(config)
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.lr = learning_rate
        self.reg = regularization # L2 正则化系数
        self.epochs = num_epochs
        self.batch_size = batch_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.user_to_idx = None
        self.item_to_idx = None

        self._build_model()

    def _build_model(self):
        """构建SVD模型的PyTorch组件"""
        self.user_factors = nn.Embedding(self.num_users, self.num_factors).to(self.device)
        self.item_factors = nn.Embedding(self.num_items, self.num_factors).to(self.device)
        self.user_biases = nn.Embedding(self.num_users, 1).to(self.device)
        self.item_biases = nn.Embedding(self.num_items, 1).to(self.device)
        self.global_bias = nn.Parameter(torch.tensor([0.0], device=self.device))

        # 初始化权重
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)
        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.item_biases.weight)

    def _forward_pass(self, user_indices, item_indices):
        """
        前向传播计算预测评分
        user_indices: (batch_size,) tensor of user indices
        item_indices: (batch_size,) tensor of item indices
        """
        user_vec = self.user_factors(user_indices)    # (batch_size, num_factors)
        item_vec = self.item_factors(item_indices)    # (batch_size, num_factors)
        
        ub = self.user_biases(user_indices).squeeze() # (batch_size,)
        ib = self.item_biases(item_indices).squeeze() # (batch_size,)
        
        dot_product = (user_vec * item_vec).sum(dim=1) # (batch_size,)
        
        prediction = self.global_bias + ub + ib + dot_product
        return prediction

    def fit(self, train_users, train_items, user_to_idx, item_to_idx):
        """
        训练模型
        train_users: 原始ID格式的训练数据 {user_id: [(item_id, rating), ...]}
        train_items: 原始ID格式的训练数据 {item_id: [(user_id, rating), ...]} (可选, 此处主要用train_users)
        user_to_idx: 用户原始ID到整数索引的映射
        item_to_idx: 物品原始ID到整数索引的映射
        """
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        
        # 1. 数据准备: 将 train_users 转换为 (user_idx, item_idx, rating) 列表
        training_samples = []
        for user_id_orig, ratings in train_users.items():
            if user_id_orig not in self.user_to_idx:
                # print(f"Warning: User {user_id_orig} not in user_to_idx during training. Skipping.")
                continue
            u_idx = self.user_to_idx[user_id_orig]
            for item_id_orig, rating in ratings:
                if item_id_orig not in self.item_to_idx:
                    # print(f"Warning: Item {item_id_orig} not in item_to_idx for user {user_id_orig} during training. Skipping.")
                    continue
                i_idx = self.item_to_idx[item_id_orig]
                training_samples.append([u_idx, i_idx, float(rating)])

        if not training_samples:
            print("Error: No training samples available for BiasSVD after mapping. Aborting fit.")
            return

        # 转换为 PyTorch Tensors 和 DataLoader
        user_indices_tensor = torch.tensor([s[0] for s in training_samples], dtype=torch.long).to(self.device)
        item_indices_tensor = torch.tensor([s[1] for s in training_samples], dtype=torch.long).to(self.device)
        ratings_tensor = torch.tensor([s[2] for s in training_samples], dtype=torch.float).to(self.device)
        
        dataset = TensorDataset(user_indices_tensor, item_indices_tensor, ratings_tensor)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 2. 定义优化器和损失函数
        # AdamW 优化器内置了 weight_decay (L2 正则化)
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.reg)
        criterion = nn.MSELoss()

        # 3. 训练循环
        print(f"Starting BiasSVD training on {self.device} for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            self.train() # 设置模型为训练模式
            
            batch_pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{self.epochs} Batches", leave=False, unit="batch")
            for u_idx_b, i_idx_b, r_b in batch_pbar:
                optimizer.zero_grad()
                predictions = self._forward_pass(u_idx_b, i_idx_b)
                loss = criterion(predictions, r_b)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * u_idx_b.size(0)
                batch_pbar.set_postfix_str(f"Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / len(dataset)
            print(f"Epoch {epoch+1}/{self.epochs}, Avg Loss: {avg_epoch_loss:.6f}")
        
        print("BiasSVD training finished.")
        # 将映射保存到模型实例中，以便 predict 方法使用
        # (在 fit 开始时已经赋值了)

    def predict(self, user_item_pairs):
        """
        对给定的用户-物品对进行评分预测
        user_item_pairs: [(user_id_orig, item_id_orig), ...] 列表
        """
        if self.user_to_idx is None or self.item_to_idx is None:
            raise ValueError("Model has not been trained yet or mappings are not set. Call fit() first.")

        self.eval() # 设置模型为评估模式
        
        predictions_output = []
        
        # 为了批量处理，先收集所有有效的索引
        valid_user_indices = []
        valid_item_indices = []
        original_indices_of_valid_pairs = [] # 记录有效对在原始 user_item_pairs 中的索引

        for i, (user_id_orig, item_id_orig) in enumerate(user_item_pairs):
            u_idx = self.user_to_idx.get(user_id_orig)
            i_idx = self.item_to_idx.get(item_id_orig)

            if u_idx is not None and i_idx is not None:
                valid_user_indices.append(u_idx)
                valid_item_indices.append(i_idx)
                original_indices_of_valid_pairs.append(i)
        
        # 初始化最终预测列表，长度与输入一致，默认值为全局平均或配置的冷启动值
        # 您需要在 config 中定义一个全局平均值，或者在 DataProcessor 中计算并存储它
        # 此处使用一个硬编码的默认值，您应该替换它
        cold_start_prediction = getattr(self.config, 'global_mean', 3.0) 
        final_predictions = [cold_start_prediction] * len(user_item_pairs)

        if not valid_user_indices: # 如果所有对都是冷启动或无效
            return final_predictions

        user_indices_tensor = torch.tensor(valid_user_indices, dtype=torch.long).to(self.device)
        item_indices_tensor = torch.tensor(valid_item_indices, dtype=torch.long).to(self.device)
        
        # 使用 DataLoader 进行批量预测，以防数据量过大
        dataset_predict = TensorDataset(user_indices_tensor, item_indices_tensor)
        dataloader_predict = DataLoader(dataset_predict, batch_size=self.batch_size, shuffle=False)

        model_predictions_list = []
        with torch.no_grad():
            for u_idx_b, i_idx_b in dataloader_predict:
                preds_batch = self._forward_pass(u_idx_b, i_idx_b)
                model_predictions_list.extend(preds_batch.cpu().numpy().tolist())
        
        # 将模型预测的有效值填充回 final_predictions 的正确位置
        for i, original_idx in enumerate(original_indices_of_valid_pairs):
            final_predictions[original_idx] = model_predictions_list[i]
            
        return final_predictions

    def save_model(self, path):
        """保存模型状态和映射"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'user_to_idx': self.user_to_idx,
            'item_to_idx': self.item_to_idx,
            'num_users': self.num_users,
            'num_items': self.num_items,
            'num_factors': self.num_factors,
            'config': self.config # 保存配置以便恢复
        }, path)
        print(f"BiasSVDRecommender model saved to {path}")

    @classmethod
    def load_model(cls, path, config_override=None): # config_override 用于在加载时覆盖部分配置
        """加载模型状态和映射"""
        checkpoint = torch.load(path)
        
        # 使用保存的 config 或 override 的 config 初始化模型
        # 如果 config_override 提供了新的 config 对象，则使用它
        # 否则，尝试从 checkpoint 中恢复 config
        # 如果两者都没有，则会报错，因为 BaseRecommender 需要 config
        
        # 优先使用 config_override 中的 config 对象
        # 如果没有，则使用 checkpoint 中的 config
        # 如果 checkpoint 中也没有，则需要外部提供一个有效的 config
        
        # 假设 config 是 ExperimentConfig 类的实例
        # 我们需要确保在加载时有一个有效的 config 对象
        
        # 简化：假设加载时会提供一个 config 对象，或者 checkpoint 中有
        # 如果 config_override 是一个完整的 ExperimentConfig 对象，直接用它
        # 否则，如果 checkpoint['config'] 存在，用它
        # 否则，会出问题，因为 BaseRecommender 需要 config
        
        # 这里的逻辑是：如果调用者提供了 config_override (完整的 ExperimentConfig 对象)，就用它。
        # 否则，如果 checkpoint 里保存了 config 对象，就用它。
        # 如果两者都没有，那么实例化会失败，因为 BaseRecommender 的 __init__ 需要 config。
        # 通常，在您的框架中，当加载模型时，您应该已经有一个 config 对象可用。
        
        # 确保我们有一个有效的 config 对象来实例化
        current_config = config_override if config_override else checkpoint.get('config')
        if current_config is None:
            raise ValueError("A valid ExperimentConfig object must be provided either via config_override or be present in the checkpoint.")

        model = cls(
            config=current_config,
            num_users=checkpoint['num_users'],
            num_items=checkpoint['num_items'],
            num_factors=checkpoint['num_factors']
            # 其他在 __init__ 中定义的参数如果也保存在 checkpoint 中，也可以加载
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.user_to_idx = checkpoint['user_to_idx']
        model.item_to_idx = checkpoint['item_to_idx']
        model.to(model.device) # 确保模型在正确的设备上
        print(f"BiasSVDRecommender model loaded from {path}")
        return model
