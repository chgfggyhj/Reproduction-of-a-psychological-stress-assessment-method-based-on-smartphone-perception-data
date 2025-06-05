import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

class StressPredictor:
    """
    压力预测器类，使用随机森林和协同训练方法进行压力水平预测
    """
    def __init__(self, n_estimators=100, max_depth=10):
        """
        初始化压力预测器
        参数:
            n_estimators: 随机森林中树的数量
            max_depth: 树的最大深度
        """
        # 初始化有标签数据的模型
        self.model_labeled = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        # 初始化无标签数据的模型
        self.model_unlabeled = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        self.feature_selector = None  # 特征选择器
        self.selected_features = None  # 选中的特征
        
    def select_features(self, X, y, threshold='mean'):
        """
        使用随机森林进行特征选择
        参数:
            X: 特征矩阵
            y: 标签
            threshold: 特征选择阈值
        返回:
            选择后的特征矩阵
        """
        # 使用随机森林进行特征重要性评估
        selector_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        selector_model.fit(X, y)
        
        # 创建特征选择器
        self.feature_selector = SelectFromModel(
            selector_model,
            threshold=threshold,
            prefit=True
        )
        
        # 获取选中的特征
        self.selected_features = self.feature_selector.get_support()
        
        return self.feature_selector.transform(X)
        
    def train_initial(self, X_labeled, y_labeled, X_unlabeled):
        """
        初始训练阶段
        参数:
            X_labeled: 有标签数据的特征矩阵
            y_labeled: 有标签数据的标签
            X_unlabeled: 无标签数据的特征矩阵
        """
        # 选择特征
        X_labeled_selected = self.select_features(X_labeled, y_labeled)
        X_unlabeled_selected = self.feature_selector.transform(X_unlabeled)
        
        # 训练有标签数据模型
        self.model_labeled.fit(X_labeled_selected, y_labeled)
        
        # 使用有标签模型预测无标签数据
        pseudo_labels = self.model_labeled.predict(X_unlabeled_selected)
        
        # 使用伪标签训练无标签数据模型
        self.model_unlabeled.fit(X_unlabeled_selected, pseudo_labels)
        
    def co_training(self, X_labeled, y_labeled, X_unlabeled, n_iterations=5, confidence_threshold=0.8):
        """
        协同训练过程
        参数:
            X_labeled: 有标签数据的特征矩阵
            y_labeled: 有标签数据的标签
            X_unlabeled: 无标签数据的特征矩阵
            n_iterations: 迭代次数
            confidence_threshold: 置信度阈值
        """
        # 转换特征
        X_labeled_selected = self.feature_selector.transform(X_labeled)
        X_unlabeled_selected = self.feature_selector.transform(X_unlabeled)
        
        # 初始化当前训练数据
        current_labeled_X = X_labeled_selected.copy()
        current_labeled_y = y_labeled.copy()
        current_unlabeled_X = X_unlabeled_selected.copy()
        
        # 迭代训练
        for iteration in range(n_iterations):
            # 获取两个模型的预测概率
            labeled_probs = self.model_labeled.predict_proba(current_unlabeled_X)
            unlabeled_probs = self.model_unlabeled.predict_proba(current_unlabeled_X)
            
            # 计算置信度和预测结果
            labeled_conf = np.max(labeled_probs, axis=1)
            unlabeled_conf = np.max(unlabeled_probs, axis=1)
            labeled_pred = np.argmax(labeled_probs, axis=1)
            unlabeled_pred = np.argmax(unlabeled_probs, axis=1)
            
            # 找出两个模型预测一致且置信度高的样本
            agree_idx = (labeled_pred == unlabeled_pred) & \
                       (labeled_conf > confidence_threshold) & \
                       (unlabeled_conf > confidence_threshold)
            
            if not np.any(agree_idx):
                break
                
            # 将高置信度的样本添加到训练集
            new_X = current_unlabeled_X[agree_idx]
            new_y = labeled_pred[agree_idx]
            
            current_labeled_X = np.vstack([current_labeled_X, new_X])
            current_labeled_y = np.append(current_labeled_y, new_y)
            
            # 更新无标签数据集
            current_unlabeled_X = current_unlabeled_X[~agree_idx]
            
            # 重新训练有标签模型
            self.model_labeled.fit(current_labeled_X, current_labeled_y)
            
            # 使用有标签模型预测剩余无标签数据
            pseudo_labels = self.model_labeled.predict(current_unlabeled_X)
            # 重新训练无标签模型
            self.model_unlabeled.fit(current_unlabeled_X, pseudo_labels)
            
    def predict(self, X):
        """
        预测压力水平
        参数:
            X: 特征矩阵
        返回:
            预测的压力水平
        """
        # 转换特征
        X_selected = self.feature_selector.transform(X)
        
        # 获取两个模型的预测概率
        labeled_proba = self.model_labeled.predict_proba(X_selected)
        unlabeled_proba = self.model_unlabeled.predict_proba(X_selected)
        
        # 获取两个模型的类别列表
        labeled_classes = self.model_labeled.classes_
        unlabeled_classes = self.model_unlabeled.classes_
        
        # 确定所有可能的类别
        all_classes = np.union1d(labeled_classes, unlabeled_classes)
        n_classes = len(all_classes)
        
        # 创建全量类别的概率矩阵
        labeled_full = np.zeros((X_selected.shape[0], n_classes))
        unlabeled_full = np.zeros((X_selected.shape[0], n_classes))
        
        # 将原始概率填充到对应的列
        labeled_full[:, np.isin(all_classes, labeled_classes)] = labeled_proba
        unlabeled_full[:, np.isin(all_classes, unlabeled_classes)] = unlabeled_proba
        
        # 集成两个模型的预测结果
        ensemble_pred = (labeled_full + unlabeled_full) / 2
        return np.argmax(ensemble_pred, axis=1)
        
    def evaluate(self, X_test, y_test):
        """
        评估模型性能
        参数:
            X_test: 测试集特征矩阵
            y_test: 测试集标签
        返回:
            包含各项评估指标的字典
        """
        # 获取预测结果
        y_pred = self.predict(X_test)
        
        # 计算各项评估指标
        return {
            'accuracy': accuracy_score(y_test, y_pred),      # 准确率
            'precision': precision_score(y_test, y_pred, average='weighted'),  # 精确率
            'recall': recall_score(y_test, y_pred, average='weighted'),        # 召回率
            'f1': f1_score(y_test, y_pred, average='weighted')                 # F1分数
        }
