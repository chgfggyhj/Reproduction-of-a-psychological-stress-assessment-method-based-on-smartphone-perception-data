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
        self.model_1 = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        # 初始化无标签数据的模型
        self.model_2 = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        self.feature_selector = None  # 特征选择器
        self.selected_features = None  # 选中的特征
        self.select_feature_mask=None
        self.unselect_feature_mask=None       

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
        self.select_feature_mask=self.feature_selector.get_support()
        self.unselect_feature_mask=~self.select_feature_mask
        return self.feature_selector.get_support()
        
    def train_initial(self, X_labeled, y_labeled, X_unlabeled):
        """
        初始训练阶段
        参数:
            X_labeled: 有标签数据的特征矩阵
            y_labeled: 有标签数据的标签
            X_unlabeled: 无标签数据的特征矩阵
        """
        # 模型1选择特征
        feature_mask=self.select_features(X_labeled,y_labeled)
        X_1_selected = X_labeled[:,feature_mask]
        # 模型2选择特征
        X_2_selected = X_labeled[:,self.unselect_feature_mask]
        
        
        # 训练模型1
        self.model_1.fit(X_1_selected, y_labeled)
        
        # 训练模型2
        self.model_2.fit(X_2_selected, y_labeled)
        
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
        X_1_selected = self.feature_selector.transform(X_labeled)

        X_2_selected = X_labeled[:,self.unselect_feature_mask]
        X_1_unlabeled_selected = self.feature_selector.transform(X_unlabeled)
        X_2_unlabeled_selected=X_unlabeled[:,self.unselect_feature_mask]
        # 初始化当前训练数据
        current_labeled_X_1 = X_1_selected.copy()
        current_labeled_X_2 = X_2_selected.copy()
        current_labeled_y = y_labeled.copy()
        current_unlabeled_X_1 = X_1_unlabeled_selected.copy()
        current_unlabeled_X_2 = X_2_unlabeled_selected.copy()
        
        # 迭代训练
        for iteration in range(n_iterations):
            # 获取两个模型的预测概率
            labeled_1_probs = self.model_1.predict_proba(current_unlabeled_X_1)
            labeled_2_probs = self.model_2.predict_proba(current_unlabeled_X_2)
            
            # 计算置信度和预测结果
            labeled_1_conf = np.max(labeled_1_probs, axis=1)
            labeled_2_conf = np.max(labeled_2_probs, axis=1)
            labeled_1_pred = np.argmax(labeled_1_probs, axis=1)
            labeled_2_pred = np.argmax(labeled_2_probs, axis=1)
            
            # 找出两个模型预测一致且置信度高的样本
            agree_idx = (labeled_1_pred == labeled_2_pred) & \
                       (labeled_1_conf > confidence_threshold) & \
                       (labeled_2_conf > confidence_threshold)
            
            if not np.any(agree_idx):
                break
                
            # 将高置信度的样本添加到训练集
            new_X_1 = current_unlabeled_X_1[agree_idx]
            new_X_2 = current_unlabeled_X_2[agree_idx]
            new_y = labeled_1_pred[agree_idx]
            
            current_labeled_X_1 = np.vstack([current_labeled_X_1, new_X_1])
            current_labeled_X_2 = np.vstack([current_labeled_X_2, new_X_2])
            current_labeled_y = np.append(current_labeled_y, new_y)
            
            # 更新无标签数据集
            current_unlabeled_X_1 = current_unlabeled_X_1[~agree_idx]
            current_unlabeled_X_2 = current_unlabeled_X_2[~agree_idx]
            
            # 若没有未标记数据了，提前终止
            if current_unlabeled_X_1.shape[0]==0:
                break
        
        self.model_1.fit(current_labeled_X_1,current_labeled_y)
        self.model_2.fit(current_labeled_X_2,current_labeled_y)
            
    def predict(self, X_labeled):
        """
        预测压力水平
        参数:
            X: 特征矩阵
        返回:
            预测的压力水平
        """
        # 转换特征
        X_1_selected = self.feature_selector.transform(X_labeled)
        X_2_selected = X_labeled[:,self.unselect_feature_mask]

        # 获取两个模型的预测概率
        labeled_1_proba = self.model_1.predict_proba(X_1_selected)
        labeled_2_proba = self.model_2.predict_proba(X_2_selected)
        
        # 获取两个模型的类别列表
        labeled_1_classes = self.model_1.classes_
        labeled_2_classes = self.model_2.classes_
        
        # 确定所有可能的类别
        all_classes = np.union1d(labeled_1_classes, labeled_2_classes)
        n_classes = len(all_classes)
        
        # 创建全量类别的概率矩阵
        labeled_1_full = np.zeros((X_1_selected.shape[0], n_classes))
        labeled_2_full = np.zeros((X_2_selected.shape[0], n_classes))
        
        # 将原始概率填充到对应的列
        labeled_1_full[:, np.isin(all_classes, labeled_1_classes)] = labeled_1_proba
        labeled_2_full[:, np.isin(all_classes, labeled_2_classes)] = labeled_2_proba
        
        # 集成两个模型的预测结果
        ensemble_pred = (labeled_1_full + labeled_2_full) / 2
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
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),  # 精确率
            'recall': recall_score(y_test, y_pred, average='weighted'),        # 召回率
            'f1': f1_score(y_test, y_pred, average='weighted')                 # F1分数
        }
