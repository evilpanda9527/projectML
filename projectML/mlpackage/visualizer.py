# visualizer.py v20251004-2
import sys
import matplotlib
# 僅在打包環境中啟用 TkAgg 後端
if getattr(sys, 'frozen', False):  # PyInstaller 打包標誌
    matplotlib.use('TkAgg')  # 強制使用 Tkinter 後端
import matplotlib.pyplot as plt
import numpy as np
import logging
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.tree import plot_tree
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self, ml_model, data=None, model=None):
        """
        初始化 Visualizer 類，設置模型和數據。

        input:
            ml_model (str): 機器學習模型名稱。
            data (dict): 包含視覺化所需的數據，預設為空字典。
            model: 訓練好的模型實例，預設為 None。
        output:
            初始化 self.ml_model, self.data, self.model。
        return:
            無返回值，初始化 Visualizer 實例。
        """
        self.ml_model = ml_model.lower().replace("_", "")
        self.data = data or {}
        self.model = model
        logger.info(f"Visualizer 初始化，模型: {self.ml_model}, 數據鍵: {self.data.keys()}, 模型: {model}")

    def get_available_plots(self):
        """
        獲取當前模型支持的圖形類型。

        input:
            無外部參數，依賴 self.ml_model。
        output:
            根據模型類型返回支持的圖形列表。
        return:
            list: 支持的圖形類型列表。
        """
        if self.ml_model in ["linearregression", "multipleregression", "polynomialregression"]:
            return ["Scatter", "Line", "ResidualPlot"]
        elif self.ml_model == "logisticregression":
            return ["Scatter", "ROCCurve", "ConfusionMatrix"]
        elif self.ml_model == "decisiontree":
            return ["TreeDiagram", "FeatureImportance", "DecisionBoundary"]
        elif self.ml_model == "knn":
            return ["Scatter", "DecisionBoundary", "KValuePlot"]
        elif self.ml_model == "svm":
            return ["Scatter", "DecisionBoundary", "ROCCurve"]
        elif self.ml_model == "baggingclassifier":
            return ["ErrorPlot", "FeatureImportance", "ConfusionMatrix"]
        elif self.ml_model == "randomforestclassifier":
            return ["FeatureImportance", "ErrorPlot", "ConfusionMatrix"]
        elif self.ml_model == "randomforestregressor":
            return ["FeatureImportance", "ErrorPlot"]
        elif self.ml_model == "dbscan":
            return ["Scatter", "ElbowPlot"]
        elif self.ml_model == "kmeans":
            return ["Scatter", "ElbowPlot"]
        elif self.ml_model == "agglomerativeclustering":
            return ["Scatter", "Dendrogram"]
        else:
            return []

    def visualize(self, plot_type, save_path=None):
        """
        調用指定圖形繪製方法。

        input:
            plot_type (str): 圖形類型。
            save_path (str): 圖形保存路徑，預設為 None（顯示圖形）。
        output:
            繪製並顯示或保存指定圖形。
        return:
            無返回值，若 plot_type 無效則拋出異常。
        """
        logger.info(f"進入 visualize，準備繪製 {plot_type} for {self.ml_model}")
        if plot_type in self.get_available_plots():
            getattr(self, plot_type)(save_path)
            if not save_path:
                plt.show()
        else:
            logger.error(f"無效的圖型: {plot_type} for {self.ml_model}")
            raise ValueError(f"無效的圖型: {plot_type} for {self.ml_model}")

    def Scatter(self, save_path=None):
        """
        繪製散點圖，比較 y_true/y_labels 和 y_pred/labels。

        input:
            save_path (str): 圖形保存路徑，預設為 None（顯示圖形）。
        output:
            繪製散點圖並顯示或保存。
        return:
            無返回值，若數據缺失或格式無效則拋出異常。
        """
        logger.info(f"開始準備 Scatter 資料，模型類型: {self.ml_model}")
        if self.ml_model in ["agglomerativeclustering", "kmeans", "dbscan"]:
            required_keys = ['X', 'labels']
        else:
            required_keys = ['X', 'y_true', 'y_pred']
        
        missing_keys = [key for key in required_keys if key not in self.data]
        if missing_keys:
            logger.error(f"缺少必要的數據: {self.data.keys()}")
            raise ValueError(f"缺少必要的數據: {', '.join(missing_keys)}")

        X = self.data['X']
        if hasattr(X, 'values'):
            X = X.values
        elif not isinstance(X, np.ndarray):
            logger.error(f"X 數據格式不支援: {type(X)}")
            raise ValueError(f"X 數據格式不支援: {type(X)}")

        if self.ml_model in ["agglomerativeclustering", "kmeans", "dbscan"]:
            y = self.data['labels']
            label_type = 'Cluster Labels'
        else:
            y_true = self.data['y_true']
            y_pred = self.data['y_pred']
            if len(y_true) != len(y_pred):
                logger.error(f"y_true 和 y_pred 長度不匹配: {len(y_true)} vs {len(y_pred)}")
                raise ValueError(f"y_true 和 y_pred 長度不匹配: {len(y_true)} vs {len(y_pred)}")
            y = y_pred
            label_type = 'Predicted Values'

        try:
            plt.figure(figsize=(8, 6))
            if X.shape[1] >= 2:
                plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.5)
                plt.xlabel('Feature 1')
                plt.ylabel('Feature 2')
            else:
                plt.scatter(range(len(y)), y, c=y, cmap='viridis', alpha=0.5)
                plt.xlabel('Sample Index')
                plt.ylabel(label_type)
            plt.title(f'Scatter Plot for {self.ml_model}')
            plt.colorbar(label=label_type)
            logger.info("Scatter 準備出圖")
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Scatter 已保存至 {save_path}")
            logger.info("Scatter 出圖/保存完成")
        except Exception as e:
            logger.error(f"繪圖錯誤: {e}")
            raise

    def Line(self, save_path=None):
        """
        繪製預測值與真值的線性趨勢（適用於 Linear_Regression）。

        input:
            save_path (str): 圖形保存路徑，預設為 None（顯示圖形）。
        output:
            繪製線性趨勢圖並顯示或保存。
        return:
            無返回值，若數據缺失或長度不匹配則拋出異常。
        """
        logger.info(f"開始準備 Line 資料，模型類型: {self.ml_model}")
        if not all(key in self.data for key in ['y_true', 'y_pred']):
            logger.error(f"缺少必要的數據: {self.data.keys()}")
            raise ValueError(f"缺少必要的數據: {', '.join(['y_true', 'y_pred'] - set(self.data.keys()))}")
        
        y_true = self.data['y_true']
        y_pred = self.data['y_pred']

        if len(y_true) != len(y_pred):
            logger.error(f"y_true 和 y_pred 長度不匹配: {len(y_true)} vs {len(y_pred)}")
            raise ValueError(f"y_true 和 y_pred 長度不匹配: {len(y_true)} vs {len(y_pred)}")

        try:
            plt.figure(figsize=(8, 6))
            plt.plot(range(len(y_true)), y_true, 'b-', label='True Values', alpha=0.5)
            plt.plot(range(len(y_pred)), y_pred, 'r-', label='Predicted Values', alpha=0.5)
            plt.xlabel('Sample Index')
            plt.ylabel('Value')
            plt.title(f'Line Plot for {self.ml_model}')
            plt.legend()
            logger.info("Line 準備出圖")
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Line 已保存至 {save_path}")
            logger.info("Line 出圖/保存完成")
        except Exception as e:
            logger.error(f"繪圖錯誤: {e}")
            raise

    def ResidualPlot(self, save_path=None):
        """
        繪製殘差圖（適用於 Linear_Regression）。

        input:
            save_path (str): 圖形保存路徑，預設為 None（顯示圖形）。
        output:
            繪製殘差圖並顯示或保存。
        return:
            無返回值，若數據缺失或長度不匹配則拋出異常。
        """
        logger.info(f"開始準備 ResidualPlot 資料，模型類型: {self.ml_model}")
        if not all(key in self.data for key in ['y_true', 'y_pred']):
            logger.error(f"缺少必要的數據: {self.data.keys()}")
            raise ValueError(f"缺少必要的數據: {', '.join(['y_true', 'y_pred'] - set(self.data.keys()))}")
        
        y_true = self.data['y_true']
        y_pred = self.data['y_pred']

        if len(y_true) != len(y_pred):
            logger.error(f"y_true 和 y_pred 長度不匹配: {len(y_true)} vs {len(y_pred)}")
            raise ValueError(f"y_true 和 y_pred 長度不匹配: {len(y_true)} vs {len(y_pred)}")

        residuals = y_true - y_pred
        try:
            plt.figure(figsize=(8, 6))
            plt.scatter(range(len(residuals)), residuals, color='purple', alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Sample Index')
            plt.ylabel('Residuals')
            plt.title(f'Residual Plot for {self.ml_model}')
            logger.info("ResidualPlot 準備出圖")
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"ResidualPlot 已保存至 {save_path}")
            logger.info("ResidualPlot 出圖/保存完成")
        except Exception as e:
            logger.error(f"繪圖錯誤: {e}")
            raise

    def ROCCurve(self, save_path=None):
        """
        繪製 ROC 曲線（適用於 Logistic_Regression，需 y_prob）。

        input:
            save_path (str): 圖形保存路徑，預設為 None（顯示圖形）。
        output:
            繪製 ROC 曲線並顯示或保存。
        return:
            無返回值，若數據缺失或長度不匹配則拋出異常。
        """
        logger.info(f"開始準備 ROCCurve 資料，模型類型: {self.ml_model}")
        if not all(key in self.data for key in ['y_true', 'y_prob']):
            logger.error(f"缺少必要的數據: {self.data.keys()}")
            raise ValueError(f"缺少必要的數據: {', '.join(['y_true', 'y_prob'] - set(self.data.keys()))}")
        
        y_true = self.data['y_true']
        y_prob = self.data['y_prob']

        if len(y_true) != len(y_prob):
            logger.error(f"y_true 和 y_prob 長度不匹配: {len(y_true)} vs {len(y_prob)}")
            raise ValueError(f"y_true 和 y_prob 長度不匹配: {len(y_true)} vs {len(y_prob)}")

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = len(np.unique(y_true))
        for i in range(n_classes):
            y_true_bin = (y_true == i).astype(int)
            fpr[i], tpr[i], _ = roc_curve(y_true_bin, y_prob[:, i] if y_prob.ndim > 1 else y_prob)
            roc_auc[i] = auc(fpr[i], tpr[i])

        try:
            plt.figure(figsize=(8, 6))
            colors = ['blue', 'red', 'green', 'yellow', 'purple']
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve class {i} (AUC = {roc_auc[i]:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {self.ml_model}')
            plt.legend(loc="lower right")
            logger.info("ROCCurve 準備出圖")
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"ROCCurve 已保存至 {save_path}")
            logger.info("ROCCurve 出圖/保存完成")
        except Exception as e:
            logger.error(f"繪圖錯誤: {e}")
            raise

    def ConfusionMatrix(self, save_path=None):
        """
        繪製混淆矩陣（適用於 Logistic_Regression）。

        input:
            save_path (str): 圖形保存路徑，預設為 None（顯示圖形）。
        output:
            繪製混淆矩陣並顯示或保存。
        return:
            無返回值，若數據缺失或長度不匹配則拋出異常。
        """
        logger.info(f"開始準備 ConfusionMatrix 資料，模型類型: {self.ml_model}")
        if not all(key in self.data for key in ['y_true', 'y_pred']):
            logger.error(f"缺少必要的數據: {self.data.keys()}")
            raise ValueError(f"缺少必要的數據: {', '.join(['y_true', 'y_pred'] - set(self.data.keys()))}")
        
        y_true = self.data['y_true']
        y_pred = self.data['y_pred']

        if len(y_true) != len(y_pred):
            logger.error(f"y_true 和 y_pred 長度不匹配: {len(y_true)} vs {len(y_pred)}")
            raise ValueError(f"y_true 和 y_pred 長度不匹配: {len(y_true)} vs {len(y_pred)}")

        le = LabelEncoder()
        y_true_encoded = le.fit_transform(y_true)
        y_pred_encoded = le.transform(y_pred)

        all_labels = np.union1d(y_true_encoded, y_pred_encoded)
        n_classes = len(all_labels)
        logger.info(f"編碼後所有唯一值: {all_labels}, 類別數: {n_classes}")

        cm = confusion_matrix(y_true_encoded, y_pred_encoded, labels=all_labels)

        try:
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix for {self.ml_model}')
            plt.colorbar()
            plt.xticks(np.arange(len(all_labels)), all_labels)
            plt.yticks(np.arange(len(all_labels)), all_labels)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            logger.info("ConfusionMatrix 準備出圖")
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"ConfusionMatrix 已保存至 {save_path}")
            logger.info("ConfusionMatrix 出圖/保存完成")
        except Exception as e:
            logger.error(f"繪圖錯誤: {e}")
            raise

    def TreeDiagram(self, save_path=None):
        """
        繪製決策樹圖（適用於 DecisionTree）。

        input:
            save_path (str): 圖形保存路徑，預設為 None（顯示圖形）。
        output:
            繪製決策樹圖並顯示或保存。
        return:
            無返回值，若數據缺失則拋出異常。
        """
        logger.info(f"開始準備 TreeDiagram 資料，模型類型: {self.ml_model}")
        if not all(key in self.data for key in ['X', 'y_true']):
            logger.error(f"缺少必要的數據: {self.data.keys()}")
            raise ValueError(f"缺少必要的數據: {', '.join(['X', 'y_true'] - set(self.data.keys()))}")
        
        X = self.data['X']
        y_true = self.data['y_true']

        try:
            plt.figure(figsize=(20, 10))
            plot_tree(self.model, feature_names=X.columns if hasattr(X, 'columns') else None, 
                      class_names=str(np.unique(y_true)) if not self.is_continuous else None, 
                      filled=True, rounded=True, impurity=False)
            plt.title(f'Tree Diagram for {self.ml_model}')
            logger.info("TreeDiagram 準備出圖")
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"TreeDiagram 已保存至 {save_path}")
            logger.info("TreeDiagram 出圖/保存完成")
        except Exception as e:
            logger.error(f"繪圖錯誤: {e}")
            raise

    def FeatureImportance(self, save_path=None):
        """
        繪製特徵重要性（適用於 RandomForest 或 DecisionTree）。

        input:
            save_path (str): 圖形保存路徑，預設為 None（顯示圖形）。
        output:
            繪製特徵重要性圖並顯示或保存。
        return:
            無返回值，若模型不支持 feature_importances_ 或數據缺失則拋出異常。
        """
        logger.info(f"開始準備 FeatureImportance 資料，模型類型: {self.ml_model}")
        if not all(key in self.data for key in ['X']):
            logger.error(f"缺少必要的數據: {self.data.keys()}")
            raise ValueError(f"缺少必要的數據: {', '.join(['X'] - set(self.data.keys()))}")
        
        X = self.data['X']
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = X.columns if hasattr(X, 'columns') else [f'Feature {i}' for i in range(X.shape[1])]
            feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
            feature_importance = feature_importance.sort_values('importance', ascending=False)

            try:
                plt.figure(figsize=(10, 6))
                sns.barplot(x='importance', y='feature', data=feature_importance)
                plt.title(f'Feature Importance for {self.ml_model}')
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                logger.info("FeatureImportance 準備出圖")
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    logger.info(f"FeatureImportance 已保存至 {save_path}")
                logger.info("FeatureImportance 出圖/保存完成")
            except Exception as e:
                logger.error(f"繪圖錯誤: {e}")
                raise
        else:
            logger.warning(f"模型 {self.ml_model} 不支持 feature_importances_")
            raise ValueError(f"模型 {self.ml_model} 不支持 feature_importances_")

    def DecisionBoundary(self, save_path=None):
        """
        繪製決策邊界（適用於 KNN 或 SVM，僅2D數據）。

        input:
            save_path (str): 圖形保存路徑，預設為 None（顯示圖形）。
        output:
            繪製決策邊界圖並顯示或保存。
        return:
            無返回值，若數據維度非2D或數據缺失則拋出異常。
        """
        logger.info(f"開始準備 DecisionBoundary 資料，模型類型: {self.ml_model}")
        if not all(key in self.data for key in ['X', 'y_pred']):
            logger.error(f"缺少必要的數據: {self.data.keys()}")
            raise ValueError(f"缺少必要的數據: {', '.join(['X', 'y_pred'] - set(self.data.keys()))}")
        
        X = self.data['X']
        y_pred = self.data['y_pred']

        if X.shape[1] != 2:
            logger.error(f"決策邊界僅支持2D數據，當前維度: {X.shape[1]}")
            raise ValueError(f"決策邊界僅支持2D數據，當前維度: {X.shape[1]}")

        try:
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
            Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            plt.figure(figsize=(8, 6))
            plt.contourf(xx, yy, Z, alpha=0.4)
            plt.scatter(X[:, 0], X[:, 1], c=y_pred, alpha=0.8)
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title(f'Decision Boundary for {self.ml_model}')
            logger.info("DecisionBoundary 準備出圖")
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"DecisionBoundary 已保存至 {save_path}")
            logger.info("DecisionBoundary 出圖/保存完成")
        except Exception as e:
            logger.error(f"繪圖錯誤: {e}")
            raise

    def KValuePlot(self, save_path=None):
        """
        繪製 K 值對模型性能的影響（適用於 KNN）。

        input:
            save_path (str): 圖形保存路徑，預設為 None（顯示圖形）。
        output:
            繪製 K 值與準確率圖並顯示或保存。
        return:
            無返回值，若數據缺失則拋出異常。
        """
        logger.info(f"開始準備 KValuePlot 資料，模型類型: {self.ml_model}")
        if not all(key in self.data for key in ['X', 'y_true']):
            logger.error(f"缺少必要的數據: {self.data.keys()}")
            raise ValueError(f"缺少必要的數據: {', '.join(['X', 'y_true'] - set(self.data.keys()))}")
        
        X = self.data['X']
        y_true = self.data['y_true']
        k_values = range(1, min(31, len(y_true) // 2))
        scores = []

        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X, y_true)
            y_pred = knn.predict(X)
            scores.append(accuracy_score(y_true, y_pred))

        try:
            plt.figure(figsize=(8, 6))
            plt.plot(k_values, scores, 'b-', marker='o')
            plt.xlabel('K Value')
            plt.ylabel('Accuracy')
            plt.title(f'K Value vs Accuracy for {self.ml_model}')
            plt.grid(True)
            logger.info("KValuePlot 準備出圖")
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"KValuePlot 已保存至 {save_path}")
            logger.info("KValuePlot 出圖/保存完成")
        except Exception as e:
            logger.error(f"繪圖錯誤: {e}")
            raise

    def ErrorPlot(self, save_path=None):
        """
        繪製錯誤率圖（適用於 Bagging 或 RandomForest）。

        input:
            save_path (str): 圖形保存路徑，預設為 None（顯示圖形）。
        output:
            繪製錯誤率圖並顯示或保存。
        return:
            無返回值，若數據缺失則拋出異常。
        """
        logger.info(f"開始準備 ErrorPlot 資料，模型類型: {self.ml_model}")
        if not all(key in self.data for key in ['X', 'y_true', 'y_pred']):
            logger.error(f"缺少必要的數據: {self.data.keys()}")
            raise ValueError(f"缺少必要的數據: {', '.join(['X', 'y_true', 'y_pred'] - set(self.data.keys()))}")
        
        X = self.data['X']
        y_true = self.data['y_true']
        y_pred = self.data['y_pred']
        errors = 1 - np.mean(y_true == y_pred)

        try:
            plt.figure(figsize=(8, 6))
            plt.bar(['Error Rate'], [errors], color='red')
            plt.ylabel('Error Rate')
            plt.title(f'Error Rate for {self.ml_model}')
            logger.info("ErrorPlot 準備出圖")
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"ErrorPlot 已保存至 {save_path}")
            logger.info("ErrorPlot 出圖/保存完成")
        except Exception as e:
            logger.error(f"繪圖錯誤: {e}")
            raise

    def ElbowPlot(self, save_path=None):
        """
        繪製肘部圖（適用於 KMeans 或 DBSCAN）。

        input:
            save_path (str): 圖形保存路徑，預設為 None（顯示圖形）。
        output:
            繪製肘部圖並顯示或保存。
        return:
            無返回值，若數據缺失則拋出異常。
        """
        logger.info(f"開始準備 ElbowPlot 資料，模型類型: {self.ml_model}")
        if not all(key in self.data for key in ['X']):
            logger.error(f"缺少必要的數據: {self.data.keys()}")
            raise ValueError(f"缺少必要的數據: {', '.join(['X'] - set(self.data.keys()))}")
        
        X = self.data['X']
        inertias = []
        k_values = range(1, min(11, len(X) // 2))

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        try:
            plt.figure(figsize=(8, 6))
            plt.plot(k_values, inertias, 'b-', marker='o')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Inertia')
            plt.title(f'Elbow Plot for {self.ml_model}')
            plt.grid(True)
            logger.info("ElbowPlot 準備出圖")
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"ElbowPlot 已保存至 {save_path}")
            logger.info("ElbowPlot 出圖/保存完成")
        except Exception as e:
            logger.error(f"繪圖錯誤: {e}")
            raise

    def Dendrogram(self, save_path=None):
        """
        繪製樹狀圖（適用於 AgglomerativeClustering）。

        input:
            save_path (str): 圖形保存路徑，預設為 None（顯示圖形）。
        output:
            繪製樹狀圖並顯示或保存。
        return:
            無返回值，若數據缺失則拋出異常。
        """
        logger.info(f"開始準備 Dendrogram 資料，模型類型: {self.ml_model}")
        if not all(key in self.data for key in ['X']):
            logger.error(f"缺少必要的數據: {self.data.keys()}")
            raise ValueError(f"缺少必要的數據: {', '.join(['X'] - set(self.data.keys()))}")
        
        X = self.data['X']
        from scipy.cluster.hierarchy import linkage
        linked = linkage(X, method='ward')

        try:
            plt.figure(figsize=(10, 7))
            dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
            plt.title(f'Dendrogram for {self.ml_model}')
            logger.info("Dendrogram 準備出圖")
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Dendrogram 已保存至 {save_path}")
            logger.info("Dendrogram 出圖/保存完成")
        except Exception as e:
            logger.error(f"繪圖錯誤: {e}")
            raise