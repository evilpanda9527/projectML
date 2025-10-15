# ML_runner.py v20251004-2
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
import time

logger = logging.getLogger(__name__)

class MLRunner:
    def __init__(self, processed_data, processed_target, model_type="Linear_Regression"):
        """
        初始化 MLRunner 類，設定資料和模型類型。

        input:
            processed_data (dict): 包含 'minmax' 和 'standard' 版本的預處理數據。
            processed_target (pd.Series): 預處理後的目標變量。
            model_type (str): 選定模型類型，預設為 "Linear_Regression"。
        output:
            初始化 self.processed_data, self.processed_target, self.model_type 和 self.model。
        return:
            無返回值，初始化 MLRunner 實例。
        """
        self.processed_data = processed_data
        self.processed_target = processed_target
        self.model_type = model_type.lower()
        self.model = None
        self.is_continuous = self._check_continuity()

    def _check_continuity(self):
        """
        檢查目標變量是否為連續變量。

        input:
            依賴 self.processed_target。
        output:
            判斷目標變量是否連續並更新相關邏輯。
        return:
            bool: True 若為連續變量，否則 False。
        """
        unique_values = len(np.unique(self.processed_target))
        range_values = np.max(self.processed_target) - np.min(self.processed_target)
        return np.issubdtype(self.processed_target.dtype, np.number) and unique_values > 10 and range_values > 1.0

    def prepare_data(self, is_unsupervised=False):
        """
        準備數據，包括動態 q 計算和數據分割。

        input:
            is_unsupervised (bool): 是否為無監督學習，預設為 False。
        output:
            準備訓練和測試數據，或無監督數據。
        return:
            tuple: (X_train, X_test, y_train, y_test, q) 或 (X, None, None, None, q)
                - X_train/X (pandas.DataFrame): 訓練數據或全部數據。
                - X_test (pandas.DataFrame): 測試數據（監督學習時）。
                - y_train/y_test (pd.Series): 訓練/測試目標（監督學習時）。
                - q (int): 動態計算的 q 值。
        """
        logger.info("開始準備資料，模型類型: %s", self.model_type)
        logger.info("選擇 Standard 版本資料，形狀: %s", self.processed_data['standard'].shape)
        logger.info("y 數據類型: %s", self.processed_target.dtype)

        if self.model_type in ["logistic_regression", "svm", "bagging_classifier", "random_forest_classifier"] and self.is_continuous:
            discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
            self.processed_target = discretizer.fit_transform(self.processed_target.values.reshape(-1, 1)).ravel().astype(np.int64)
            logger.warning("警告: 該模型不適合連續目標變量。已自動分箱為離散類別，但將繼續執行。")
            self.is_continuous = False

        q = int(min(max(2, len(self.processed_target) / 100), len(np.unique(self.processed_target))))
        logger.info("y 唯一值數量: %d, 範圍: %d-%d", len(np.unique(self.processed_target)), 
                    np.min(self.processed_target), np.max(self.processed_target))
        logger.info("y 分佈: %s", str(dict(zip(np.unique(self.processed_target), np.bincount(self.processed_target)))))
        logger.info("y 樣本數: %d", len(self.processed_target))
        logger.info("y 連續性判斷: %s", self.is_continuous)

        if is_unsupervised:
            X = self.processed_data['standard']
            logger.info("無監督學習，使用全部數據，形狀: %s", X.shape)
            return X, None, None, None, q
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                self.processed_data['standard'], self.processed_target, test_size=0.3, random_state=42
            )
            logger.info("y_train 唯一值數量: %d, 範圍: %d-%d", len(np.unique(y_train)), 
                        np.min(y_train), np.max(y_train))
            logger.info("資料分割完成 - 訓練集: %s, 測試集: %s", X_train.shape, X_test.shape)
            return X_train, X_test, y_train, y_test, q

    def run(self):
        """
        運行選定模型，返回結果字典。

        input:
            無外部參數，依賴 self.processed_data, self.processed_target, self.model_type。
        output:
            訓練模型並生成結果字典。
        return:
            dict: 包含訓練結果的字典，包括 "X", "y_true", "y_pred", "metric", "report", "training_time", "status"。
        """
        start_time = time.time()

        if self.model_type in ["dbscan", "kmeans", "agglomerative_clustering"]:
            X, _, _, _, q = self.prepare_data(is_unsupervised=True)
        else:
            X_train, X_test, y_train, y_test, q = self.prepare_data()

        if self.model_type == "linear_regression":
            self.model = LinearRegression()
            result = self._run_regression(X_train, X_test, y_train, y_test)
        elif self.model_type == "multiple_regression":
            self.model = LinearRegression()
            result = self._run_regression(X_train, X_test, y_train, y_test)
        elif self.model_type == "polynomial_regression":
            self.model = LinearRegression()
            result = self._run_polynomial_regression(X_train, X_test, y_train, y_test, q)
        elif self.model_type == "logistic_regression":
            self.model = LogisticRegression(max_iter=1000)
            result = self._run_classification(X_train, X_test, y_train, y_test)
        elif self.model_type == "decision_tree":
            if self.is_continuous:
                self.model = DecisionTreeRegressor()
            else:
                self.model = DecisionTreeClassifier()
            result = self._run_tree_based(X_train, X_test, y_train, y_test)
        elif self.model_type == "knn":
            if self.is_continuous:
                self.model = KNeighborsRegressor()
            else:
                self.model = KNeighborsClassifier()
            result = self._run_knn(X_train, X_test, y_train, y_test)
        elif self.model_type == "svm":
            self.model = SVC()
            result = self._run_classification(X_train, X_test, y_train, y_test)
        elif self.model_type == "bagging_classifier":
            self.model = BaggingClassifier()
            result = self._run_classification(X_train, X_test, y_train, y_test)
        elif self.model_type == "random_forest_classifier":
            self.model = RandomForestClassifier()
            result = self._run_classification(X_train, X_test, y_train, y_test)
        elif self.model_type == "random_forest_regressor":
            self.model = RandomForestRegressor()
            result = self._run_regression(X_train, X_test, y_train, y_test)
        elif self.model_type == "dbscan":
            self.model = DBSCAN()
            result = self._run_dbscan(X)
        elif self.model_type == "kmeans":
            self.model = KMeans()
            result = self._run_kmeans(X)
        elif self.model_type == "agglomerative_clustering":
            self.model = AgglomerativeClustering()
            result = self._run_agglomerative_clustering(X)
        else:
            raise ValueError(f"未知的模型類型: {self.model_type}")

        total_time = time.time() - start_time
        logger.info("%s 流程執行完成，總耗時: %.4f 秒", self.model_type.replace("_", " ").title(), total_time)
        result["training_time"] = total_time
        result["status"] = "completed"
        return result

    def _run_regression(self, X_train, X_test, y_train, y_test):
        """
        運行回歸模型。

        input:
            X_train (pandas.DataFrame): 訓練數據。
            X_test (pandas.DataFrame): 測試數據。
            y_train (pd.Series): 訓練目標。
            y_test (pd.Series): 測試目標。
        output:
            訓練回歸模型並生成預測結果。
        return:
            dict: 包含 "X", "y_true", "y_pred", "metric", "report" 的字典。
        """
        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        logger.info("%s 訓練完成，訓練時間: %.4f 秒", self.model_type.replace("_", " ").title(), training_time)
    
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        metric = {
            "mse": mse_test,
            "r2": r2,
            "level": "高" if r2 > 0.8 else "中" if r2 > 0.5 else "低"
        }
        report = {
            "train_mse": mse_train,
            "test_mse": mse_test,
            "difference": mse_train - mse_test
        }
        logger.info("%s 均方誤差 (MSE): %.4f, R²: %.4f, 級別: %s", 
                    self.model_type.replace("_", " ").title(), mse_test, r2, metric["level"])
        logger.info("訓練集 MSE: %.4f, 測試集 MSE: %.4f, 差異: %.4f", 
                    mse_train, mse_test, report["difference"])
    
        if not self.is_continuous:
            logger.warning("警告: 該模型不適合離散目標變量。建議選擇分類模型，但將繼續執行。")
    
        return {
            "X": X_test,
            "y_true": y_test,
            "y_pred": y_pred_test,
            "metric": metric,
            "report": report
        }

    def _run_polynomial_regression(self, X_train, X_test, y_train, y_test, q):
        """
        運行多項式回歸模型。

        input:
            X_train (pandas.DataFrame): 訓練數據。
            X_test (pandas.DataFrame): 測試數據。
            y_train (pd.Series): 訓練目標。
            y_test (pd.Series): 測試目標。
            q (int): 多項式階數。
        output:
            訓練多項式回歸模型並生成預測結果。
        return:
            dict: 包含 "X", "y_true", "y_pred", "metric", "report" 的字典。
        """
        start_time = time.time()
        poly = PolynomialFeatures(degree=q)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        self.model.fit(X_train_poly, y_train)
        training_time = time.time() - start_time
        logger.info("%s 訓練完成，訓練時間: %.4f 秒", self.model_type.replace("_", " ").title(), training_time)

        y_pred_train = self.model.predict(X_train_poly)
        y_pred_test = self.model.predict(X_test_poly)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        metric = {
            "mse": mse_test,
            "r2": r2,
            "level": "高" if r2 > 0.8 else "中" if r2 > 0.5 else "低"
        }
        report = {
            "train_mse": mse_train,
            "test_mse": mse_test,
            "difference": mse_train - mse_test
        }
        logger.info("%s 均方誤差 (MSE): %.4f, R²: %.4f, 級別: %s", 
                    self.model_type.replace("_", " ").title(), mse_test, r2, metric["level"])
        logger.info("訓練集 MSE: %.4f, 測試集 MSE: %.4f, 差異: %.4f", 
                    mse_train, mse_test, report["difference"])

        if not self.is_continuous:
            logger.warning("警告: 該模型不適合離散目標變量。建議選擇分類模型，但將繼續執行。")

        return {
            "X": X_test,
            "y_true": y_test,
            "y_pred": y_pred_test,
            "metric": metric,
            "report": report
        }

    def _run_classification(self, X_train, X_test, y_train, y_test):
        """
        運行分類模型。

        input:
            X_train (pandas.DataFrame): 訓練數據。
            X_test (pandas.DataFrame): 測試數據。
            y_train (pd.Series): 訓練目標。
            y_test (pd.Series): 測試目標。
        output:
            訓練分類模型並生成預測結果。
        return:
            dict: 包含 "X", "y_true", "y_pred", "y_prob", "metric", "report" 的字典。
        """
        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        logger.info("%s 訓練完成，訓練時間: %.4f 秒", self.model_type.replace("_", " ").title(), training_time)
    
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test) if hasattr(self.model, 'predict_proba') else None
        accuracy = np.mean(y_pred == y_test)
        metric = {"accuracy": accuracy}
        report = {"confusion_matrix": str(np.histogram2d(y_test, y_pred, bins=[np.unique(y_test), np.unique(y_pred)])[0])}
        logger.info("%s 準確率: %.4f", self.model_type.replace("_", " ").title(), accuracy)
    
        if self.is_continuous:
            logger.warning("警告: 該模型不適合連續目標變量。建議選擇回歸模型，但將繼續執行。")
    
        return {
            "X": X_test,
            "y_true": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "metric": metric,
            "report": report
        }

    def _run_tree_based(self, X_train, X_test, y_train, y_test):
        """
        運行基於樹的模型（分類或回歸）。

        input:
            X_train (pandas.DataFrame): 訓練數據。
            X_test (pandas.DataFrame): 測試數據。
            y_train (pd.Series): 訓練目標。
            y_test (pd.Series): 測試目標。
        output:
            訓練樹模型並生成預測結果。
        return:
            dict: 包含 "X", "y_true", "y_pred", "metric", "report" 的字典。
        """
        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        logger.info("%s 訓練完成，訓練時間: %.4f 秒", self.model_type.replace("_", " ").title(), training_time)

        if self.is_continuous:
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            metric = {"mse": mse, "r2": r2, "level": "高" if r2 > 0.8 else "中" if r2 > 0.5 else "低"}
            report = {"train_mse": mean_squared_error(y_train, self.model.predict(X_train)),
                      "test_mse": mse, "difference": mean_squared_error(y_train, self.model.predict(X_train)) - mse}
            logger.info("%s 均方誤差 (MSE): %.4f, R²: %.4f, 級別: %s", 
                        self.model_type.replace("_", " ").title(), mse, r2, metric["level"])
            logger.info("訓練集 MSE: %.4f, 測試集 MSE: %.4f, 差異: %.4f", 
                        report["train_mse"], mse, report["difference"])
        else:
            result = self._run_classification(X_train, X_test, y_train, y_test)
            return result

        return {
            "X": X_test,
            "y_true": y_test,
            "y_pred": y_pred,
            "metric": metric,
            "report": report
        }

    def _run_knn(self, X_train, X_test, y_train, y_test):
        """
        運行 KNN 模型（分類或回歸）。

        input:
            X_train (pandas.DataFrame): 訓練數據。
            X_test (pandas.DataFrame): 測試數據。
            y_train (pd.Series): 訓練目標。
            y_test (pd.Series): 測試目標。
        output:
            訓練 KNN 模型並生成預測結果。
        return:
            dict: 包含 "X", "y_true", "y_pred", "metric", "report" 的字典。
        """
        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        logger.info("%s 訓練完成，訓練時間: %.4f 秒", self.model_type.replace("_", " ").title(), training_time)

        if self.is_continuous:
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            metric = {"mse": mse, "r2": r2, "level": "高" if r2 > 0.8 else "中" if r2 > 0.5 else "低"}
            report = {"train_mse": mean_squared_error(y_train, self.model.predict(X_train)),
                      "test_mse": mse, "difference": mean_squared_error(y_train, self.model.predict(X_train)) - mse}
            logger.info("%s 均方誤差 (MSE): %.4f, R²: %.4f, 級別: %s", 
                        self.model_type.replace("_", " ").title(), mse, r2, metric["level"])
            logger.info("訓練集 MSE: %.4f, 測試集 MSE: %.4f, 差異: %.4f", 
                        report["train_mse"], mse, report["difference"])
        else:
            result = self._run_classification(X_train, X_test, y_train, y_test)
            return result

        return {
            "X": X_test,
            "y_true": y_test,
            "y_pred": y_pred,
            "metric": metric,
            "report": report
        }

    def _run_dbscan(self, X):
        """
        運行 DBSCAN 聚類。

        input:
            X (pandas.DataFrame): 輸入數據。
        output:
            訓練 DBSCAN 模型並生成聚類標籤。
        return:
            dict: 包含 "X", "labels", "metric", "report" 的字典。
        """
        start_time = time.time()
        self.model.fit(X)
        training_time = time.time() - start_time
        logger.info("%s 訓練完成，訓練時間: %.4f 秒", self.model_type.replace("_", " ").title(), training_time)
        metric = {"n_clusters": len(set(self.model.labels_) - {-1}), "noise_points": list(self.model.labels_).count(-1)}
        report = {"labels": str(self.model.labels_)}
        logger.info("%s 聚類數: %d, 噪點數: %d", self.model_type.replace("_", " ").title(), 
                    metric["n_clusters"], metric["noise_points"])
    
        return {
            "X": X,
            "labels": self.model.labels_,
            "metric": metric,
            "report": report
        }

    def _run_kmeans(self, X):
        """
        運行 KMeans 聚類。

        input:
            X (pandas.DataFrame): 輸入數據。
        output:
            訓練 KMeans 模型並生成聚類標籤。
        return:
            dict: 包含 "X", "labels", "metric", "report" 的字典。
        """
        start_time = time.time()
        self.model.fit(X)
        training_time = time.time() - start_time
        logger.info("%s 訓練完成，訓練時間: %.4f 秒", self.model_type.replace("_", " ").title(), training_time)
        metric = {"n_clusters": self.model.n_clusters, "inertia": self.model.inertia_}
        report = {"labels": str(self.model.labels_)}
        logger.info("%s 聚類數: %d, 慣性: %.4f", self.model_type.replace("_", " ").title(), 
                    metric["n_clusters"], metric["inertia"])

        return {
            "X": X,
            "labels": self.model.labels_,
            "metric": metric,
            "report": report
        }

    def _run_agglomerative_clustering(self, X):
        """
        運行層次聚類。

        input:
            X (pandas.DataFrame): 輸入數據。
        output:
            訓練 AgglomerativeClustering 模型並生成聚類標籤。
        return:
            dict: 包含 "X", "labels", "metric", "report" 的字典。
        """
        start_time = time.time()
        self.model.fit(X)
        training_time = time.time() - start_time
        logger.info("%s 訓練完成，訓練時間: %.4f 秒", self.model_type.replace("_", " ").title(), training_time)
        metric = {"n_clusters": len(set(self.model.labels_) - {-1})}
        report = {"labels": str(self.model.labels_)}
        logger.info("%s 聚類數: %d", self.model_type.replace("_", " ").title(), metric["n_clusters"])

        return {
            "X": X,
            "labels": self.model.labels_,
            "metric": metric,
            "report": report
        }