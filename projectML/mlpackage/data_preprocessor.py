# data_preprocessor.py v20251004-2
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import pandas as pd
import numpy as np
import logging
from io import StringIO
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

def convert_height_to_cm(height_str):
    """
    將 '6-2' 格式的身高轉為厘米。

    input:
        height_str (str or float): 身高字符串（如 '6-2'）或 NaN。
    output:
        將身高轉換為厘米並返回。
    return:
        float: 轉換後的身高（厘米），若失敗返回 None。
    """
    if pd.isna(height_str):
        return None
    try:
        feet, inches = map(int, height_str.split('-'))
        total_inches = (feet * 12) + inches
        return total_inches * 2.54
    except (ValueError, AttributeError):
        logger.warning(f"無法處理身高格式: {height_str}, 返回 None")
        return None

def convert_date_to_features(date_str, formats=["%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y", "%Y/%m/%d", "%m/%d/%Y"]):
    """
    將日期字符串轉為年、月、日特徵，支持多種格式並保留時間（若存在）。

    input:
        date_str (str or float): 日期字符串或 NaN。
        formats (list): 支持的日期格式列表。
    output:
        將日期分解為年、月、日、時特徵。
    return:
        pd.Series: 包含 [year, month, day, hour] 的 Series，NaN 若解析失敗。
    """
    if pd.isna(date_str):
        return pd.Series([np.nan, np.nan, np.nan, np.nan])
    for fmt in formats:
        try:
            dt = pd.to_datetime(date_str, format=fmt, errors='coerce')
            if pd.notna(dt):
                year = dt.year if pd.notna(dt.year) else np.nan
                month = dt.month if pd.notna(dt.month) else np.nan
                day = dt.day if pd.notna(dt.day) else np.nan
                hour = dt.hour if pd.notna(dt.hour) else np.nan
                return pd.Series([year, month, day, hour])
        except ValueError:
            continue
    logger.warning(f"無法解析日期格式: {date_str}, 返回 [NaN, NaN, NaN, NaN]")
    return pd.Series([np.nan, np.nan, np.nan, np.nan])

def preprocess_data(df, df_selected, selected_x_columns, selected_y_column, is_categorical):
    """
    預處理資料，處理 object 類型的文字欄位，應用正規化和降維。

    input:
        df (pandas.DataFrame): 原始 CSV 數據。
        df_selected (pandas.DataFrame): 用戶選取的 X 和 y 數據。
        selected_x_columns (list): 選取的 X 欄位名稱列表。
        selected_y_column (str): 選取的 y 欄位名稱。
        is_categorical (bool): y 是否為類別變量。
    output:
        更新 processed_df_selected 並分離為 X 和 y，應用正規化和降維。
    return:
        tuple: (minmax_data, standard_data, processed_df_target, processed_df_data)
            - minmax_data (pandas.DataFrame): MinMax 正規化後的數據。
            - standard_data (pandas.DataFrame): Standard 正規化後的數據。
            - processed_df_target (pandas.Series): 處理後的目標變量。
            - processed_df_data (pandas.DataFrame): 處理後的 X 數據。
    """
    logger.info("=== 開始資料預處理 ===")

    logger.info("Step 1: 確認 raw csv data 已讀入 df")
    logger.info(f"輸入 df 形狀: {df.shape}")
    logger.info(f"df 數據樣本:\n{df.head()}")

    logger.info("Step 2: 確認 user 選取的 X & y 數據")
    logger.info(f"輸入 df_selected 形狀: {df_selected.shape}")
    logger.info(f"df_selected 數據樣本:\n{df_selected.head()}")
    if selected_y_column not in df_selected.columns:
        logger.error("選定的 y 欄位不存在")
        raise ValueError("選定的 y 欄位不存在")

    logger.info("Step 3: 移除 NaN 並保存到 df_selected_clean")
    df_selected_clean = df_selected.dropna()
    nan_rows_removed = len(df_selected) - len(df_selected_clean)
    logger.info(f"移除 NaN 後行數減少: {nan_rows_removed} 行 ({nan_rows_removed/len(df_selected)*100:.2f}%)")
    logger.info(f"df_selected_clean 形狀: {df_selected_clean.shape}")
    logger.info(f"df_selected_clean 數據樣本:\n{df_selected_clean.head()}")

    logger.info("Step 4: 處理特殊數據")
    processed_df_selected = df_selected_clean.copy()
    object_columns = processed_df_selected.select_dtypes(include=['object']).columns
    logger.info(f"檢測到的 object 欄位: {list(object_columns)}")

    for col in object_columns:
        processed_df_selected[col] = processed_df_selected[col].fillna("未知")
        logger.info(f"處理欄位 {col}: 填補 NaN 為 '未知'")

        if col.lower() in ['height']:
            processed_df_selected[col] = processed_df_selected[col].apply(convert_height_to_cm)
            logger.info(f"欄位 {col} 轉為厘米，唯一值數: {processed_df_selected[col].nunique()}")
        elif col.lower() in ['weight']:
            processed_df_selected[col] = processed_df_selected[col].str.extract(r'(\d+\.?\d*)').astype(float)
            processed_df_selected[col] = processed_df_selected[col].fillna(processed_df_selected[col].mean())
            logger.info(f"欄位 {col} 提取數值並填補均值，唯一值數: {processed_df_selected[col].nunique()}")
        elif col.lower() in ['date', 'datetime']:
            date_features = processed_df_selected[col].apply(lambda x: convert_date_to_features(x))
            processed_df_selected = pd.concat([
                processed_df_selected.drop(col, axis=1),
                date_features.rename(columns={0: f"{col}_year", 1: f"{col}_month", 2: f"{col}_day", 3: f"{col}_hour"})
            ], axis=1)
            logger.info(f"欄位 {col} 轉為日期特徵，生成欄位: {col}_year, {col}_month, {col}_day, {col}_hour")
        elif col.lower() in ['name', 'person']:
            le = LabelEncoder()
            processed_df_selected[col] = le.fit_transform(processed_df_selected[col])
            logger.info(f"欄位 {col} 應用 LabelEncoder，唯一值數: {processed_df_selected[col].nunique()}")
        elif col.lower() in ['location', 'city', 'address']:
            if processed_df_selected[col].str.contains(',').any():
                split_cols = processed_df_selected[col].str.split(',', expand=True)
                for i, split_col in enumerate(split_cols.columns):
                    new_col_name = f"{col}_part{i}"
                    one_hot = pd.get_dummies(split_cols[split_col].fillna("未知"), prefix=new_col_name)
                    processed_df_selected = pd.concat([processed_df_selected.drop(col, axis=1), one_hot], axis=1)
                logger.info(f"欄位 {col} 應用 one-hot 編碼，分拆後欄位數: {len(one_hot.columns)}")
            else:
                le = LabelEncoder()
                processed_df_selected[col] = le.fit_transform(processed_df_selected[col])
                logger.info(f"欄位 {col} 應用 LabelEncoder，唯一值數: {processed_df_selected[col].nunique()}")
        else:
            unique_values = processed_df_selected[col].nunique()
            if unique_values < 10:
                one_hot = pd.get_dummies(processed_df_selected[col], prefix=col)
                processed_df_selected = pd.concat([processed_df_selected.drop(col, axis=1), one_hot], axis=1)
                logger.info(f"欄位 {col} 應用 one-hot 編碼，生成欄位數: {len(one_hot.columns)}")
            else:
                le = LabelEncoder()
                processed_df_selected[col] = le.fit_transform(processed_df_selected[col])
                logger.info(f"欄位 {col} 應用 LabelEncoder，唯一值數: {processed_df_selected[col].nunique()}")

    processed_df_selected = processed_df_selected.dropna()
    logger.info(f"特殊數據處理後 processed_df_selected 形狀: {processed_df_selected.shape}")

    logger.info("Step 5: 根據 X & y 分割 processed_df_selected")
    processed_df_data = processed_df_selected[selected_x_columns]
    processed_df_target = processed_df_selected[selected_y_column]

    if is_categorical and processed_df_target.dtype == 'object':
        le = LabelEncoder()
        processed_df_target = pd.Series(le.fit_transform(processed_df_target), index=processed_df_target.index)
        logger.info("y 為類別，應用 LabelEncoder")
    elif not is_categorical and processed_df_target.name.lower() == 'salary':
        processed_df_target = np.log1p(processed_df_target)
        logger.info("y 為數值，應用 log1p 轉換")
    logger.info(f"processed_df_data 形狀: {processed_df_data.shape}, processed_df_target 形狀: {processed_df_target.shape}")

    logger.info("Step 6: 應用 MinMaxScaler 和 StandardScaler")
    numeric_columns = processed_df_data.select_dtypes(include=['int64', 'float64']).columns
    minmax_data = processed_df_data.copy()
    standard_data = processed_df_data.copy()

    if not numeric_columns.empty:
        logger.info("正規化前數值欄位統計:")
        for col in numeric_columns:
            logger.info(f"欄位 {col}: 最小值 {processed_df_data[col].min()}, 最大值 {processed_df_data[col].max()}")

        minmax_scaler = MinMaxScaler()
        minmax_data[numeric_columns] = minmax_scaler.fit_transform(minmax_data[numeric_columns])
        logger.info("MinMaxScaler 應用後數值欄位統計:")
        for col in numeric_columns:
            logger.info(f"欄位 {col}: 最小值 {minmax_data[col].min()}, 最大值 {minmax_data[col].max()}")

        standard_scaler = StandardScaler()
        standard_data[numeric_columns] = standard_scaler.fit_transform(standard_data[numeric_columns])
        logger.info("StandardScaler 應用後數值欄位統計:")
        for col in numeric_columns:
            logger.info(f"欄位 {col}: 最小值 {standard_data[col].min()}, 最大值 {standard_data[col].max()}")

    logger.info(f"正規化後 minmax_data 形狀: {minmax_data.shape}")
    logger.info(f"正規化後 standard_data 形狀: {standard_data.shape}")

    logger.info("Step 7: 進行關聯性判斷、降維和異常值檢測")
    if False:
        if len(minmax_data.columns) > 10:
            logger.info(f"關聯性判斷前特徵數: {len(minmax_data.columns)}")
            selector = SelectKBest(score_func=mutual_info_regression, k=10)
            selector.fit(minmax_data, processed_df_target)
            selected_features = minmax_data.columns[selector.get_support()].tolist()
            logger.info(f"關聯性判斷後選擇的特徵: {selected_features}")
            minmax_data = minmax_data[selected_features]
            standard_data = standard_data[selected_features]
            logger.info(f"關聯性判斷後特徵數: {len(minmax_data.columns)}")

        if len(minmax_data.columns) > 20:
            logger.info(f"降維前特徵數: {len(minmax_data.columns)}")
            pca = PCA(n_components=0.95)
            minmax_data = pd.DataFrame(pca.fit_transform(minmax_data), index=minmax_data.index, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
            standard_data = pd.DataFrame(pca.fit_transform(standard_data), index=standard_data.index, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
            logger.info(f"降維後特徵數: {minmax_data.shape[1]}, 解釋方差比: {sum(pca.explained_variance_ratio_):.4f}")

        Q1 = processed_df_target.quantile(0.25)
        Q3 = processed_df_target.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = processed_df_target[(processed_df_target < lower_bound) | (processed_df_target > upper_bound)]
        logger.info(f"檢測到 {len(outliers)} 個目標變數異常值，範圍: [{lower_bound:.2f}, {upper_bound:.2f}]")
        mask = (processed_df_target >= lower_bound) & (processed_df_target <= upper_bound)
        processed_df_target = processed_df_target[mask]
        minmax_data = minmax_data.loc[mask]
        standard_data = standard_data.loc[mask]
        logger.info(f"移除異常值後 processed_df_target 形狀: {processed_df_target.shape}")
    else:
        logger.info("關聯性判斷、降維和異常值檢測已禁用")

    logger.info("=== 最終資料資訊 ===")
    logger.info(f"最終 MinMax 版本 processed_df_data 形狀: {minmax_data.shape}")
    logger.info(f"最終 MinMax 版本 processed_df_data:\n{minmax_data}")
    logger.info(f"最終 Standard 版本 processed_df_data 形狀: {standard_data.shape}")
    logger.info(f"最終 Standard 版本 processed_df_data:\n{standard_data}")
    logger.info(f"最終 processed_df_target 形狀: {processed_df_target.shape}")
    logger.info(f"最終 processed_df_target:\n{processed_df_target}")

    info_buffer = StringIO()
    minmax_data.info(buf=info_buffer)
    logger.info("最終 MinMax 版本 processed_df_data 資訊:\n" + info_buffer.getvalue())
    info_buffer = StringIO()
    standard_data.info(buf=info_buffer)
    logger.info("最終 Standard 版本 processed_df_data 資訊:\n" + info_buffer.getvalue())
    info_buffer = StringIO()
    processed_df_target.info(buf=info_buffer)
    logger.info("最終 processed_df_target 資訊:\n" + info_buffer.getvalue())

    return minmax_data, standard_data, processed_df_target, processed_df_data