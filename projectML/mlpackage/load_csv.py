# load_csv.py v20251004-2
import pandas as pd
from charset_normalizer import from_path
import os
import csv
import logging

logger = logging.getLogger(__name__)

class LoadCsv:
    def __init__(self):
        """
        初始化 LoadCsv 類，設置檔案處理相關屬性。

        input:
            無外部參數。
        output:
            初始化 self.filepath, self.encoding, self.delimiter, self.df 為預設值。
        return:
            無返回值，初始化 LoadCsv 實例。
        """
        self.filepath = None
        self.encoding = 'utf-8'
        self.delimiter = ','
        self.df = None

    def detect_encoding(self, filepath):
        """
        自動偵測檔案的編碼。

        input:
            filepath (str): 檔案路徑。
        output:
            更新 self.encoding 為檢測到的編碼。
        return:
            self.encoding (str): 檢測到的編碼（若失敗則返回 'utf-8'）。
        """
        if not filepath or not os.path.exists(filepath):
            logger.warning("檔案路徑無效")
            return self.encoding
        result = from_path(filepath).best()
        self.encoding = result.encoding if result and result.encoding else 'utf-8'
        logger.info(f"檢測到的編碼: {self.encoding}")
        return self.encoding

    def detect_delimiter(self, filepath):
        """
        自動偵測 CSV 檔案的分隔符號。

        input:
            filepath (str): 檔案路徑。
        output:
            更新 self.delimiter 為檢測到的分隔符號。
        return:
            self.delimiter (str): 檢測到的分隔符號（若失敗則返回 ','）。
        """
        if not filepath or not os.path.exists(filepath):
            logger.warning("檔案路徑無效")
            return self.delimiter
        try:
            with open(filepath, 'r', encoding=self.encoding) as f:
                sample = f.read(4096)
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample, delimiters=[',', ';', '\t', ' ', '|'])
                self.delimiter = dialect.delimiter
        except csv.Error as e:
            logger.warning(f"分隔符檢測失敗，採用預設逗號: {e}")
            self.delimiter = ','
        logger.info(f"檢測到的分隔符: '{self.delimiter}'")
        return self.delimiter

    def load_csv_file(self, filepath):
        """
        使用指定路徑讀取 CSV 檔案。

        input:
            filepath (str): 檔案路徑。
        output:
            更新 self.filepath 和 self.df 為載入的數據。
        return:
            self.df (pandas.DataFrame): 載入的數據，失敗時返回 None。
        """
        logger.info("進入load_csv_file")
        if not filepath or not os.path.exists(filepath):
            logger.info("檔案路徑無效或未提供")
            return None

        self.filepath = filepath

        try:
            self.detect_encoding(filepath)
            self.detect_delimiter(filepath)
            self.df = pd.read_csv(filepath, delimiter=self.delimiter, encoding=self.encoding,
                                low_memory=False, on_bad_lines='warn')
            logger.info(f"LoadCsv.load_csv_file成功讀取檔案: {filepath}")
            logger.info(f"{self.df}")
            return self.df
        except FileNotFoundError:
            logger.error(f"檔案未找到: {filepath}")
            return None
        except UnicodeDecodeError as e:
            logger.error(f"編碼錯誤: {e}")
            return None
        except pd.errors.ParserError as e:
            logger.error(f"CSV 解析錯誤: {e}")
            return None
        except Exception as e:
            logger.error(f"未知錯誤: {e}")
            return None

if __name__ == "__main__":
    """
    測試 LoadCsv 類的功能。

    input:
        無外部參數，假設 "example.csv" 存在。
    output:
        創建 LoadCsv 實例並嘗試載入 "example.csv"。
    return:
        無返回值，輸出載入結果。
    """
    loader = LoadCsv()
    df = loader.load_csv_file("example.csv")
    if df is not None:
        print("載入成功，DataFrame 可用")