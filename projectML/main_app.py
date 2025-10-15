# _main_app.py v20251004-2
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter import filedialog
from mlpackage.load_csv import LoadCsv
from mlpackage.data_preprocessor import preprocess_data
from mlpackage.ML_runner import MLRunner
from mlpackage.visualizer import Visualizer
import logging
import os
import re
import joblib
import time

# 全局日誌配置，設定級別為 INFO，輸出到控制台和 app.log 檔案
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

class MainApp:
    def __init__(self):
        """
        初始化主應用程式，設置數據和狀態變數並創建 GUI 視窗。

        input:
            無外部參數。
        output:
            初始化 self.df, self.df_data, self.df_target 等變數，創建 Tkinter 根視窗和 Notebook 結構。
        return:
            無返回值，初始化 MainApp 實例。
        """
        # 初始化資料和狀態變數
        self.df = None
        self.df_data = None
        self.df_target = None
        self.current_filename = "未載入"
        self.filename_labels = []
        self.page1_complete = False
        self.page2_complete = False
        self.page3_complete = False
        self.processed_data = None
        self.processed_target = None
        
        self.ml_runner = None
        self.model_type_var = None
        self.result_text = None
        self.ml_runner_results = {}
        logger.info("******************MainApp 初始化完成******************")

        """
        建立主視窗（root window）。
        Tk() 是整個 tkinter 應用的「根」。
        只能呼叫一次（應用通常只有一個 Tk()）；若要額外視窗用 tk.Toplevel()。
        """
        self.root = tk.Tk()
        self.root.title("CSV 機器學習工具") # 設定視窗標題列文字。
        self.root.geometry("800x800") # 設定視窗初始大小為 800×800（寬×高，單位像素）

        """
        建立一個 ttk.Notebook（分頁控制元件，tabbed interface），並把它的父元件設為 self.root。
        ttk 是 themed widgets（會跟系統樣式更一致）。
        """
        self.notebook = ttk.Notebook(self.root)
        """
        把 Notebook 加到視窗（使用 pack 排版器），參數含義：
        expand=True：允許在父容器中擴展（被分配額外空間時會放大）。
        fill="both"：在水平 & 垂直方向都填滿可用空間（與 expand 搭配常用）。
        padx=10, pady=10：Notebook 四周留 10px 的外距（水平與垂直）。
        """
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)
        """
        把事件綁到 Notebook：當分頁切換時會觸發 <<NotebookTabChanged>> 事件，然後呼叫 self.on_tab_change。
        重要：self.on_tab_change 必須是可呼叫的函式/方法，且通常簽名為 def on_tab_change(self, event):（至少一個 event 參數）。
        如果該方法不存在，或名稱打錯，程式在執行時會產生 AttributeError。
        event 物件可用來取得更多資訊（例如觸發該事件的 widget）。
        """
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

        """
        建一個 ttk.Frame 作為第一個分頁的內容。注意：Frame 的父元件是 notebook（所以這個 frame 會被 notebook 管理）。
        self.notebook.add(self.page1, text="1. 匯入 CSV")：把這個 frame 新增為一個 tab，tab 的標籤文字是 "1. 匯入 CSV"。
        self.setup_page1()：呼叫一個方法來在 page1 裡放置按鈕、Treeview、Label 等元件（你會把該分頁的 UI 與事件綁定寫在這個函式中）。
        接下來的 page2/page3/page4 都是同一模式：
        每個分頁各自一個 Frame，並呼叫對應的 setup 函式填入 UI 邏輯。這是清晰的分離方式（建議作法）。
        """
        self.page1 = ttk.Frame(self.notebook)
        self.notebook.add(self.page1, text="1. 匯入 CSV")
        self.setup_page1()

        self.page2 = ttk.Frame(self.notebook)
        self.notebook.add(self.page2, text="2. 預處理資料")
        self.setup_page2()

        self.page3 = ttk.Frame(self.notebook)
        self.notebook.add(self.page3, text="3. 運行模型")
        self.setup_page3()

        self.page4 = ttk.Frame(self.notebook)
        self.notebook.add(self.page4, text="4. 可視化")
        self.setup_page4()

        """
        選取（顯示）page1 為預設分頁。
        select() 可以接受 tab 的 index、tab 的 id，或是對應的子 widget（frame）本身；傳 self.page1 是可行的。
        也可以用 self.notebook.select(0) 選第一個分頁。
        """
        self.notebook.select(self.page1)

    def create_filename_label(self, parent_frame): # self 代表該類別實例，parent_frame 是傳入的父框架（通常是一個 Frame 或 ttk.Frame）。
        """
        這個方法的功能是：
        在指定的父框架（parent_frame）內建立一個 tk.Label
        顯示目前正在使用的 CSV 檔案名稱（從 self.current_filename 取得）
        並將這個 Label 返回給呼叫者，以便日後更新或操作
        創建並返回顯示檔案名稱的標籤。

        input:
            parent_frame (tk.Frame): 父框架，用於放置標籤。
        output:
            創建並返回一個 tk.Label 物件，顯示當前檔案名稱。
        return:
            filename_label (tk.Label): 包含檔案名稱的標籤物件。
        """
        """
        建立一個 Label 元件
        放在 parent_frame 裡
        顯示的文字是：檔案名稱: XXX.csv
        """
        filename_label = tk.Label(parent_frame, text=f"檔案名稱: {self.current_filename}", font=("Arial", 10))
        filename_label.pack(side=tk.LEFT, padx=10)
        return filename_label

    def setup_page1(self):
        """
        設置畫面 1，提供 CSV 檔案匯入功能。

        input:
            無外部參數，依賴 self.page1 和 self.filename_labels。
        output:
            設置畫面 1 的 UI 組件，包括標題、檔案名稱標籤、按鈕和狀態標籤。
        return:
            無返回值。
        """
        frame_title = tk.Frame(self.page1)
        frame_title.pack(pady=10)
        tk.Label(frame_title, text="請選擇 CSV 檔案以開始", font=("Arial", 12)).pack(side=tk.LEFT)
        filename_label = self.create_filename_label(frame_title)
        self.filename_labels.append(filename_label)

        logger.info("設置畫面 1: 匯入 CSV")
        load_button = tk.Button(self.page1, text="選擇 CSV 檔案", command=self.load_csv)
        logger.info("創建選擇 CSV 檔案按鈕")
        load_button.pack(pady=10)
        self.status_label = tk.Label(self.page1, text="", fg="green")
        self.status_label.pack(pady=10)

    def setup_page2(self):
        """
        設置畫面 2，提供特徵 (X) 和目標 (y) 選擇功能。

        input:
            無外部參數，依賴 self.page2 和 self.df。
        output:
            設置畫面 2 的 UI 組件，包括表格、下拉選單和按鈕。
        return:
            無返回值。
        """
        frame_title = tk.Frame(self.page2) # 建一個標題列 frame_title 放到 page2
        frame_title.pack(pady=5)
        tk.Label(frame_title, text="選擇特徵 (X) 和目標 (y)", font=("Arial", 10)).pack(side=tk.LEFT) # 顯示標題文字並呼叫 create_filename_label 顯示目前檔名（方便使用者知道正在編輯哪個 CSV）
        filename_label = self.create_filename_label(frame_title) # self.filename_labels 假設是一個 list，儲存所有 label 物件（以便後續更新/清除）。
        self.filename_labels.append(filename_label)
        
        """
        建一個 Treeview 用來顯示 DataFrame 的欄位清單（名稱、類型、非空值數）。
        show="headings" 表示只顯示欄位標題，不顯示最左邊的 tree column。
        height=20 指定顯示列數。
        selectmode="extended" 允許多選（方便選多個 X 欄位）。
        需注意：這段只建立 UI，沒把 DataFrame 的資料填入。通常在載入 CSV 後會呼叫一個 populate_columns_tree() 把 self.df 的欄位逐行 insert 進來。
        """
        self.columns_tree = ttk.Treeview(self.page2, columns=("Name", "Type", "Non-Null Count"), show="headings", height=10)
        self.columns_tree.heading("Name", text="欄位名稱")
        self.columns_tree.heading("Type", text="資料類型")
        self.columns_tree.heading("Non-Null Count", text="非Null計數")
        self.columns_tree.pack(pady=10, fill="x")
        self.columns_tree.configure(selectmode="extended")
        
        """
        顯示前5筆資料
        """
        # 獨立標題框架
        preview_title = tk.Frame(self.page2)
        preview_title.pack(pady=5)
        tk.Label(preview_title, text="前5筆預覽", font=("Arial", 10)).pack(side=tk.LEFT)
        
        # 初始化 preview_tree，暫不填充數據
        self.preview_tree = ttk.Treeview(self.page2, show="headings", height=5)
        self.preview_tree.pack(fill="x", padx=10)
        
        """
        建兩個按鈕：Select X 與 Select y，分別由 select_x / select_y 處理選擇行為。
        Select X 應該從 Treeview 的目前選取項目取多個欄位名稱（作為 X）；Select y 取單個欄位名稱（作為 y）。
        """
        frame_select = tk.Frame(self.page2)
        frame_select.pack(pady=10)
        tk.Button(frame_select, text="Select X", bg="orange", command=self.select_x).pack(side=tk.LEFT, padx=5)
        tk.Button(frame_select, text="Select y", bg="blue", fg="white", command=self.select_y).pack(side=tk.LEFT, padx=5)
        
        """
        用 BooleanVar 綁定一個勾選框，用來告知模型 y 是否為類別（classification）或數值（regression）。
        """
        self.is_categorical_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.page2, text="y 是類別 (預設是數值)", variable=self.is_categorical_var).pack(pady=5)
        
        """
        selection_label 顯示目前被選的 X 與 y（讓使用者有即時回饋）。
        確認並繼續 的按鈕會呼叫 confirm_and_next：應該在此檢查選取是否合理（X 不為空、y 存在且在 df 中、y 與 is_categorical_var 的一致性等），然後設 self.page2_complete = True 並導到下一頁。
        """
        self.selection_label = tk.Label(self.page2, text="選擇的 X: None\ny: None")
        self.selection_label.pack(pady=10)
        tk.Button(self.page2, text="確認並繼續", command=self.confirm_and_next).pack(pady=10)

    def setup_page3(self):
        """
        設置畫面 3，提供模型選擇和運行功能。

        input:
            無外部參數，依賴 self.page3。
        output:
            設置畫面 3 的 UI 組件，包括模型選單、按鈕和結果顯示區域。
        return:
            無返回值。
        """
        frame_title = tk.Frame(self.page3)
        frame_title.pack(pady=10)
        tk.Label(frame_title, text="選擇並運行 ML 模型", font=("Arial", 12)).pack(side=tk.LEFT)
        filename_label = self.create_filename_label(frame_title)
        self.filename_labels.append(filename_label)

        self.model_type_var = tk.StringVar(value="Linear_Regression")
        models = [
            "Linear_Regression", "Multiple_Regression", "Polynomial_Regression",
            "Logistic_Regression", "Decision_Tree", "KNN", "SVM",
            "Bagging_Classifier", "Random_Forest_Classifier", "Random_Forest_Regressor",
            "DBSCAN", "Kmeans", "Agglomerative_Clustering"
        ]
        tk.Label(self.page3, text="選擇模型:").pack(pady=5)
        model_menu = tk.OptionMenu(self.page3, self.model_type_var, *models)
        model_menu.pack(pady=5)

        tk.Button(self.page3, text="確認並運行", command=self.run_model).pack(pady=10)

        self.result_text = tk.Text(self.page3, height=10, width=80)
        self.result_text.pack(pady=10, fill="both", expand=True)
        self.result_text.insert(tk.END, "結果將顯示在此處...\n")

        scrollbar = tk.Scrollbar(self.page3)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.result_text.yview)

        tk.Button(self.page3, text="進入可視化", command=self.switch_to_screen4).pack(pady=10)

    def setup_page4(self):
        """
        設置畫面 4，提供模型結果的可視化功能。

        input:
            無外部參數，依賴 self.page4。
        output:
            設置畫面 4 的 UI 組件，包括圖型選單、按鈕和結果標籤。
        return:
            無返回值。
        """
        frame_title = tk.Frame(self.page4)
        frame_title.pack(pady=10)
        tk.Label(frame_title, text="模型結果與可視化", font=("Arial", 12)).pack(side=tk.LEFT)
        filename_label = self.create_filename_label(frame_title)
        self.filename_labels.append(filename_label)

        self.plot_var = tk.StringVar()
        label = tk.Label(self.page4, text="選擇圖型:")
        label.pack(pady=5)
        self.plot_options = []
        try:
            self.plot_menu = ttk.OptionMenu(self.page4, self.plot_var, self.plot_options[0] if self.plot_options else "", *self.plot_options)
            self.plot_menu.pack(pady=5)
            logger.info("plot_menu 初始化完成")
        except Exception as e:
            logger.error(f"plot_menu 初始化失敗: {e}")

        confirm_button = tk.Button(self.page4, text="確定", command=self.on_plot_select)
        confirm_button.pack(pady=10)

        export_button = tk.Button(self.page4, text="匯出訓練模型和圖形", command=self.export_model_and_plot)
        export_button.pack(pady=10)

        self.result_label = tk.Label(self.page4, text="結果將顯示在此", wraplength=700)
        self.result_label.pack(pady=10)

    def load_csv(self):
        """
        處理 CSV 檔案的載入並更新狀態。

        input:
            無直接參數，通過 filedialog.askopenfilename 獲取 filepath。
        output:
            更新 self.df (pandas.DataFrame)、self.current_filename 和 self.filename_labels 顯示。
            更新 self.status_label 文本，標記 self.page1_complete。
        return:
            無返回值。
        """
        root = tk.Tk()
        root.withdraw()
        filepath = filedialog.askopenfilename(
            title="選擇CSV檔案",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        root.destroy()

        if filepath:
            self.df = LoadCsv().load_csv_file(filepath)
            if self.df is not None:
                self.status_label.config(text="CSV 載入成功！", fg="green")
                self.current_filename = os.path.basename(filepath)
                for label in self.filename_labels:
                    label.config(text=f"檔案名稱: {self.current_filename}")
                self.update_columns()
                self.update_preview_tree()
                self.page1_complete = True
                self.notebook.select(2)
                logger.info(f"MainApp.load_csv成功讀取檔案: {filepath}")
            else:
                self.status_label.config(text="CSV 載入失敗！", fg="red")
        else:
            self.status_label.config(text="未選擇檔案", fg="red")

    def update_columns(self):
        """
        更新畫面 2 的欄位表格，顯示 CSV 資料結構。

        input:
            依賴 self.df (pandas.DataFrame)。
        output:
            更新 self.columns_tree 顯示欄位名稱、資料類型和非 Null 計數。
        return:
            無返回值。
        """
        self.columns_tree.delete(*self.columns_tree.get_children())
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            non_null_count = self.df[col].count()
            self.columns_tree.insert("", "end", values=(col, dtype, non_null_count))
    
    def update_preview_tree(self):
        self.preview_tree.delete(*self.preview_tree.get_children())
        if hasattr(self, 'df') and self.df is not None and not self.df.empty:
            # 動態定義列
            self.preview_tree["columns"] = self.df.columns.tolist()
            # for col in self.df.columns:
            #     self.preview_tree.heading(col, text=col)
            
            # 計算每個欄位的最長字元長度（基於前5行和欄位名稱）
            max_lengths = {}
            preview_data = self.df.head(5)
            for col in self.df.columns:
                # 取前5行數據的最大長度和欄位名稱長度的最大值
                max_length = max(preview_data[col].astype(str).str.len().max(), len(col))
                max_lengths[col] = max_length * 8  # 將字元數轉為像素（約8像素/字元，依字體調整）

            # 設置列標題和寬度
            for col in self.df.columns:
                self.preview_tree.heading(col, text=col)
                self.preview_tree.column(col, width=max_lengths[col], minwidth=50)  # 最小寬度50像素
                
            # 插入前5行數據
            for index, row in self.df.head(5).iterrows():
                self.preview_tree.insert("", "end", values=list(row))
        else:
            self.preview_tree["columns"] = ("狀態",)
            self.preview_tree.heading("狀態", text="狀態")
            self.preview_tree.insert("", "end", values=("無數據",))

    def select_x(self):
        """
        處理多個 X 變數的選擇。

        input:
            依賴 self.columns_tree.selection() 和 self.df。
        output:
            更新 self.df_data (pandas.DataFrame)、self.selection_label 文本。
        return:
            無返回值。
        """
        selected_items = self.columns_tree.selection()
        if selected_items and self.df is not None:
            self.df_data = self.df[[self.columns_tree.item(item)["values"][0] for item in selected_items]]
            self.selection_label.config(text=f"選擇的 X: {', '.join(self.df_data.columns)}\ny: {self.df_target.name if self.df_target is not None else 'None'}")

    def select_y(self):
        """
        處理單一 y 變數的選擇。

        input:
            依賴 self.columns_tree.selection() 和 self.df。
        output:
            更新 self.df_target (pandas.Series)、self.selection_label 文本。
        return:
            無返回值。
        """
        selected_items = self.columns_tree.selection()
        if selected_items and self.df is not None:
            self.df_target = self.df[self.columns_tree.item(selected_items[0])["values"][0]]
            self.columns_tree.selection_remove(*selected_items[1:])
            self.selection_label.config(text=f"選擇的 X: {', '.join(self.df_data.columns) if self.df_data is not None else 'None'}\ny: {self.df_target.name}")

    def confirm_and_next(self):
        """
        確認選擇並切換到下一畫面。

        input:
            依賴 self.df_data, self.df_target, self.is_categorical_var。
        output:
            更新 self.processed_data (字典)、self.processed_target (pandas.Series)。
            設置 self.page2_complete，切換到畫面 3。
        return:
            無返回值。
        """
        logger.info("準備進行資料處理")
        if self.notebook.tab(self.notebook.select(), "text") == "2. 預處理資料":
            if self.df_data is not None and self.df_target is not None:
                selected_x_columns = self.df_data.columns.tolist()
                selected_y_column = self.df_target.name
                try:
                    processed_minmax_data, processed_standard_data, processed_df_target, processed_df_data = preprocess_data(
                        self.df, self.df_data.join(self.df_target), selected_x_columns, selected_y_column, self.is_categorical_var.get()
                    )
                    self.processed_data = {'minmax': processed_minmax_data, 'standard': processed_standard_data}
                    self.processed_target = processed_df_target
                    self.processed_df_data = processed_df_data
                    logger.info(f"預處理完成：MinMax 版本形狀 {processed_minmax_data.shape}, Standard 版本形狀 {processed_standard_data.shape}, df_target 形狀 {processed_df_target.shape}")
                    self.page2_complete = True
                    self.notebook.select(3)
                except Exception as e:
                    logger.error(f"資料預處理失敗: {str(e)}")
                    messagebox.showwarning("警告", "資料預處理失敗，請檢查輸入數據")
            else:
                messagebox.showwarning("警告", "請至少選擇一個 X 和 y 欄位")

    def run_model(self):
        """
        執行選定模型並將結果顯示在畫面 3，存儲結果字典。

        input:
            依賴 self.processed_data, self.processed_target, self.model_type_var。
        output:
            更新 self.ml_runner_results (字典)、self.result_text (tk.Text) 內容。
            設置 self.page3_complete。
        return:
            無返回值。
        """
        if self.processed_data is None or self.processed_target is None:
            messagebox.showwarning("警告", "請先在畫面 2 完成資料預處理")
            return
    
        model_type = self.model_type_var.get().lower().replace(" ", "_")
        
        try:
            self.ml_runner = MLRunner(self.processed_data, self.processed_target, model_type)
            logger.info(f"run_model: processed_data={self.processed_data}, processed_target={self.processed_target}")
            self.ml_runner_results = self.ml_runner.run()
            logger.info(f"{self.ml_runner_results}")
            
            if self.ml_runner_results.get("status") != "completed" or not isinstance(self.ml_runner_results, dict):
                raise ValueError("模型執行未正常完成或返回無效結果")
            
            with open('app.log', 'r', encoding='utf-8') as file:
                lines = file.readlines()
                relevant_logs = []
                start_index = -1
                end_index = len(lines)
                
                for i, line in enumerate(lines):
                    if "開始準備資料，模型類型: " in line and model_type in line.lower():
                        start_index = i
                    if "流程執行完成" in line and start_index != -1:
                        end_index = i + 1
                
                if start_index != -1:
                    relevant_logs = [line.strip() for line in lines[start_index:end_index]]
                else:
                    relevant_logs = ["警告: 未找到模型執行記錄"]
            
            cleaned_logs = [re.sub(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - ', '', line) for line in relevant_logs]
            
            self.result_text.insert(tk.END, "結果將顯示在此處...\n")
            self.result_text.insert(tk.END, f"--- {self.model_type_var.get()} 執行結果 ---\n")
            for log in cleaned_logs:
                self.result_text.insert(tk.END, log + "\n")
            
            line_count = int(self.result_text.index('end-1c').split('.')[0])
            if line_count > 500:
                self.result_text.delete(1.0, f"{line_count - 500}.0")
            
            self.run_count = getattr(self, 'run_count', 0) + 1
            if self.run_count >= 10:
                self.run_count = 0
            self.page3_complete = True
            
        except Exception as e:
            logger.error(f"模型執行錯誤: {e}")
            self.result_text.insert(tk.END, f"--- {self.model_type_var.get()} 執行結果 ---\n")
            self.result_text.insert(tk.END, f"ERROR - 模型執行錯誤: {e}\n")

    def visualize_plot(self, plot_type):
        """
        調用 Visualizer 繪製圖形。

        input:
            plot_type (str): 圖形類型（如 "Scatter", "Line" 等）。
        output:
            創建 Visualizer 實例並繪製指定圖形。
        return:
            無返回值。
        """
        try:
            logger.info(f"初始化 Visualizer 為 {self.model_type_var.get()}, 數據: {self.ml_runner_results.keys()}")
            viz = Visualizer(self.model_type_var.get(), self.ml_runner_results, self.ml_runner.model)
            viz.visualize(plot_type)
            logger.info(f"完成繪製 {plot_type} for {self.model_type_var.get()}")
        except ImportError as e:
            logger.error(f"導入 Visualizer 失敗: {e}")
        except Exception as e:
            logger.error(f"視覺化錯誤: {e}")

    def on_plot_select(self):
        """
        處理用戶選擇圖型的動作，觸發視覺化。

        input:
            無直接參數，依賴 self.plot_var。
        output:
            調用 visualize_plot 繪製選定圖形。
        return:
            無返回值。
        """
        selected_plot = self.plot_var.get()
        logger.info(f"用戶選擇圖型: {selected_plot} for {self.model_type_var.get()}")
        if not self.ml_runner_results:
            logger.warning("ml_runner_results 為空")
            return
        self.visualize_plot(selected_plot)

    def switch_to_screen4(self):
        """
        切換到畫面 4，顯示圖型選項。

        input:
            無外部參數。
        output:
            切換到 self.page4，更新圖型選項。
        return:
            無返回值。
        """
        self.notebook.select(self.page4)
        self.update_plot_options()

    def update_plot_options(self):
        """
        根據當前模型更新下拉選單的圖型選項。

        input:
            無外部參數，依賴 self.model_type_var。
        output:
            更新 self.plot_options 和 self.plot_menu 內容。
        return:
            無返回值。
        """
        if not hasattr(self, 'plot_menu') or self.plot_menu is None:
            logger.error("plot_menu 未初始化")
            return
        model_type = self.model_type_var.get().replace("_", "").lower()
        logger.info(f"更新圖型選項，當前模型: {model_type}")
        if model_type in ["linearregression", "multipleregression", "polynomialregression"]:
            self.plot_options = ["Scatter", "Line", "ResidualPlot"]
        elif model_type == "logisticregression":
            self.plot_options = ["Scatter", "ROCCurve", "ConfusionMatrix"]
        elif model_type == "decisiontree":
            self.plot_options = ["TreeDiagram", "FeatureImportance", "DecisionBoundary"]
        elif model_type == "knn":
            self.plot_options = ["Scatter", "DecisionBoundary", "KValuePlot"]
        elif model_type == "svm":
            self.plot_options = ["Scatter", "DecisionBoundary", "ROCCurve"]
        elif model_type == "baggingclassifier":
            self.plot_options = ["ErrorPlot", "FeatureImportance", "ConfusionMatrix"]
        elif model_type == "randomforestclassifier":
            self.plot_options = ["FeatureImportance", "ErrorPlot", "ConfusionMatrix"]
        elif model_type == "randomforestregressor":
            self.plot_options = ["FeatureImportance", "ErrorPlot"]
        elif model_type == "dbscan":
            self.plot_options = ["Scatter", "ElbowPlot"]
        elif model_type == "kmeans":
            self.plot_options = ["Scatter", "ElbowPlot"]
        elif model_type == "agglomerativeclustering":
            self.plot_options = ["Scatter", "Dendrogram"]
        else:
            self.plot_options = []
        logger.info(f"更新後的圖型選項: {self.plot_options}")
        self.plot_var.set(self.plot_options[0] if self.plot_options else "")
        menu = self.plot_menu['menu']
        menu.delete(0, 'end')
        for option in self.plot_options:
            menu.add_command(label=option, command=lambda x=option: self.plot_var.set(x))

    def export_model_and_plot(self):
        """
        匯出訓練模型和當前圖形。

        input:
            無直接參數，依賴 self.ml_runner 和 self.plot_var。
        output:
            將模型和圖形匯出至指定路徑，顯示成功或錯誤訊息。
        return:
            無返回值。
        """
        if not self.ml_runner or not self.ml_runner.model:
            messagebox.showwarning("警告", "無可匯出的模型，請先運行模型")
            logger.warning("嘗試匯出模型但 ml_runner.model 為空")
            return

        export_dir = filedialog.asksaveasfilename(
            title="選擇匯出路徑",
            defaultextension=".zip",
            filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")],
            initialfile=f"{self.model_type_var.get()}_{time.strftime('%Y%m%d_%H%M%S')}"
        )

        if export_dir:
            try:
                model_path = os.path.join(os.path.dirname(export_dir), f"{self.model_type_var.get()}_model.joblib")
                joblib.dump(self.ml_runner.model, model_path)
                logger.info(f"模型已匯出至: {model_path}")

                plot_type = self.plot_var.get()
                if plot_type:
                    viz = Visualizer(self.model_type_var.get(), self.ml_runner_results)
                    plot_path = os.path.join(os.path.dirname(export_dir), f"{self.model_type_var.get()}_{plot_type}.png")
                    viz.visualize(plot_type, save_path=plot_path)
                    logger.info(f"圖形已匯出至: {plot_path}")
                else:
                    logger.warning("無選定圖型，僅匯出模型")

                messagebox.showinfo("成功", f"模型和圖形已匯出至: {os.path.dirname(export_dir)}")
            except Exception as e:
                logger.error(f"匯出失敗: {e}")
                messagebox.showerror("錯誤", f"匯出失敗: {e}")

    def on_tab_change(self, event):
        """
        事件處理函式。
        處理畫面切換事件，檢查前置條件。
        當使用者在 Notebook（分頁式介面）中切換頁籤時，就會自動呼叫這個方法。

        input:
            event (tk.Event): Notebook 切換事件。
        output:
            根據當前畫面檢查前置條件，必要時切回前一畫面。
        return:
            無返回值。
        """
        """
        取得目前的分頁名稱
        self.notebook.select()：取得目前被選中的頁籤 ID。
        self.notebook.tab(..., "text")：取得該分頁的標題文字（例如 "2. 預處理資料"）。
        """
        current_tab = self.notebook.tab(self.notebook.select(), "text")
        if current_tab == "1. 匯入 CSV":
            return
        elif current_tab == "2. 預處理資料" and not self.page1_complete:
            self.notebook.select(0)
        elif current_tab == "3. 運行模型" and not self.page2_complete:
            self.notebook.select(1)
        elif current_tab == "4. 可視化" and not self.page3_complete:
            self.notebook.select(2)

    def run(self):
        """
        啟動主應用程式主迴圈。

        input:
            無外部參數，依賴 self.root。
        output:
            啟動 Tkinter 事件迴圈，運行 GUI 應用程式。
        return:
            無返回值。
        """
        self.root.mainloop()
"""
在使用 Tkinter GUI 的程式中，root.mainloop() 幾乎是必須的，因為它負責啟動事件循環
（event loop），讓你的視窗能夠互動、回應使用者操作。

🔍 什麼是 root.mainloop()？

mainloop() 是 Tkinter 的事件監聽迴圈，用來：
保持視窗打開
監聽滑鼠點擊、鍵盤輸入、按鈕點擊等事件
執行對應的事件處理函式（callback）
讓視窗可以動態更新（例如 label 文字變更）

🧠 沒有 mainloop() 會怎樣？

如果你不呼叫 mainloop()，雖然視窗物件 (Tk() 或 Toplevel()) 會建立，但它會：
一閃即逝（視窗一打開就關掉）
或 根本沒顯示出來
不會回應任何事件（例如按鈕不會動）
原因是 Python 程式會跑完就結束，沒有 mainloop() 它不知道要「停在那裡等你操作」。

問題	                               解釋
GUI 一定需要 mainloop() 嗎？	✅ 是的，除非你用其他工具來控制事件循環（進階用法）
沒有它會怎樣？	            GUI 不會互動、立刻關閉，無法使用
可以有多個 mainloop() 嗎？	❌ 通常只允許有一個主 mainloop()。次視窗用 Toplevel()，不要再開第二個 mainloop()。
"""