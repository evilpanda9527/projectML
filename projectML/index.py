#index.py
"""
[開始]
   |
   v
[if __name__ == "__main__":]
   | (條件: 直接執行檔案)
   v
[創建 MainApp 實例] --> app = MainApp()
   | (初始化 GUI 根視窗)
   v
[執行 app.run()] --> self.root.mainloop()
   | (進入事件迴圈)
   v
[畫面 1-4 互動] --> (處理用戶輸入: 匯入 CSV -> 選擇 columns -> 顯示模型成績 -> 繪製圖形)
   |
   v
[結束] (用戶關閉視窗)
"""
"""
畫面 1：匯入 CSV (步驟 1)。
畫面 2：顯示 columns 和 types，讓 user 選擇 (你的圖)。
畫面 3：顯示模型成績 (步驟 5)，用 Table 或 Label。
畫面 4：選擇模型並顯示圖形 (步驟 6)，用 matplotlib 嵌入 tkinter (Canvas)。
"""
from main_app import MainApp

if __name__ == "__main__":
    """
    應用程式的進入點，負責創建並運行主應用程式。

    input:
        無外部參數，依賴 main_app.py 中定義的 MainApp 類。
    output:
        創建 MainApp 實例並啟動 GUI 應用程式。
    return:
        無返回值，程式進入事件迴圈直到手動關閉。
    """
    app = MainApp()
    app.run()
"""
if __name__ == "__main__":
表示這個檔案是被 直接執行 時，才會跑這段程式。
如果這個檔案是被別的程式 import，就不會執行這段（避免重複啟動）。

app = MainApp()
建立你定義的 MainApp 物件。

app.run()
進入 MainApp.run()，通常裡面會是 self.root.mainloop()，啟動整個 GUI 事件迴圈。
"""