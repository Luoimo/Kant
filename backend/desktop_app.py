import sys
import threading
import time
import webview
import uvicorn
import os
from pathlib import Path
from fastapi.staticfiles import StaticFiles

# 确保工作目录可写（将数据存放在用户主目录下的隐藏文件夹中）
app_data_dir = Path.home() / ".KantData"
app_data_dir.mkdir(parents=True, exist_ok=True)
os.chdir(app_data_dir)

# tiktoken 缓存目录（避免每次启动都去网络下载编码文件）
tiktoken_cache_dir = app_data_dir / "tiktoken_cache"
tiktoken_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["TIKTOKEN_CACHE_DIR"] = str(tiktoken_cache_dir)

from main import app

# 确定前端打包文件的路径
# 如果是通过 PyInstaller 打包后的运行，路径在 sys._MEIPASS 中
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    base_path = Path(sys._MEIPASS) / "frontend_dist"
else:
    base_path = Path(__file__).parent.parent / "frontend" / "dist"

# 挂载前端静态文件
if base_path.exists():
    app.mount("/", StaticFiles(directory=str(base_path), html=True), name="frontend")
else:
    print(f"Warning: Frontend build directory not found at {base_path}")

def run_server():
    # 运行 FastAPI 后端
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")

if __name__ == '__main__':
    # 在后台线程启动服务器
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # 给服务器一点时间启动
    time.sleep(2)

    # 创建并启动桌面窗口
    webview.create_window("Kant - AI Reading Agent", "http://127.0.0.1:8000", width=1200, height=800)
    webview.start()
