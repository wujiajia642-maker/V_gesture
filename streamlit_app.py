import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# 1. 頁面基本配置
st.set_page_config(page_title="學生吳佳佳專題：勝利手勢偵測", layout="wide")

# --- 側邊欄 (Sidebar) ---
st.sidebar.header("📁 學生吳佳佳基本資料")
school = st.sidebar.text_input("高中名稱", "臺南大學附屬高級中學")
name = st.sidebar.text_input("學生姓名", "吳佳佳")
interest = st.sidebar.text_area("興趣", "電腦視覺、人工智慧")

st.sidebar.divider()

st.sidebar.header("⚙️ 模型參數設定")
# 置信度滑桿：預設 0.25，範圍 0.0 到 1.0
conf_threshold = st.sidebar.slider(
    "信心程度 (Confidence Score)",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05
)

st.sidebar.info(f"當前過濾：僅顯示置信度 ≥ {conf_threshold} 的結果")

# --- 主畫面 (Main) ---
st.title("✌️ 勝利手勢 (Victory Gesture) 偵測系統")
st.markdown(f"**開發者：** {school} - {name}")

# 2. 載入模型
@st.cache_resource
def load_yolo_model():
    # 確保 best.pt 與此程式碼在同一目錄
    return YOLO("best.pt")

model = load_yolo_model()

# 3. 上傳檔案區塊
uploaded_file = st.file_uploader("上傳照片進行測試", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 讀取影像
    img_pil = Image.open(uploaded_file)
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # 4. 執行偵測 (傳入滑桿設定的 conf 值)
    # YOLO 會自動過濾掉低於 conf_threshold 的框
    results = model.predict(source=img_cv, conf=conf_threshold)

    # 5. 繪製並顯示結果
    res_plotted = results[0].plot()
    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("原始影像")
        st.image(img_pil, use_container_width=True)

    with col2:
        st.subheader("偵測結果")
        st.image(res_rgb, use_container_width=True)

    # 6. 數據統計
    num_boxes = len(results[0].boxes)
    if num_boxes > 0:
        st.success(f"✅ 偵測完成！在信心門檻 {conf_threshold} 下，共偵測到 {num_boxes} 個目標。")

        # 可選：列出每個框的詳細分數
        with st.expander("查看詳細偵測分數"):
            for i, box in enumerate(results[0].boxes):
                score = float(box.conf)
                st.write(f"目標 {i+1}: 信心程度 {score:.2f}")
    else:
        st.warning(f"❌ 在信心門檻 {conf_threshold} 下，未發現任何目標。請嘗試調低滑桿數值。")

else:
    st.info("👋 請在上方上傳照片，並於左側調整模型參數。")