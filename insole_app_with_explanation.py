import streamlit as st
import cv2
import numpy as np
from PIL import Image
from fpdf import FPDF
import base64
import tempfile
from uuid import uuid4
import socket

# === 足圧データ取得関数（TCP通信） ===
def get_pressure_matrix():
    HOST = "192.168.4.1"
    PORT = 9999
    REQUEST_BYTES = bytes([0x00, 0x01, 0x02])

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(3)
            s.connect((HOST, PORT))
            s.sendall(REQUEST_BYTES)
            data = b""
            while len(data) < 5404:
                packet = s.recv(1024)
                if not packet:
                    break
                data += packet
    except Exception as e:
        st.error(f"通信エラー: {e}")
        return None

    # 5400バイト → 3600個のセンサ値
    sensor_values = []
    for i in range(0, 5400, 3):
        if i+2 >= len(data):
            break
        v1, v2, v3 = data[i], data[i+1], data[i+2]
        val1 = (v1 << 4) | (v2 >> 4)
        val2 = ((v2 & 0x0F) << 8) | v3
        sensor_values.extend([val1, val2])

    matrix = np.array(sensor_values[:3600]).reshape((60, 60))
    return matrix

# === 足圧マップ画像保存 ===
def save_pressure_image(matrix):
    norm = cv2.normalize(matrix, None, 0, 255, cv2.NORM_MINMAX)
    norm_uint8 = norm.astype(np.uint8)
    heatmap = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_JET)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(tmp_file.name, heatmap)
    return tmp_file.name, heatmap

# === アーチ分類 ===
def classify_arch_by_image(image_cv):
    hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) | cv2.inRange(hsv, (160, 100, 100), (179, 255, 255))
    yellow_mask = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    _, binary = cv2.threshold(morph, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    total_bbox_area = sum([cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3] for c in contours])
    red_area = cv2.countNonZero(red_mask)
    yellow_area = cv2.countNonZero(yellow_mask)
    total_ratio = (red_area + yellow_area) / total_bbox_area * 100 if total_bbox_area else 0
    if total_ratio < 22:
        return "High"
    elif total_ratio > 28:
        return "Flat"
    else:
        return "Normal"

# === 説明文 ===
arch_explains = {
    "Flat": "土踏まずが低く衝撃吸収が弱いため、疲れやすさや足・膝・腰への負担が増します。",
    "High": "土踏まずが高く接地が少ないため、衝撃が集中しやすく痛みや不安定さの原因になります。",
    "Normal": "バランスの良い足型ですが、定期的なチェックと予防が重要です。",
    "外反母趾": "親指の付け根が突出し、痛みや変形が生じやすいため靴選びや補正が必要です。"
}

leg_explains = {
    "O脚": "膝が外に開いて関節に負担がかかりやすく、姿勢や歩行の見直しが必要です。",
    "X脚": "膝が内側に寄ることで足首や膝の内側に負担が集中しやすくなります。",
    "正常": "脚全体のバランスが取れており、安定した歩行が可能な理想的な状態です。"
}

pattern_table = {
    ("Flat", "O脚"): 1, ("Flat", "X脚"): 2, ("Flat", "正常"): 3,
    ("High", "O脚"): 4, ("High", "X脚"): 5, ("High", "正常"): 6,
    ("外反母趾", "O脚"): 7, ("外反母趾", "X脚"): 8, ("外反母趾", "正常"): 9,
    ("Normal", "O脚"): 10, ("Normal", "X脚"): 11, ("Normal", "正常"): 12,
}

# === PDF生成 ===
def create_pdf(image_path, arch_type, leg_shape, insole_number):
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("ArialUnicode", "", fname="fonts/NotoSansCJKjp-Regular.ttf", uni=True)
    pdf.set_font("ArialUnicode", size=12)

    pdf.cell(0, 10, "インソール提案レポート", ln=True)
    pdf.cell(0, 10, f"アーチタイプ: {arch_type}", ln=True)
    pdf.multi_cell(0, 10, f"説明: {arch_explains.get(arch_type, '')}")
    pdf.cell(0, 10, f"脚の形状: {leg_shape}", ln=True)
    pdf.multi_cell(0, 10, f"説明: {leg_explains.get(leg_shape, '')}")
    pdf.cell(0, 10, f"推奨インソール番号: {insole_number}", ln=True)

    if image_path:
        pdf.image(image_path, x=10, y=pdf.get_y(), w=100)

    pdf_path = f"insole_report_{uuid4().hex}.pdf"
    pdf.output(pdf_path)
    return pdf_path

# === Streamlit アプリ ===
def main():
    st.title("🦶 足圧測定＋インソール選定システム")

    hallux_valgus = st.checkbox("👣 外反母趾がある")
    leg_shape = st.radio("🦵 脚の形状", ["O脚", "X脚", "正常"])

    if st.button("📡 足圧測定スタート"):
        matrix = get_pressure_matrix()
        if matrix is not None:
            img_path, image_cv = save_pressure_image(matrix)
            arch_type = "外反母趾" if hallux_valgus else classify_arch_by_image(image_cv)
            pattern_key = (arch_type, leg_shape)
            insole_number = pattern_table.get(pattern_key, "該当なし")

            st.image(image_cv, caption="足圧ヒートマップ", channels="BGR", use_column_width=True)
            st.success(f"✅ アーチタイプ: {arch_type} ／ 脚の形状: {leg_shape}")
            st.info(f"推奨インソール番号：**{insole_number}**")

            st.markdown("### 📝 アーチ説明")
            st.info(arch_explains.get(arch_type, ""))
            st.markdown("### 📝 脚の形状説明")
            st.info(leg_explains.get(leg_shape, ""))

            if st.button("📄 PDF出力"):
                pdf_path = create_pdf(img_path, arch_type, leg_shape, insole_number)
                with open(pdf_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="insole_report.pdf">📥 PDFをダウンロード</a>'
                    st.markdown(href, unsafe_allow_html=True)

main()


