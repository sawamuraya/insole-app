import streamlit as st
import cv2
import numpy as np
from PIL import Image
from fpdf import FPDF
import base64
import tempfile
from uuid import uuid4

# アーチ分類関数
def classify_arch_by_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) | cv2.inRange(hsv, (160, 100, 100), (179, 255, 255))
    yellow_mask = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

# インソールパターンテーブル
pattern_table = {
    ("Flat", "O脚"): 1, ("Flat", "X脚"): 2, ("Flat", "正常"): 3,
    ("High", "O脚"): 4, ("High", "X脚"): 5, ("High", "正常"): 6,
    ("外反母趾", "O脚"): 7, ("外反母趾", "X脚"): 8, ("外反母趾", "正常"): 9,
    ("Normal", "O脚"): 10, ("Normal", "X脚"): 11, ("Normal", "正常"): 12,
}

# 説明文（簡略版）
arch_explains = {
    "Flat": "土踏まずが低く衝撃吸収が弱いため、疲れやすさや足・膝・腰への負担が増します。サポート力のあるインソールでの補正が効果的です。",
    "High": "土踏まずが高く足裏の接地が少ないため、衝撃が集中しやすく痛みやバランス不良の原因になります。クッション性が重要です。",
    "Normal": "バランスの取れた足型で衝撃吸収に優れていますが、加齢や姿勢の乱れで崩れることも。定期的なチェックと予防が大切です。",
    "外反母趾": "親指が外側に曲がる症状で、痛みや変形を伴いやすく靴選びが重要です。初期対応やインソールでの補正が進行防止につながります。"
}

leg_explains = {
    "O脚": "膝が外側に開いて湾曲した状態で、膝・股関節への負担が大きくなります。歩き方や筋力バランスの見直しが予防につながります。",
    "X脚": "膝が内側に寄り足首が離れた状態で、関節に負担がかかりやすくなります。姿勢改善や筋力強化での予防が効果的です。",
    "正常": "脚全体のバランスが取れており、関節への負担が少ない理想的な状態です。姿勢と歩き方を維持し、継続的なケアが大切です。"
}

# PDF生成
def create_pdf(image_path, arch_type, leg_shape, insole_number):
    pdf = FPDF()
    pdf.add_page()
    # フォント登録（初回のみ必要）
    font_path = "NotoSansJP-VariableFont_wght.ttf"  # リポジトリ内のパス
    pdf.add_font("Noto", "", font_path, uni=True)
    pdf.set_font("Noto", size=12)

    pdf.cell(0, 10, "インソール提案レポート", ln=True)
    pdf.cell(0, 10, f"アーチタイプ: {arch_type}", ln=True)
    pdf.multi_cell(0, 10, f"説明: {arch_explains.get(arch_type, '')}")
    pdf.cell(0, 10, f"脚の形状: {leg_shape}", ln=True)
    pdf.multi_cell(0, 10, f"説明: {leg_explains.get(leg_shape, '')}")
    pdf.cell(0, 10, f"推奨インソール番号: {insole_number}", ln=True)

    if image_path:
        pdf.image(image_path, x=10, y=pdf.get_y(), w=100)

    output_path = f"insole_report_{uuid4().hex}.pdf"
    pdf.output(output_path)
    return output_path

# Streamlitアプリ本体
def main():
    st.title("🦶 インソール提案＆PDF出力アプリ")

    hallux_valgus = st.checkbox("👣 外反母趾がある")
    leg_shape = st.radio("🦵 脚の形状", ["O脚", "X脚", "正常"])

    uploaded_file = st.file_uploader("🖼 足圧画像をアップロード", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        # 一時保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            image.save(tmp_file.name)
            image_path = tmp_file.name

        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        st.image(image, caption="アップロード画像", use_column_width=True)

        arch_type = "外反母趾" if hallux_valgus else classify_arch_by_image(image_cv)
        pattern_key = (arch_type, leg_shape)
        insole_number = pattern_table.get(pattern_key, "該当なし")

        st.success(f"🟢 推奨インソール番号：**{insole_number}**")
        st.markdown("### 📝 アーチ説明")
        st.info(arch_explains.get(arch_type, ""))
        st.markdown("### 📝 脚の形状説明")
        st.info(leg_explains.get(leg_shape, ""))

        if st.button("📄 PDF生成"):
            pdf_path = create_pdf(image_path, arch_type, leg_shape, insole_number)
            with open(pdf_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="insole_report.pdf">📥 PDFをダウンロード</a>'
                st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

