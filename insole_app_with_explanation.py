import streamlit as st
import cv2
import numpy as np
from PIL import Image
...

# アーチ分類ルール（色面積比ベース）
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

# インソール提案マッピング
pattern_table = {
    ("Flat", "O脚"): 1,
    ("Flat", "X脚"): 2,
    ("Flat", "正常"): 3,
    ("High", "O脚"): 4,
    ("High", "X脚"): 5,
    ("High", "正常"): 6,
    ("外反母趾", "O脚"): 7,
    ("外反母趾", "X脚"): 8,
    ("外反母趾", "正常"): 9,
    ("Normal", "O脚"): 10,
    ("Normal", "X脚"): 11,
    ("Normal", "正常"): 12,
}

# 説明文（簡略化）
arch_explains = {
    "Flat": "土踏まずが低く衝撃を吸収しにくいため、疲れやすさや痛みが出やすくなります。",
    "High": "土踏まずが高く接地面が少ないため、足裏に圧力が集中しやすくなります。",
    "Normal": "土踏まずがバランスよく形成された理想的な足型です。",
    "外反母趾": "親指の付け根が突出しており、痛みや変形のリスクがあります。"
}

leg_explains = {
    "O脚": "膝が外側に開き、膝や足首に負担がかかりやすい状態です。",
    "X脚": "膝が内側に寄り、関節に歪みが生じやすい状態です。",
    "正常": "脚のバランスが良く、関節への負担も少ない理想的な状態です。"
}

# Streamlitアプリ本体
def main():
    st.title("🦶 インソール提案アプリ")

    st.markdown("足圧画像と目視チェックに基づいて、適切なインソール番号を自動提案します。")

    # 外反母趾チェック
    hallux_valgus = st.checkbox("👣 外反母趾がある")

    # 脚の形状選択
    leg_shape = st.radio("🦵 脚の形状", ["O脚", "X脚", "正常"])

    # 足圧画像アップロード
    uploaded_file = st.file_uploader("🖼 足圧画像をアップロード", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        st.image(image, caption="アップロード画像", use_column_width=True)

        # アーチ分類
        if hallux_valgus:
            arch_type = "外反母趾"
        else:
            arch_type = classify_arch_by_image(image_cv)

        pattern_key = (arch_type, leg_shape)
        if pattern_key in pattern_table:
            insole_number = pattern_table[pattern_key]
            st.success(f"🟢 推奨インソール番号：**{insole_number}**")
        else:
            st.warning("該当するインソール番号が見つかりませんでした。")

        # 説明表示
        st.subheader("📝 足の形状の説明")
        st.info(arch_explains.get(arch_type, "情報なし"))

        st.subheader("📝 脚の形状の説明")
        st.info(leg_explains.get(leg_shape, "情報なし"))

if __name__ == "__main__":
    main()
