import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="XGBoost Prediction App",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义CSS样式 ====================
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #3498DB;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2C3E50;
    }
    h1 {
        color: #2C3E50;
        text-align: center;
        padding-bottom: 1rem;
        border-bottom: 3px solid #3498DB;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #ECF0F1;
        border-left: 5px solid #3498DB;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #FFF3CD;
        border-left: 5px solid #FFC107;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== 设置matplotlib白底样式 ====================
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# ==================== 标题 ====================
st.title("🔬 XGBoost Machine Learning Prediction App")
st.markdown("---")

# ==================== 侧边栏说明 ====================
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    This application uses an XGBoost model trained on clinical data 
    with LASSO feature selection to make predictions.

    **Features:**
    - Real-time prediction
    - Probability estimation
    - SHAP explanation visualization
    """)

    st.header("📋 Instructions")
    st.markdown("""
    1. Enter patient information
    2. Click **Predict** button
    3. View prediction result and probability
    4. Check SHAP explanation
    """)

# ==================== LASSO筛选后的特征列表 ====================
SELECTED_FEATURES = [
    "Classification", "Location", "Number", "FOB", "CHOL",
    "CA199", "CEA", "HB", "Villous", "Erosion",
    "Smooth", "Color", "BMI", "Size", "Age"
]

# ==================== 标签映射字典 ====================
# 定义显示标签到数值的映射
LABEL_MAPPINGS = {
    'HB': {'Normal': 1, 'Low ': 2},
    'CEA': {'Normal': 1, 'High': 2},
    'CA199': {'Normal': 1, 'High': 2},
    'CHOL': {'Normal': 1, 'High': 2},
    'FOB': {'Negative': 1, 'Positive': 2},
    'Number': {'Single': 1, 'Multiple': 2},
    'Classification': {'Sessile': 1, 'Semi-pedunculated': 2, 'Pedunculated': 3},
    'Location': {
        'Cecum': 1,
        'Ascending': 2,
        'Transverse': 3,
        'Descending': 4,
        'Sigmoid': 5,
        'Rectum': 6
    },
    'Villous': {'No': 1, 'Yes': 2},
    'Erosion': {'No': 1, 'Yes': 2},
    'Smooth': {'No': 1, 'Yes': 2},
    'Color': {'Normal': 1, 'Red': 2, 'White': 3}
}

# ==================== 主界面 - 用户输入 ====================
st.header("📝 Patient Information Input")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographic Information")
    Age = st.number_input("Age (years)", min_value=1, max_value=120, value=50, step=1)
    BMI = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0, step=0.1)
    Size = st.number_input("Size (cm)", min_value=0.0, max_value=20.0, value=2.0, step=0.1)

with col2:
    st.subheader("Laboratory Tests")
    HB_label = st.selectbox("HB", options=list(LABEL_MAPPINGS['HB'].keys()), index=0)
    CEA_label = st.selectbox("CEA", options=list(LABEL_MAPPINGS['CEA'].keys()), index=0)
    CA199_label = st.selectbox("CA199", options=list(LABEL_MAPPINGS['CA199'].keys()), index=0)
    CHOL_label = st.selectbox("CHOL", options=list(LABEL_MAPPINGS['CHOL'].keys()), index=0)
    FOB_label = st.selectbox("FOB (Fecal Occult Blood)", options=list(LABEL_MAPPINGS['FOB'].keys()), index=0)

with col3:
    st.subheader("Clinical Characteristics")
    Number_label = st.selectbox("Number", options=list(LABEL_MAPPINGS['Number'].keys()), index=0)
    Classification_label = st.selectbox("Classification", options=list(LABEL_MAPPINGS['Classification'].keys()),
                                        index=0)
    Location_label = st.selectbox("Location", options=list(LABEL_MAPPINGS['Location'].keys()), index=0)
    Villous_label = st.selectbox("Villous Component", options=list(LABEL_MAPPINGS['Villous'].keys()), index=0)
    Erosion_label = st.selectbox("Erosion", options=list(LABEL_MAPPINGS['Erosion'].keys()), index=0)
    Smooth_label = st.selectbox("Smooth Surface", options=list(LABEL_MAPPINGS['Smooth'].keys()), index=0)
    Color_label = st.selectbox("Color", options=list(LABEL_MAPPINGS['Color'].keys()), index=0)

st.markdown("---")

# ==================== 预测按钮 ====================
if st.button("🔮 Predict"):
    try:
        with st.spinner("Loading model and making prediction..."):
            # ==================== 加载模型 ====================
            model_path = os.path.join(os.getcwd(), "XGB_model.pkl")

            if not os.path.exists(model_path):
                st.error("❌ Model file 'XGB_model.pkl' not found!")
                st.stop()

            model = joblib.load(model_path)
            st.success("✅ XGBoost Model Loaded Successfully")

            # ==================== 将标签转换为数值 ====================
            HB = LABEL_MAPPINGS['HB'][HB_label]
            CEA = LABEL_MAPPINGS['CEA'][CEA_label]
            CA199 = LABEL_MAPPINGS['CA199'][CA199_label]
            CHOL = LABEL_MAPPINGS['CHOL'][CHOL_label]
            FOB = LABEL_MAPPINGS['FOB'][FOB_label]
            Number = LABEL_MAPPINGS['Number'][Number_label]
            Classification = LABEL_MAPPINGS['Classification'][Classification_label]
            Location = LABEL_MAPPINGS['Location'][Location_label]
            Villous = LABEL_MAPPINGS['Villous'][Villous_label]
            Erosion = LABEL_MAPPINGS['Erosion'][Erosion_label]
            Smooth = LABEL_MAPPINGS['Smooth'][Smooth_label]
            Color = LABEL_MAPPINGS['Color'][Color_label]

            # ==================== 构建输入数据 ====================
            input_data = {
                "Classification": int(Classification),
                "Location": int(Location),
                "Number": int(Number),
                "FOB": int(FOB),
                "CHOL": int(CHOL),
                "CA199": int(CA199),
                "CEA": int(CEA),
                "HB": int(HB),
                "Villous": int(Villous),
                "Erosion": int(Erosion),
                "Smooth": int(Smooth),
                "Color": int(Color),
                "BMI": float(BMI),
                "Size": float(Size),
                "Age": float(Age)
            }

            # 创建DataFrame
            X = pd.DataFrame([input_data], columns=SELECTED_FEATURES)

            # 显示输入数据（用于调试）
            with st.expander("📋 View Input Data"):
                # 创建带标签的显示版本
                display_data = {
                    "Feature": [],
                    "Value (Label)": [],
                    "Value (Numeric)": []
                }

                display_data["Feature"].extend(["Age", "BMI", "Size"])
                display_data["Value (Label)"].extend([f"{Age} years", f"{BMI:.1f}", f"{Size:.1f} cm"])
                display_data["Value (Numeric)"].extend([Age, BMI, Size])

                for feature in ["HB", "CEA", "CA199", "CHOL", "FOB", "Number",
                                "Classification", "Location", "Villous", "Erosion", "Smooth", "Color"]:
                    label_var = f"{feature}_label"
                    display_data["Feature"].append(feature)
                    display_data["Value (Label)"].append(locals()[label_var])
                    display_data["Value (Numeric)"].append(input_data[feature])

                st.dataframe(pd.DataFrame(display_data), use_container_width=True)

            # ==================== 预测 ====================
            pred = model.predict(X)[0]
            prob = model.predict_proba(X)[0]

            # ==================== 显示结果 ====================
            st.markdown("---")
            st.header("📊 Prediction Results")

            # 结果展示
            col_res1, col_res2 = st.columns(2)

            with col_res1:
                st.markdown(f"""
                <div class="prediction-box">
                    <h3 style="color: #2C3E50;">Prediction Result</h3>
                    <h1 style="color: {'#E74C3C' if pred == 1 else '#27AE60'}; text-align: center;">
                        {'Positive (1)' if pred == 1 else 'Negative (0)'}
                    </h1>
                </div>
                """, unsafe_allow_html=True)

            with col_res2:
                st.markdown(f"""
                <div class="prediction-box">
                    <h3 style="color: #2C3E50;">Prediction Probability</h3>
                    <h1 style="color: #3498DB; text-align: center;">
                        {prob[1]:.2%}
                    </h1>
                    <p style="text-align: center; color: #7F8C8D;">
                        Probability of Positive Class
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # 概率条形图
            st.subheader("Probability Distribution")
            prob_df = pd.DataFrame({
                'Class': ['Negative (0)', 'Positive (1)'],
                'Probability': [prob[0], prob[1]]
            })

            fig_prob, ax = plt.subplots(figsize=(10, 4), facecolor='white')
            colors = ['#27AE60', '#E74C3C']
            bars = ax.barh(prob_df['Class'], prob_df['Probability'],
                           color=colors, alpha=0.8, edgecolor='#2C3E50', linewidth=2)

            ax.set_xlabel('Probability', fontsize=13, fontweight='bold', color='#34495E')
            ax.set_title('Class Probability Distribution',
                         fontsize=15, fontweight='bold', pad=15, color='#2C3E50')
            ax.set_xlim([0, 1])
            ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
            ax.set_axisbelow(True)

            # 添加数值标签
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height() / 2,
                        f'{width:.2%}', ha='left', va='center',
                        fontsize=12, fontweight='bold', color='#2C3E50')

            plt.tight_layout()
            st.pyplot(fig_prob)
            plt.close(fig_prob)

            # ==================== SHAP解释 ====================
            st.markdown("---")
            st.header("🔍 SHAP Explanation")

            with st.spinner("Computing SHAP values..."):
                try:
                    # 使用TreeExplainer
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X)

                    # 如果返回的是列表（多分类），取正类
                    if isinstance(shap_values, list):
                        shap_values_positive = shap_values[1]
                    else:
                        shap_values_positive = shap_values

                    # 构造Explanation对象
                    shap_exp = shap.Explanation(
                        values=shap_values_positive[0],
                        base_values=explainer.expected_value if not isinstance(explainer.expected_value, list) else
                        explainer.expected_value[1],
                        data=X.iloc[0],
                        feature_names=X.columns.tolist()
                    )

                    # ==================== SHAP Waterfall图 ====================
                    st.subheader("SHAP Waterfall Plot")
                    st.info(
                        "This chart shows how each feature contributes to the prediction for this specific patient.")

                    fig_waterfall, ax = plt.subplots(figsize=(12, 8), facecolor='white')
                    shap.plots.waterfall(shap_exp, max_display=15, show=False)
                    plt.tight_layout()
                    st.pyplot(fig_waterfall)
                    plt.close(fig_waterfall)

                    # ==================== 特征重要性表格 ====================
                    st.subheader("Feature Contribution Summary")

                    feature_importance = pd.DataFrame({
                        'Feature': X.columns,
                        'Value': X.iloc[0].values,
                        'SHAP Value': shap_values_positive[0],
                        'Abs SHAP': np.abs(shap_values_positive[0])
                    }).sort_values('Abs SHAP', ascending=False)

                    # 格式化显示
                    st.dataframe(
                        feature_importance.style.background_gradient(
                            subset=['SHAP Value'], cmap='RdYlGn', vmin=-0.5, vmax=0.5
                        ).format({
                            'Value': '{:.2f}',
                            'SHAP Value': '{:.4f}',
                            'Abs SHAP': '{:.4f}'
                        }),
                        use_container_width=True,
                        height=400
                    )

                    # ==================== 下载SHAP图表 ====================
                    st.subheader("📥 Download SHAP Plot")

                    # 重新生成图表用于下载
                    fig_download, ax = plt.subplots(figsize=(12, 8), facecolor='white')
                    shap.plots.waterfall(shap_exp, max_display=15, show=False)
                    plt.title(f'SHAP Waterfall Plot - Prediction: {pred}',
                              fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
                    plt.tight_layout()

                    # 保存到临时文件
                    temp_path = "temp_shap_waterfall.png"
                    plt.savefig(temp_path, dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close(fig_download)

                    # 提供下载按钮
                    with open(temp_path, "rb") as file:
                        st.download_button(
                            label="⬇️ Download SHAP Waterfall Plot (PNG)",
                            data=file,
                            file_name=f"SHAP_Waterfall_Pred_{pred}.png",
                            mime="image/png"
                        )

                    # 删除临时文件
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                except Exception as e:
                    st.error(f"❌ SHAP calculation failed: {str(e)}")
                    st.warning("SHAP explanation is not available, but the prediction is still valid.")

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        st.exception(e)

# ==================== 页脚 ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7F8C8D; padding: 1rem;">
    <p>© 2026 XGBoost Prediction System | Powered by Streamlit & SHAP</p>
    <p style="font-size: 12px;">⚠️ This tool is for research purposes only. Clinical decisions should be made by qualified medical professionals.</p>
</div>
""", unsafe_allow_html=True)