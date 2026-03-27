import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=1600");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .overlay {
        background: rgba(0, 0, 0, 0.92);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #555;
        backdrop-filter: blur(8px);
    }
    h1 { color: #f5a623 !important; text-shadow: 2px 2px 8px #000; }
    .subtitle { color: #ffffff; font-size: 1rem; margin-bottom: 1.5rem; }
    .result-box {
        background: linear-gradient(135deg, #1a1a2e, #f5a623);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: white;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

cluster_names = {
    0: "💸 High Income, Low Spender",
    1: "🎯 High Income, High Spender",
    2: "😐 Average Income, Average Spender",
    3: "⚠️ Low Income, High Spender",
    4: "💰 Low Income, Low Spender"
}

left, main_col, right = st.columns([0.2, 2, 3])

with main_col:
    st.markdown("<div class='overlay'>", unsafe_allow_html=True)
    st.markdown("<h1>🛍️ Customer Segmentation</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Enter customer details to find their segment</p>", unsafe_allow_html=True)

    income   = st.slider("💰 Annual Income (k$)", 10, 140, 60)
    spending = st.slider("🛒 Spending Score (1-100)", 1, 100, 50)

    if st.button("🔍 Find Segment", use_container_width=True):
        input_scaled = scaler.transform([[income, spending]])
        cluster = kmeans.predict(input_scaled)[0]
        label = cluster_names.get(cluster, f"Cluster {cluster}")

        st.markdown(f"""
            <div class='result-box'>
                Segment: {label}
            </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📊 Cluster Visualization")

        np.random.seed(42)
        n = 200
        inc = np.concatenate([np.random.normal(20,5,40), np.random.normal(60,10,60),
                               np.random.normal(90,8,40), np.random.normal(25,6,30),
                               np.random.normal(85,7,30)])
        spe = np.concatenate([np.random.normal(20,5,40), np.random.normal(50,10,60),
                               np.random.normal(80,8,40), np.random.normal(75,7,30),
                               np.random.normal(20,5,30)])
        df_plot = pd.DataFrame({
            'Income': np.clip(inc, 10, 140).astype(int),
            'Spending': np.clip(spe, 1, 100).astype(int)
        })
        all_scaled = scaler.transform(df_plot[['Income', 'Spending']])
        df_plot['Cluster'] = kmeans.predict(all_scaled).astype(str)

        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_alpha(0)
        ax.set_facecolor('#1a1a1a')
        sns.scatterplot(data=df_plot, x='Income', y='Spending',
                        hue='Cluster', palette='tab10', s=60, ax=ax)
        ax.scatter(income, spending, color='yellow', s=250,
                   marker='*', label='You ⭐', zorder=5)
        ax.set_title("Your Position in Clusters", color='white')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.legend(facecolor='#1a1a1a', labelcolor='white')
        st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)