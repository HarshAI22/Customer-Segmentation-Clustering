# Customer Segmentation using Clustering

Customer segmentation based on income and spending patterns using KMeans and DBSCAN clustering algorithms.

## Tech Stack
- Python — core language
- Scikit-learn — KMeans, DBSCAN
- Pandas, NumPy — data processing
- Matplotlib, Seaborn — visualizations
- Streamlit — interactive web app
- Joblib — model serialization

## Features
- KMeans clustering with Elbow method for optimal K
- DBSCAN for outlier detection
- Silhouette score evaluation
- Interactive Streamlit app to predict customer segment
- Cluster visualization with user position marked

## Project Structure

Customer-Segmentation-Clustering/
├── customer_segmentation.ipynb
├── app.py
├── kmeans_model.pkl
├── scaler.pkl
├── requirements.txt
└── .env


## Setup
bash
pip install -r requirements.txt
streamlit run app.py
