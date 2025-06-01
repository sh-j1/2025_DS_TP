import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def train_kmeans(data_path: str, output_dir: str, n_clusters: int = 3):
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['stress_level'])  # 필수 결측 제거

    # 클러스터링에서 target 제외
    X = df.drop(columns=['stress_level'])

    # KMeans 학습
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    # 결과 저장
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels

    os.makedirs(output_dir, exist_ok=True)
    df_clustered.to_csv(os.path.join(output_dir, 'kmeans_cluster_labels.csv'), index=False)

    # 2D 시각화를 위한 PCA 결과 저장도 가능 (선택)
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
    pca_df['cluster'] = cluster_labels
    pca_df.to_csv(os.path.join(output_dir, 'kmeans_pca_2d.csv'), index=False)

    print("[OK] KMeans clustering completed and results saved.")

