# 🔍 Data Analytics and Mining: Clustering with K-Means & DBSCAN

A modular, well-documented Python implementation of **K-Means** and **DBSCAN** clustering algorithms applied to real-world datasets from the `scikit-learn` library. This project was completed for the *SC4020 - Data Analytics and Mining* course at NTU Singapore and investigates the strengths, limitations, and tuning requirements of two popular unsupervised learning techniques.

---

## 📂 Datasets Used

- **Diabetes** – High-dimensional numeric dataset (10 features)  
- **Breast Cancer** – Medical imaging-derived data (30 features, 2 classes)  
- **Iris** – Classic 4-feature flower classification (3 classes)  
- **Wine** – Chemical analysis of wines (13 features, 3 classes)

Each dataset is preprocessed using **StandardScaler** and optionally **PCA** to enable 2D visualization and dimensionality reduction for better clustering performance.

---

## 📌 What This Project Demonstrates

- Hands-on implementation and visualization of:
  - 📈 **K-Means**, including K-Means++
  - 🔍 **DBSCAN** with hyperparameter tuning (`eps`, `min_samples`)
  - 🧬 **Agglomerative Clustering** (for Wine dataset)

- Evaluation with:
  - ✅ **Silhouette Score**
  - ✅ **Adjusted Rand Index (ARI)**
  - ✅ **Adjusted Mutual Information (AMI)**

- Application of:
  - 📉 Elbow Method to determine optimal number of clusters
  - 🧭 K-NN for DBSCAN `eps` selection
  - 📊 PCA for 2D projection and clearer insights

---

## 🧪 How to Run

Each dataset has a separate notebook or script:

```bash
# Clone this repository
git clone https://github.com/your-username/data-mining-clustering.git
cd data-mining-clustering

# Run notebooks individually
jupyter notebook breast_cancer_src.ipynb
jupyter notebook diabetes_src.ipynb
jupyter notebook iris_src.ipynb

# Or run the wine analysis Python script
python wine_src.py
```
---

## 🧠 Key Insights

- **K-Means** performs better on spherical, equally-sized clusters  
- **DBSCAN** excels with irregular shapes and noise handling, but struggles in high dimensions  
- **Agglomerative Clustering** can serve as a powerful alternative when `k` is known  
- PCA helps mitigate high-dimensionality challenges, especially in medical and chemical datasets  

---

## 🛠️ Technologies Used

- Python  
- NumPy, Pandas  
- scikit-learn  
- Matplotlib, Seaborn  
- Jupyter Notebooks  

---

## 👨‍💻 Authors

- Poh Zi Jie Isaac  
- Dexter Voon Kai Xian  
- Lim Jun Yu  
- **Tathagato Mukherjee**

---

## 📚 References

- [scikit-learn documentation](https://scikit-learn.org/)
- [Clustering & PCA in Python – Medium](https://medium.com/@jackiee.jecksom/clustering-and-principal-component-analysis-pca-from-sklearn-c8ea5fed6648)
- [DBSCAN with scikit-learn – StackAbuse](https://stackabuse.com/dbscan-with-scikit-learn-in-python/)
- [K-Means in Python – Real Python](https://realpython.com/k-means-clustering-python/)

