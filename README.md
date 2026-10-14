# 🏨 Hotel Booking Reservation Status Classification

A machine-learning project for **CSE422: Artificial Intelligence** at BRAC University. It predicts whether a hotel booking will end as **Check-Out**, **Canceled**, or **No-Show**, using booking and guest characteristics, and compares four supervised models plus an unsupervised K-Means analysis.

**Team:** Mehrun-Nisa Rahman, Nowshin Tabassum (Section 06)

## 🎯 Problem

Reservation outcomes affect room availability, revenue planning, staffing, and overbooking policy. This project frames the task as a **three-class classification problem** and pays special attention to the severe class imbalance (No-Show is only about 1% of the data) and to **target leakage**.

## 📊 Dataset

- **Hotel Booking Demand** dataset: 119,390 bookings, 32 columns (20 numerical, 12 categorical)
- Target: `reservation_status` (Check-Out 62.96%, Canceled 36.03%, No-Show 1.01%)
- Included here as `hotel_bookings.csv`

## 🔬 Workflow

1. **Exploratory analysis:** class distribution, status by hotel type, arrival month, market segment, deposit type, customer type, lead time, ADR, and special requests, plus a correlation analysis
2. **Preprocessing:**
   - Dropped `company` (~94% missing)
   - Filled missing `agent` values with an "Unknown" category
   - Removed bookings with zero guests
   - Removed leakage columns (`is_canceled`, `reservation_status_date`, `assigned_room_type`)
   - Engineered features such as total nights and total guests
3. **Splitting:** stratified 80/20 train/test split
4. **Encoding and scaling:** median imputation and standard scaling for numeric features, one-hot encoding for categorical ones, fit on training data only
5. **Feature selection:** `SelectKBest` (top 30 features)
6. **Models:** K-Nearest Neighbors, Logistic Regression, Gaussian Naive Bayes, and a multilayer neural network (MLP)
7. **Unsupervised analysis:** K-Means with the elbow method, visualised with PCA and compared against the real labels
8. **Evaluation:** accuracy, balanced accuracy, macro precision/recall/F1, confusion matrices, and multiclass ROC-AUC

## 📈 Results

| Model | Accuracy | Balanced Acc. | Macro F1 | Weighted AUC |
| --- | --- | --- | --- | --- |
| K-Nearest Neighbors | 0.8196 | **0.5760** | **0.6032** | 0.8842 |
| Logistic Regression | 0.7885 | 0.5007 | 0.5069 | 0.8666 |
| Gaussian Naive Bayes | 0.2718 | 0.5184 | 0.2935 | 0.7447 |
| Neural Network | **0.8308** | 0.5637 | 0.5896 | **0.9098** |

**Takeaways**

- The neural network has the best accuracy and AUC, while KNN is the most balanced across classes (best Macro F1 and balanced accuracy).
- The rare **No-Show** class remains hard to predict reliably for every model.
- Naive Bayes performs poorly because its independence assumptions don't fit the encoded booking features.

See the full analysis, figures, and discussion in the [project report](CSE422_Project_Report.pdf).

## 📁 Repository contents

| File | Description |
| --- | --- |
| `CSE422_Project.ipynb` | Full notebook: EDA, preprocessing, modelling, and evaluation |
| `hotel_bookings.csv` | Dataset used in the notebook |
| `CSE422_Project_Report.pdf` | Complete project report |

## 🚀 How to run

**Option 1: Google Colab (the notebook was written for this)**

1. Open the notebook in [Google Colab](https://colab.research.google.com/).
2. Run the cells in order. When the upload cell appears, upload `hotel_bookings.csv`.

**Option 2: Locally**

1. Install the dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
2. Open `CSE422_Project.ipynb` in Jupyter.
3. In the data-loading cell, replace the Colab upload code (`files.upload()`) with:
   ```python
   df = pd.read_csv("hotel_bookings.csv")
   ```
4. Run all cells.

## 🧰 Built with

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn

## 📚 Course

CSE422: Artificial Intelligence, BRAC University, Section 06
Faculty: Farhan Faruk, Nirjhor Datta
