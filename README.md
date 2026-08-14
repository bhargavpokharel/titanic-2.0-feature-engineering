# titanic-2.0-feature-engineering
Improved Titanic survival prediction using advanced feature engineering (Title, FamilySize, IsAlone, AgeBin, FareBin).
# 🚢 Titanic 2.0: Feature Engineering Challenge

## 📖 Overview
This project is an advanced follow-up to the classic Titanic survival prediction task.  
The original model used only 7 raw features. Here, I applied **feature engineering** to extract deeper patterns from the data, resulting in a measurable accuracy improvement.

## 🎯 Goal
To improve model accuracy by creating new, informative features from the original dataset — demonstrating that **good features beat complex algorithms**.

## 🛠️ Features Engineered
| Feature | Description |
|---------|-------------|
| **Title** | Extracted from passenger names (Mr, Mrs, Miss, Master, Rare) — captures social status. |
| **FamilySize** | `SibSp` + `Parch` + 1 — indicates whether the passenger traveled with family. |
| **IsAlone** | Binary: `1` if `FamilySize == 1`, else `0`. |
| **AgeBin** | Age grouped into: Child, Teen, Adult, Senior, Elder. |
| **FareBin** | Fare grouped into quartiles: Low, Medium, High, Very High. |

## 📈 Results
| Model | Features | Accuracy |
|-------|----------|----------|
| Random Forest (Original) | 7 raw features | **81.56%** |
| Random Forest (Engineered) | 12 features | **82.68%** |

**Improvement:** +1.12% — a significant gain given the small dataset size.

## 🧠 Top 5 Feature Importances
1. **Fare** (`0.2138`)  
2. **Age** (`0.1843`)  
3. **Title** (`0.1702`) — *engineered feature!*  
4. **Sex** (`0.1496`)  
5. **Pclass** (`0.0659`)

---

## 🛠️ Tools & Libraries Used
- **Python 3.13**  
- **Pandas** — Data loading & manipulation  
- **NumPy** — Numerical operations  
- **Scikit-learn** — `train_test_split`, `StandardScaler`, `RandomForestClassifier`, evaluation metrics  
- **Jupyter Notebook** — Interactive development environment

---

## 📂 Repository Contents
| File | Description |
|------|-------------|
| `Titanic_Feature_Engineering.ipynb` | Complete notebook with cleaning, feature engineering, encoding, training, and evaluation. |
| `train.csv` / `test.csv` | Original Titanic datasets. |
| `titanic_model_engineered.pkl` | Saved trained Random Forest model. |
| `titanic_scaler_engineered.pkl` | Saved fitted StandardScaler. |
| `full_train_data_pleasinglook.html` | Aesthetic HTML view of raw data. |
| `README.md` | This file. |

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/bhargavpokharel/titanic-2.0-feature-engineering.git
   