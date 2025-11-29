# House_prices_prediction_ML
A Complete end-to-end ML Pipeline to predict house prices

Overview-This project predicts house sale prices using the Ames Housing dataset — a modern alternative to the Boston Housing dataset, widely used in machine learning regression problems.

built a complete ML pipeline that includes:

- 🧼 Data cleaning  
- 🏗 Feature preprocessing  
- 🔢 Ordinal & one-hot encoding  
- 🧮 Scaling  
- 🤖 Model training with Linear Regression, Random Forest, and Gradient Boosting  
- 📊 Cross-validation  
- 📝 Final model export (`gb_model.pkl`)  
- 📤 Submission generator (`submission.csv`)

## 🧼 Data Preprocessing

### **1. Missing Value Handling**
- Numerical columns → median imputation  
- Categorical columns → `"None"` or grouped as `"other"`  
- Removed extremely sparse columns (`Alley`, `PoolQC`, `Fence`, `MiscFeature`, `MiscVal`)

### **2. Ordinal Encoding of Quality Features**
Features like:
- `ExterQual`
- `ExterCond`
- `KitchenQual`
- `BsmtQual`
- `HeatingQC`
- `FireplaceQu`
- `GarageQual`
- `GarageCond`

Were mapped using:None = 0,Po = 1,Fa = 2,TA = 3,Gd = 4,Ex = 5

**3. Rare Category Grouping**
Any categorical category with frequency < 1% was replaced with `"other"`.

### **4. One-Hot Encoding & Scaling**
- Numeric → StandardScaler  
- Categorical → OneHotEncoder  

All preprocessing was wrapped in a **ColumnTransformer** inside a full ML pipeline.

---

## 🤖 Models Trained

| Model | CV RMSE |
|-------|---------|
| Linear Regression | ~35,053 |
| Random Forest | ~30,406 |
| **Gradient Boosting (Best)** | **~28,299** |

Gradient Boosting was selected as the final model because of:

- strong performance  
- robustness to outliers  
- ability to handle complex feature interactions  

---

## 🎯 Final Output

### ✔ Saved Model-models/gb_model.pkl
🔮 Future Improvements

- Add XGBoost / LightGBM models  
- Perform hyperparameter tuning  
- Add feature engineering (log transforms, interactions)  
- Deploy as a web app (Flask/Streamlit)

