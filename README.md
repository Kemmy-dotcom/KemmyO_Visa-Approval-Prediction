# 📊 Visa Approval Prediction  

### **Objective**  
Analyze visa applicant data to build a machine learning model that predicts visa approval status. Additionally, provide recommendations for optimizing applicant profiles to improve approval likelihood.  

---

## 💡 Problem Statement  
The visa approval process is often subjective and time-consuming. By analyzing applicant data and identifying key factors influencing approval status, we can:
1. Automate the prediction of visa outcomes.
2. Provide actionable recommendations for applicants to improve their chances of approval.

---

## 🔧 Project Pipeline  
1. **Data Cleaning & Preprocessing**  
   - Handled missing values and outliers.
   - Encoded categorical variables (e.g., applicant occupation, education level).
   - Scaled numerical features for machine learning compatibility.

2. **Exploratory Data Analysis (EDA)**  
   - Identified trends and patterns in applicant data.
   - Analyzed the impact of features like education, occupation, and salary on visa status.

3. **Model Building**  
   - Built and tuned classification models such as Logistic Regression, Random Forest, and XGBoost.  
   - Selected the best-performing model based on precision, recall, and F1-score.

4. **Recommendations**  
   - Analyzed feature importance to suggest profiles likely to succeed in visa approval.

---

## 📂 Repository Contents  
- **`data/`**: Contains raw and cleaned datasets.  
- **`notebooks/`**: Step-by-step Jupyter notebooks for data cleaning, EDA, modeling, and recommendations.  
- **`models/`**: Saved predictive models and preprocessing tools.  
- **`images/`**: Visualizations from EDA and model results.  

---

## 📊 Exploratory Data Analysis Highlights  
- **Key Finding 1**: Applicants with higher education levels and higher salaries had an 85% higher chance of visa approval.  
- **Key Finding 2**: Certain occupations were more likely to result in visa denials due to policy restrictions.  

<p align="center">
  <img src="images/feature_importance.png" alt="Feature Importance" width="600">
</p>

---

## 🤖 Model Performance  
Tuned AdaBoost Classifier and Stacking Classifier have the highest F1 scores on the testing set and generalised well on both training and testing set.

The final model - Tuned AdaBoost Classifier will be chosen as the final model because Recall is higher at 0.87
Achieved the following metrics:
- **Recall**: 87.30%  
- **Precision**: 78.41%  
- **Accuracy**: 75.10%  
- **F1-Score**: 82.62%  

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Stacking Classifier | 75.79%      | 78.88%       | 86.74%    | 82.62%      |
| Tuned Random Forest | 74.80%      | 77.85%       | 87.82%    | 82.53%      |
| Tuned Gradient Boost | 74.42% | 76.98% | 88.82% | 82.48% |
| Tuned XGBoost Classifier | 75.25%      | 78.99%       | 86.49%    | 82.57%      |

---

## 📈 Recommendations  
Based on the model's insights, visa applicants are more likely to succeed if they:
1. **Job Experience**: Possess job experience in full-time position.
1. **Occupation**: Work in industries with higher visa approval rates (e.g., IT, healthcare).  
2. **Education**: Possess advanced degrees (e.g., Master’s or Ph.D.).  
3. **Salary**: Earn salaries above the threshold of $80,000/year.  

---

## 🛠 Tools & Technologies  
- **Programming Languages:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Sklearn.ensemble, Sklearn, Sklearn.metrics, Slearn.model_selection, XGBoost  
- **Visualization:** Matplotlib, Seaborn  
- **Version Control:** Git, GitHub  

---

## 🚀 How to Run the Project  
1. Clone this repository:
   ```bash
   git clone https://github.com/kemmy-dotcom/KemmyO_Visa-Approval-Prediction.git
