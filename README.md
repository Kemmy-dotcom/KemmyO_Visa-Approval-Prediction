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
The final model achieved the following metrics:
- **Accuracy**: 88%  
- **Precision**: 85%  
- **Recall**: 90%  
- **F1-Score**: 87%  

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 84%      | 82%       | 86%    | 84%      |
| Random Forest       | 88%      | 85%       | 90%    | 87%      |
| XGBoost             | 89%      | 86%       | 91%    | 88%      |

---

## 📈 Recommendations  
Based on the model's insights, visa applicants are more likely to succeed if they:
1. **Occupation**: Work in industries with higher visa approval rates (e.g., IT, healthcare).  
2. **Education**: Possess advanced degrees (e.g., Master’s or Ph.D.).  
3. **Salary**: Earn salaries above the threshold of $80,000/year.  

---

## 🛠 Tools & Technologies  
- **Programming Languages:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost  
- **Visualization:** Matplotlib, Seaborn  
- **Version Control:** Git, GitHub  

---

## 🚀 How to Run the Project  
1. Clone this repository:
   ```bash
   git clone https://github.com/kemmy-dotcom/Visa-Approval-Prediction.git
