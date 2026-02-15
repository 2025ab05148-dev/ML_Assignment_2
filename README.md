\# **Stroke Prediction using Machine Learning**



\## **1. Problem Statement**



The objective of this project is to develop multiple machine learning classification models to predict whether a person is likely to suffer a stroke based on clinical and demographic features. The project demonstrates a complete end-to-end ML pipeline including preprocessing, model training, evaluation using multiple metrics, and deployment using Streamlit.







**## 2. Dataset Description**



The dataset used is the \*\*Stroke Prediction Dataset\*\* from Kaggle.



\- Total Records: 5110  

\- Total Features: 12  

\- Target Variable: `stroke`  

&nbsp; - 0 → No Stroke  

&nbsp; - 1 → Stroke  



Features include:

\- Gender  

\- Age  

\- Hypertension  

\- Heart Disease  

\- Ever Married  

\- Work Type  

\- Residence Type  

\- Average Glucose Level  

\- BMI  

\- Smoking Status  



\### Data Preprocessing Steps:

\- Removed unnecessary ID column  

\- Handled missing BMI values using mean imputation  

\- Encoded categorical variables using Label Encoding  

\- Addressed class imbalance using upsampling of minority class  


\###Dataset Source

The dataset was obtained from Kaggle:

https://www.kaggle.com/fedesoriano/stroke-prediction-dataset

Note:
Due to submission size constraints, the full dataset is not included in this repository.
The dataset can be downloaded directly from the Kaggle link above.

Reproducibility

All preprocessing steps and model training procedures are clearly defined in train.py.



---



**## 3. Machine Learning Models Implemented**



The following six classification models were implemented on the same dataset:



1\. Logistic Regression  

2\. Decision Tree Classifier  

3\. K-Nearest Neighbors (KNN)  

4\. Naive Bayes (GaussianNB)  

5\. Random Forest (Ensemble Model)  

6\. XGBoost (Ensemble Model)  



---



**## 4. Evaluation Metrics Used**



Each model was evaluated using the following performance metrics:



\- Accuracy  

\- AUC Score  

\- Precision  

\- Recall  

\- F1 Score  

\- Matthews Correlation Coefficient (MCC)  



---



**## 5. Model Comparison Table**



| ML Model              | Accuracy | AUC     | Precision | Recall  | F1 Score | MCC     |

|-----------------------|----------|---------|-----------|---------|----------|---------|

| Logistic Regression   | 0.7794   | 0.8523  | 0.7751    | 0.8008  | 0.7877   | 0.5586  |

| Decision Tree         | 0.9666   | 0.9658  | 0.9386    | 1.0000  | 0.9683   | 0.9351  |

| KNN                   | 0.9378   | 0.9713  | 0.8915    | 1.0000  | 0.9426   | 0.8821  |

| Naive Bayes           | 0.7707   | 0.8314  | 0.7729    | 0.7807  | 0.7768   | 0.5411  |

| Random Forest         | 0.9933   | 1.0000  | 0.9871    | 1.0000  | 0.9935   | 0.9867  |

| XGBoost               | 0.9717   | 0.9972  | 0.9476    | 1.0000  | 0.9731   | 0.9449  |



---



**## 6. Observations**



\*\*Logistic Regression:\*\*  

Performed moderately well but achieved lower MCC compared to ensemble models.



\*\*Decision Tree:\*\*  

High recall (1.0) indicates excellent detection of stroke cases. Slight possibility of overfitting.



\*\*KNN:\*\*  

Strong recall and balanced F1 score after feature scaling.



\*\*Naive Bayes:\*\*  

Lower AUC compared to tree-based models due to independence assumption of features.



\*\*Random Forest:\*\*  

Best overall performing model with highest Accuracy (99.33%), AUC (1.00), and MCC (0.9867). Demonstrates excellent generalization ability.



\*\*XGBoost:\*\*  

Very strong performance with high AUC (0.9972) and F1 score. Slightly below Random Forest but highly robust.







**## 7. Streamlit Web Application**



An interactive Streamlit web application was developed with the following features:



\- CSV file upload option  

\- Model selection dropdown  

\- Real-time stroke prediction  

\- Interactive data preview  



Live App Link: (Add your Streamlit deployment link here)







**## 8. GitHub Repository**



GitHub Repository Link: https://github.com/2025ab05148-dev/ML_Assignment_2



---



**## . Conclusion**



This project demonstrates a complete machine learning workflow including preprocessing, handling class imbalance, training multiple classification models, evaluating them using comprehensive metrics, and deploying the best models using Streamlit for real-time prediction.



Ensemble models such as Random Forest and XGBoost significantly outperformed traditional models after addressing dataset imbalance.



---







## Streamlit App Features
- CSV dataset upload option
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix and classification report
