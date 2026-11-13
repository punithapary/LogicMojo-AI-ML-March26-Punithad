AI-Powered Financial Fraud & Risk Intelligence Platform
Project Overview
This project explores how Artificial Intelligence and Machine Learning can help identify fraudulent financial transactions. The PaySim synthetic financial transaction dataset is used to build and evaluate fraud detection models.
Objectives
- Identify potentially fraudulent transactions.
- Compare different Machine Learning models.
- Experiment with a Deep Learning model.
- Evaluate model performance using suitable classification metrics.
- Support better understanding of financial fraud detection.
Dataset
The project uses the PaySim synthetic financial transaction dataset.
- Target column: isFraud
- Selected features: step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
- Identifier columns and isFlaggedFraud are excluded from the main model features.
Note: PaySim is a synthetic dataset. Results on this dataset do not guarantee performance on real financial transactions.

Technologies Used
- Python
- Jupyter Notebook
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- Joblib
Models
1. Dummy Classifier – baseline for comparison.
2. Logistic Regression – traditional Machine Learning model.
3. Random Forest – ensemble Machine Learning model.
4. Neural Network – Deep Learning experiment.
Evaluation Metrics
- Precision
- Recall
- F1-score
- Average Precision (PR-AUC)
- ROC-AUC
- Confusion Matrix
These metrics help assess how well the models identify rare fraud cases while considering false alarms.
Project Structure
project-folder/
├── data/
│   └── payment_dataset.csv
├── notebooks/
│   └── fraud_detection.ipynb
├── artifacts/
│   ├── best_ml_model.joblib
│   └── ml_validation_comparison.csv
├── utils/
├── README.md
└── requirements.txt
Update the file and folder names above to match your actual project structure.
How to Run
1. Place the PaySim dataset in the data/ folder.
2. Open the project folder in Jupyter Notebook.
3. Install the required packages if needed:
   %pip install pandas numpy matplotlib scikit-learn joblib tensorflow
4. Update the dataset path in the notebook if your file name or folder is different.
5. Run the notebook cells in order, starting with data loading and preprocessing.
6. Review the model comparison, evaluation metrics, confusion matrices, and saved output files.
Results
Enter the actual results produced by your notebook below. Do not use assumed values.
Metric	Result
Precision	Add actual result
Recall	Add actual result
F1-score	Add actual result
Average Precision (PR-AUC)	Add actual result
ROC-AUC	Add actual result


Also identify which model produced these results and whether they came from the validation set or the final test set.
Conclusion
This project demonstrates the use of traditional Machine Learning and a Deep Learning experiment for financial fraud detection. The models are compared using metrics suited to an imbalanced classification problem. The results provide an experimental foundation for further fraud detection work. Additional validation using real-world financial data and operational requirements would be needed before production deployment.
