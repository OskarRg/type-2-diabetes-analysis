## 📅 Example Project Timeline

### 🔹 Stage 1 – Project Setup
- [x] Create folder and source file structure  
- [x] Download dataset from Kaggle and save to `/data/`  
- [x] Create `requirements.txt` to enable easy virtual environment setup  
- [x] Create an initial proposal for the program structure (could change anytime when the need occurs).

### 🔹 Stage 2 – Exploratory Data Analysis (EDA)
- [ ] Analyze feature distributions and descriptive statistics  
- [ ] Detect missing values and outliers  
- [ ] Visualize relationships between features  

### 🔹 Stage 3 – Data Preparation
- [ ] Examine and handle missing values (fill or drop)  
- [ ] Encode categorical variables  
- [ ] Normalize / standardize features  
- [ ] Split data into training and testing sets  

### 🔹 Stage 4 – Feature Selection
- [ ] Perform correlation analysis between features and target  
- [ ] Determine the most important features (feature importance)  

### 🔹 Stage 5 – Modeling
- [ ] Implement `ModelTrainer` class  
- [ ] Train models: Logistic Regression, Random Forest, XGBoost  
- [ ] Save the best models to a file  

### 🔹 Stage 6 – Evaluation and Interpretation
- [ ] Compute evaluation metrics (Accuracy, Precision, Recall, F1, AUC)  
- [ ] Visualize results (ROC curve, confusion matrix)  
- [ ] Compare model performance  

### 🔹 Stage 7 – Reporting and Presentation
- [ ] Prepare a report with results and conclusions
- [ ] Prepare a poster visualizing results 

---

### ⚡ Notes on Configuration or creating a Dynamic Pipeline for research purposes - it would be great but the time and willingness might be an issue
 
The key idea is to **allow the pipeline to accept parameters for models, features, and data**, so that multiple experiments could be run sequentially without modifying the source code.  

This approach also allows saving **experiment results separately**, making it easier to compare models, hyperparameters, and preprocessing choices. 

---

### How it could Works

1. **Dynamic `Pipeline.run()`**  
   - The `run()` method would be dependant on parameters like:
     - `model_type`: Which model to train (e.g., `"random_forest"`)  
     - `top_n_features`: How many features to use based on feature importance  
     - `experiment_name`: Optional name to save results separately  
     - and more params connected to data preprocessing and training

2. **Optional configuration file**  
   - Storing experiment parameters in a JSON or YAML file would prove very convenient:
     ```yaml
     data_path: "data/diabetes.csv"
     target: "Outcome"
     test_size: 0.2
     models:
       logistic_regression:
         C: 1.0
         max_iter: 1000
       random_forest:
         n_estimators: 200
         max_depth: 10
     ```

3. **Experiment versioning**  
   - Each experiment is saved in a dedicated folder:
     ```
     results/
       experiment_01_random_forest/
         model.pkl
         metrics.json
         plots/
       experiment_02_xgboost/
         ...
     ```
   - This makes it easy to compare different runs and track model performance over time.
