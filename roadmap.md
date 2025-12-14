## 📅 Example Project Timeline

### 🔹 Stage 1 – Project Setup
- [x] Create folder and source file structure  
- [x] Download dataset from Kaggle and save to `/data/`  
- [x] Create `requirements.txt` to enable easy virtual environment setup  
- [x] Create an initial proposal for the program structure (could change anytime when the need occurs).

### 🔹 Stage 2 – Exploratory Data Analysis (EDA)
- [x] Analyze feature distributions and descriptive statistics
- [x] Analyze data types in the dataset
- [x] Detect missing values
- [x] Detect outliers
- [x] Perform correlation analysis between features and target 
- [x] Visualize relationships between features
- [x] Evaluate data quality

### 🔹 Stage 3 – Data Preparation
- [x] Examine and handle missing values (fill or drop)  
- [x] Encode categorical variables  
- [x] Normalize / standardize features  
- [x] Split data into training and testing sets  

### 🔹 Stage 4 – Feature Selection
- [x] Perform correlation analysis between features and target (similarly to the step in EDA)
- [x] Determine the most important features (feature importance)
- [ ] Optionally MIC could be performed, but `minepy` is not compatible with currently used `numpy` version, it could be performed outside of the project environment.

### 🔹 Stage 5 – Modeling
- [x] Implement something like a `ModelTrainer` class
- [x] Train models: Logistic Regression, Random Forest, XGBoost, ...
- [x] Save the best models to files
- [x] Implement `ModelEvaluator` class
- [x] Create a simple YAML config file

### 🔹 Stage 6 – Evaluation and Interpretation
- [x] Compute evaluation metrics (Accuracy, Precision, Recall, F1, AUC) (might be done in stage 5 already)  
- [x] Visualize results (ROC curve, confusion matrix)
- [x] Create a dynamic configuration enabling easy testing and research
- [x] Refactor the code (`pipeline.py` mostly)
- [x] Add proper train/val/test split or use cross-validation to validate more reliably
- [x] Optimize models hyperparameters - run a few experiments with good testing methodology
- [x] Compare model performance - save results to a Markdown file or LaTeX report


### 🔹 Stage 7 – Reporting and Presentation
- [x] Prepare a report with results and conclusions
- [ ] Prepare a poster visualizing results
- [ ] Add missing unit tests for tests for `pipeline.py`, `utils.py`, and `config.py`

---

### ⚡ Notes on Configuration or creating a Dynamic Pipeline for research purposes - it would be great but the time and willingness might be an issue
 
The key idea is to **allow the pipeline to accept parameters for models, features, and data**, so that multiple experiments could be run sequentially without modifying the source code.  

This approach also allows saving **experiment results separately**, making it easier to compare models, hyperparameters, and preprocessing choices. 

---

### How it could work (outdated atm)

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
     test_size: 0.2  # idk if seed would also be useful
     models:
       logistic_regression:  # just an example
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
