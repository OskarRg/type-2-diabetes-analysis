## Project Description

The goal of this project is to develop a predictive model that estimates the risk of developing Type II diabetes based on patient data.  
The project involves data analysis to identify relationships between medical features and lifestyle factors with the probability of developing diabetes.  
Additionally, it focuses on identifying the most important variables, which will facilitate the creation of the predictive model.

---

## Dataset

The dataset **“Diabetes Dataset with 18 Features”**, available on [Kaggle](https://www.kaggle.com/datasets/pkdarabi/diabetes-dataset-with-18-features), was designed to analyze factors influencing diabetes occurrence.  
It contains information about patients, their health parameters, and lifestyle habits.

### Dataset Characteristics

- **Source:** [Kaggle: Diabetes Dataset with 18 Features](https://www.kaggle.com/datasets/pkdarabi/diabetes-dataset-with-18-features)  
- **Number of features:** 18  
- **Number of observations:** 4303  
- **Data format:** CSV  

### Data Structure

The dataset consists of 18 variables describing, among others:

- **Demographic parameters** (e.g., age, gender)  
- **Clinical parameters** (e.g., glucose level, blood pressure, BMI)  
- **Lifestyle information** (e.g., smoking, drinking)  
- **Other factors**

## Start The Project

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
.venv\Scripts\activate          # Windows
```
### 2. Install requirements
```
pip install -r requirements.txt
```

### 3. Current way to run the project does not include any params or configuration file

```
python main.py
```

### 4. Remember to test the project after making changes

```
pytest -vv
```