# Fog-Cloud Task Scheduler using Machine Learning

## 🚀 Overview
This project predicts task execution time in fog-cloud environments using machine learning.

Traditional scheduling methods use simple formulas and often give inaccurate results.  
This project uses ML to make better predictions and improve decision-making.

---

## 🧠 Features
- Synthetic dataset generation
- Machine learning model (Random Forest)
- Execution time prediction
- Best node selection for task scheduling

---

## ⚙️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- (Optional) Streamlit

---

## 📊 How it Works
1. Generate dataset with task and node features  
2. Train ML model on execution time  
3. Predict execution time for new inputs  
4. Select best node with lowest predicted time  

---

## ▶️ How to Run

```bash
python data_generator.py
python train_model.py
python predict.py
