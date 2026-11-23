# Cyber Threat Detection Based on Artificial Neural Networks Using Event Profiles

This was my B.Tech final year project. It is based on the IEEE Access paper with the same title.

The idea: security logs are huge, and rule-based systems miss a lot. So the app turns raw network log records into "event profiles" with TF-IDF, then trains a neural network to tell normal traffic from attacks. I compared it with classic machine learning models to see which one works best.

## What the app does

It is a small Tkinter desktop app. You go through the buttons in order:

1. Upload the training dataset
2. Preprocess the data with TF-IDF
3. Generate event vectors
4. Train the neural network (Keras)
5. Train SVM, KNN, Random Forest, Naive Bayes and Decision Tree for comparison
6. Extension: SVM tuned with Particle Swarm Optimization (PSO)
7. Compare accuracy, precision, recall and F-measure in graphs

## Datasets

- CIC-IDS2017 (the day-wise CSV files)
- NSL-KDD (`kdd_train.csv`, `kdd_test.csv`)

## Tech stack

Python, pandas, NumPy, scikit-learn, TensorFlow / Keras, Matplotlib, Tkinter

## How to run

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib
python CyberThreatDetection.py
```

On Windows you can also double-click `run.bat`.

## What I learned

- How to turn text-like log data into numbers a model can learn from
- How much preprocessing matters compared to the model choice
- Comparing models fairly with precision, recall and F1, not only accuracy
