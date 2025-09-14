# Titanic ML Project

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/stable/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-brightgreen.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=jupyter)](https://jupyter.org/)

Proyecto académico para analizar el **dataset del Titanic** usando técnicas modernas de *Machine Learning*, con énfasis en **interpretabilidad** y **fairness**.  

---

## Objetivos

- Identificar los **factores determinantes de supervivencia** en el Titanic.  
- Implementar un **pipeline reproducible** con múltiples modelos (Logistic Regression, Random Forest, XGBoost, SVM, Neural Networks).  
- Evaluar desempeño con métricas clásicas (Accuracy, F1, ROC-AUC) y métricas de equidad (*Demographic Parity, Equalized Odds*).  
- Explorar técnicas de interpretabilidad (**SHAP, LIME, feature importance**).  
- Comunicar resultados en un **paper, presentación y dashboard interactivo**.  

---

## Estructura del Repositorio

```bash
titanic-ml-project/
├── README.md                 # Este archivo
├── requirements.txt          # Dependencias exactas
├── LICENSE
├── data/
│   ├── raw/                  # Datos originales
│   ├── processed/            # Datos listos para modelar
│   └── README.md             # Documentación de fuentes de datos
├── notebooks/
│   ├── 01_EDA.ipynb          # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Modeling.ipynb
│   ├── 04_Interpretability.ipynb
│   └── 05_Fairness.ipynb
├── models/
│   └── best_model.pkl        # Modelo entrenado final
├── results/
│   ├── figures/              # Figuras del paper
│   ├── tables/               # Tablas en CSV
│   └── metrics.json          # Métricas finales en JSON
├── paper/
│   ├── main.tex              # Manuscrito en LaTeX
│   ├── main.pdf              # Paper compilado
│   └── references.bib
├── presentation/
│   └── slides.pdf            # Presentación final
└── dashboard/
    └── app.py                # Dashboard interactivo en Streamlit/Flask
