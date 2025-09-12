##run with  streamlit run dashboard/dashboard.py
import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st

from typing import Dict, Any, Tuple, List
from dataclasses import dataclass

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, matthews_corrcoef,
                             balanced_accuracy_score, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.base import clone

try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    _HAS_LIME = True
except Exception:
    _HAS_LIME = False

# config
st.set_page_config(page_title="Titanic Survival — Interactive Modeling", layout="wide")

SAFE_FEATURES = ['Title','FamilySize','IsAlone','AgeGroup','FarePerPerson','IsMother',
                 'FareAboveMedian','CabinKnown','Pclass','Sex','Embarked','Age','Fare']
TARGET = 'Survived'

# subir el dataset
@st.cache_data
def load_data(default_path: str = "data/Titanic-Dataset.csv") -> pd.DataFrame:
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
    else:
        st.warning(f"No se encontró {default_path}. Sube un CSV con las columnas esperadas.")
        df = pd.DataFrame()
    return df

df = load_data()

with st.sidebar:
    st.header("Datos")
    uploaded = st.file_uploader("Sube tu CSV (opcional)", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        df.to_csv("Sesgos Entrega 2.csv", index=False)
        st.success("CSV cargado.")

    st.markdown("---")
    st.caption("Modelos incluidos: RandomForest, LogisticRegression, XGBoost/LightGBM (si disponibles), SVM, Neural Network.")
    seed = st.number_input("Random state", min_value=0, value=42, step=1)

# pre procesado del archivo
def basic_clean(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in ['Age','Fare','FarePerPerson']:
        if c in X.columns:
            X[c] = X[c].fillna(X[c].median())
    for c in ['AgeGroup','Title','Embarked','Sex']:
        if c in X.columns:
            X[c] = X[c].astype(object).fillna("Unknown")
    return X

def build_preprocessors(X: pd.DataFrame):
    cat_feats = [c for c in X.columns if X[c].dtype == 'object'] + (['Pclass'] if 'Pclass' in X.columns else [])
    cat_feats = list(dict.fromkeys(cat_feats))
    num_feats = [c for c in X.columns if X[c].dtype != 'object' and c not in cat_feats]

    pre_tree = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats),
        ("num", "passthrough", num_feats)
    ])

    pre_linear = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats),
        ("num", Pipeline(steps=[("scaler", StandardScaler())]), num_feats)
    ])
    return pre_tree, pre_linear, cat_feats, num_feats

# modelos
def make_models(seed: int, for_tree_pre, for_linear_pre):
    models = {}

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=7, min_samples_split=2, min_samples_leaf=1,
        max_features='sqrt', class_weight='balanced', random_state=seed,
        bootstrap=True, oob_score=False
    )
    models['RandomForest'] = Pipeline([("pre", for_tree_pre), ("clf", rf)])

    # Logistic Regression
    lr = LogisticRegression(
        max_iter=2000, class_weight='balanced', solver='saga', random_state=seed
    )
    models['LogisticRegression'] = Pipeline([("pre", for_linear_pre), ("clf", lr)])

    # XGBoost
    xgb_ok = False
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, random_state=seed, eval_metric="logloss"
        )
        models['XGBoost'] = Pipeline([("pre", for_tree_pre), ("clf", xgb)])
        xgb_ok = True
    except Exception:
        pass
    if not xgb_ok:
        gb = GradientBoostingClassifier(random_state=seed)
        models['XGBoost (fallback=SklearnGB)'] = Pipeline([("pre", for_tree_pre), ("clf", gb)])

    # LightGBM
    lgb_ok = False
    try:
        from lightgbm import LGBMClassifier
        lgb = LGBMClassifier(
            n_estimators=600, max_depth=-1, num_leaves=31, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=seed
        )
        models['LightGBM'] = Pipeline([("pre", for_tree_pre), ("clf", lgb)])
        lgb_ok = True
    except Exception:
        pass
    if not lgb_ok:
        hgb = HistGradientBoostingClassifier(random_state=seed)
        models['LightGBM (fallback=HistGB)'] = Pipeline([("pre", for_tree_pre), ("clf", hgb)])

    # SVM (probabilities ON)
    svm = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=seed)
    models['SVM'] = Pipeline([("pre", for_linear_pre), ("clf", svm)])

    # Neural Net
    mlp = MLPClassifier(hidden_layer_sizes=(64,32), activation='relu', alpha=1e-4,
                        learning_rate_init=1e-3, max_iter=150, random_state=seed)
    models['NeuralNet'] = Pipeline([("pre", for_linear_pre), ("clf", mlp)])

    return models

# Train/test split
def split_xy(df: pd.DataFrame, seed: int):
    assert TARGET in df.columns, "Column 'Survived' not found in the dataset."
    cols = [c for c in SAFE_FEATURES if c in df.columns]
    X = basic_clean(df[cols].copy())
    y = df[TARGET].astype(int).copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed
    )
    return X_train, X_test, y_train, y_test

@st.cache_resource
def fit_all_models(df: pd.DataFrame, seed: int):
    X_train, X_test, y_train, y_test = split_xy(df, seed)
    pre_tree, pre_linear, cat_feats, num_feats = build_preprocessors(X_train)
    models = make_models(seed, pre_tree, pre_linear)

    fitted = {}
    metrics = {}
    for name, pipe in models.items():
        with st.spinner(f"Entrenando {name}..."):
            pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:,1]

        metrics[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "pr_auc": average_precision_score(y_test, y_proba),
            "mcc": matthews_corrcoef(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        }
        fitted[name] = pipe

    info = {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "cat_feats": cat_feats, "num_feats": num_feats
    }
    return fitted, pd.DataFrame(metrics).T.sort_values("roc_auc", ascending=False), info

# EDA
def eda_section(df: pd.DataFrame):
    st.header("4.1 Exploración (EDA)")
    if df.empty:
        st.info("Carga un CSV para comenzar.")
        return
    cols = [c for c in SAFE_FEATURES + [TARGET] if c in df.columns]
    dfx = basic_clean(df[cols].copy())

    # filros
    with st.expander("Filtros demográficos"):
        c1, c2, c3, c4 = st.columns(4)
        sex_f = c1.multiselect("Sex", sorted(dfx["Sex"].dropna().unique().tolist()) if "Sex" in dfx else [], default=[])
        pclass_f = c2.multiselect("Pclass", sorted(dfx["Pclass"].dropna().unique().tolist()) if "Pclass" in dfx else [], default=[])
        ageg_f = c3.multiselect("AgeGroup", sorted(dfx["AgeGroup"].dropna().unique().tolist()) if "AgeGroup" in dfx else [], default=[])
        cabin_f = c4.multiselect("CabinKnown", sorted(dfx["CabinKnown"].dropna().unique().tolist()) if "CabinKnown" in dfx else [], default=[])

    mask = pd.Series(True, index=dfx.index)
    if sex_f and "Sex" in dfx: mask &= dfx["Sex"].isin(sex_f)
    if pclass_f and "Pclass" in dfx: mask &= dfx["Pclass"].isin(pclass_f)
    if ageg_f and "AgeGroup" in dfx: mask &= dfx["AgeGroup"].isin(ageg_f)
    if cabin_f and "CabinKnown" in dfx: mask &= dfx["CabinKnown"].isin(cabin_f)

    dff = dfx[mask].copy()
    st.write(f"Registros filtrados: **{len(dff)}** / {len(dfx)}")

    # Summary stats
    c1, c2, c3 = st.columns(3)
    if TARGET in dff:
        surv_rate = dff[TARGET].mean()
        c1.metric("Tasa de supervivencia", f"{surv_rate:.2%}")
    if "FarePerPerson" in dff:
        c2.metric("Mediana FarePerPerson", f"{dff['FarePerPerson'].median():.2f}")
    if "Age" in dff:
        c3.metric("Edad mediana", f"{dff['Age'].median():.1f}")

    st.subheader("Distribuciones")
    c1, c2 = st.columns(2)
    if "Sex" in dff and TARGET in dff:
        sex_pivot = dff.groupby("Sex")[TARGET].mean().reset_index()
        c1.bar_chart(sex_pivot.set_index("Sex"))
    if "Pclass" in dff and TARGET in dff:
        p_pivot = dff.groupby("Pclass")[TARGET].mean().reset_index()
        c2.bar_chart(p_pivot.set_index("Pclass"))
    st.caption("Nota: barras muestran tasa media de `Survived` por categoría.")

    st.dataframe(dff.head(100))

# predicciones
def single_prediction_ui(models: Dict[str, Pipeline], info: Dict[str, Any]):
    st.header("4.2 Predicción en tiempo real")
    X_train = info["X_train"]
    # Build input widgets from training columns
    st.subheader("Ingresar datos del pasajero")
    inputs = {}
    ccols = st.columns(3)
    # Categorical-like
    inputs['Sex'] = ccols[0].selectbox("Sex", sorted(X_train['Sex'].unique()), index=0 if 'Sex' not in X_train else 0)
    inputs['Pclass'] = ccols[1].selectbox("Pclass", sorted(X_train['Pclass'].unique()) if 'Pclass' in X_train else [1,2,3], index=0)
    inputs['Embarked'] = ccols[2].selectbox("Embarked", sorted(X_train['Embarked'].unique()) if 'Embarked' in X_train else ["S","C","Q"], index=0)

    c2 = st.columns(3)
    inputs['Title'] = c2[0].selectbox("Title", sorted(X_train['Title'].unique()) if 'Title' in X_train else ["Mr","Mrs","Miss","Master","Other"], index=0)
    inputs['AgeGroup'] = c2[1].selectbox("AgeGroup", sorted(X_train['AgeGroup'].unique()) if 'AgeGroup' in X_train else ["0-12","13-18","19-64","65+"], index=0)
    inputs['CabinKnown'] = c2[2].selectbox("CabinKnown", sorted(X_train['CabinKnown'].unique()) if 'CabinKnown' in X_train else [0,1], index=0)

    c3 = st.columns(3)
    inputs['Age'] = c3[0].slider("Age", 0, 80, int(X_train['Age'].median() if 'Age' in X_train else 30))
    inputs['Fare'] = c3[1].slider("Fare", 0.0, float(np.nanmax(X_train['Fare']) if 'Fare' in X_train else 100.0), float(np.nanmedian(X_train['Fare']) if 'Fare' in X_train else 20.0))
    inputs['FarePerPerson'] = c3[2].slider("FarePerPerson", 0.0, float(np.nanmax(X_train['FarePerPerson']) if 'FarePerPerson' in X_train else 100.0), float(np.nanmedian(X_train['FarePerPerson']) if 'FarePerPerson' in X_train else 20.0))

    c4 = st.columns(3)
    inputs['FamilySize'] = c4[0].slider("FamilySize", 0, 11, int(X_train['FamilySize'].median() if 'FamilySize' in X_train else 1))
    inputs['IsAlone'] = c4[1].selectbox("IsAlone", [0,1], index=0)
    inputs['IsMother'] = c4[2].selectbox("IsMother", [0,1], index=0)

    c5 = st.columns(2)
    inputs['FareAboveMedian'] = c5[0].selectbox("FareAboveMedian", [0,1], index=0)

    row = pd.DataFrame([inputs])

    # predecir con todos los modelos
    st.subheader("Predicciones")
    preds = {}
    for name, model in models.items():
        try:
            proba = model.predict_proba(row)[:,1][0]
            preds[name] = proba
        except Exception as e:
            preds[name] = np.nan
            st.warning(f"No se pudo predecir con {name}: {e}")
    st.dataframe(pd.DataFrame.from_dict(preds, orient='index', columns=['Prob(Survive)']).sort_values('Prob(Survive)', ascending=False))

    # Confidence intervals via perturbation bootstrap
    st.markdown("**Intervalos de confianza (aprox.)**")
    n_mc = st.slider("Muestras para bootstrap local", 50, 500, 200, step=50)
    jitter = 0.02  # 2% noise for numeric fields
    num_cols = [c for c in row.columns if row[c].dtype != 'object' and c not in ['Pclass']]
    pert = pd.concat([row]*n_mc, ignore_index=True)
    for c in num_cols:
        sigma = max(1e-6, float(X_train[c].std()) * jitter) if c in X_train else 0.1
        pert[c] = pert[c].astype(float) + np.random.normal(0, sigma, size=n_mc)

    cis = {}
    for name, model in models.items():
        try:
            ps = model.predict_proba(pert)[:,1]
            lo, hi = np.quantile(ps, [0.025, 0.975])
            cis[name] = (float(lo), float(hi))
        except Exception:
            cis[name] = (None, None)
    st.dataframe(pd.DataFrame(cis, index=['2.5%','97.5%']).T)

    # Explanations (try SHAP, else local permutation)
    st.subheader("Explicación local (SHAP/LIME si disponibles)")
    exp_model = st.selectbox("Modelo para explicar", options=list(models.keys()))
    if _HAS_SHAP:
        st.caption("Usando SHAP (Kernel/Tree explainer según modelo)")
        model = models[exp_model]
        try:
            # Try tree explainer if model is tree-based
            is_tree = any(k in exp_model.lower() for k in ['forest','xgb','lgb','boost'])
            if is_tree:
                explainer = shap.Explainer(model.predict_proba, masker=X_train, algorithm='auto')
            else:
                explainer = shap.KernelExplainer(model.predict_proba, X_train.sample(min(200, len(X_train)), random_state=0))
            sv = explainer(row, max_evals=600)
            shap_values = sv.values[0,:,1] if hasattr(sv, "values") else sv[0].values[:,1]
            contrib = pd.Series(shap_values, index=row.columns).sort_values(key=np.abs, ascending=False)
            st.dataframe(contrib.to_frame("SHAP value").head(15))
        except Exception as e:
            st.warning(f"SHAP falló: {e}. Usando permutación local.")
    if _HAS_LIME and df.shape[0] > 0:
        st.caption("(Opcional) LIME Tabular")
        try:
            expl = LimeTabularExplainer(training_data=np.array(X_train), feature_names=X_train.columns.tolist(),
                                        class_names=['No','Yes'], discretize_continuous=True, mode='classification')
            exp = expl.explain_instance(np.array(row.iloc[0]), models[exp_model].predict_proba, num_features=10)
            st.text(exp.as_list())
        except Exception as e:
            st.warning(f"LIME falló: {e}")

    st.caption("Si no hay SHAP/LIME, usa análisis de permutación local: medir Δ de probabilidad al barajar cada variable alrededor de esta instancia.")

# analisis de modelo
def model_analysis_section(models: Dict[str, Pipeline], metrics_df: pd.DataFrame, info: Dict[str, Any]):
    st.header("4.3 Análisis de Modelos")
    st.subheader("Comparación de métricas")
    st.dataframe(metrics_df.style.format({col:"{:.4f}" for col in metrics_df.columns if col!='support'}))

    # Feature importance (Permutation) for selected model
    st.subheader("Feature importance (Permutation)")
    X_test, y_test = info["X_test"], info["y_test"]
    model_name = st.selectbox("Modelo para importancias", list(models.keys()))
    model = models[model_name]

    with st.spinner("Calculando importancias por permutación (ROC-AUC)..."):
        res = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=0, scoring='roc_auc')
    perm_imp = pd.Series(res.importances_mean, index=X_test.columns).sort_values(ascending=False)
    st.bar_chart(perm_imp.head(20))

    # Error analysis
    st.subheader("Análisis de errores")
    y_pred = model.predict(X_test)
    errs = X_test.copy()
    errs["y_true"] = info["y_test"].values
    errs["y_pred"] = y_pred
    errs["correct"] = errs["y_true"] == errs["y_pred"]
    view = st.radio("Mostrar", ["Mal clasificados", "Bien clasificados"])
    subset = errs[~errs["correct"]] if view == "Mal clasificados" else errs[errs["correct"]]
    st.dataframe(subset.head(200))

# What-if
def what_if_section(models: Dict[str, Pipeline], info: Dict[str, Any]):
    st.header('4.4 What-If / Contrafactual')
    st.caption("Modifica características con sliders para ver el impacto en las probabilidades.")

    X_train = info["X_train"]
    base_idx = st.number_input("Fila base del set de validación (índice)", min_value=0, max_value=len(info["X_test"])-1, value=0, step=1)
    base_row = info["X_test"].iloc[[base_idx]].copy()

    st.write("**Observación base**")
    st.json(base_row.to_dict(orient='records')[0])

    # Controls to modify numeric features
    mods = {}
    for c in [c for c in base_row.columns if base_row[c].dtype != 'object' and c not in ['Pclass']]:
        val = float(base_row[c].iloc[0])
        mods[c] = st.slider(f"{c}", float(max(0, val*0.0)), float(val*2 + 10), float(val), step=0.5)

    # Apply modifications
    row2 = base_row.copy()
    for k,v in mods.items():
        row2[k] = v

    st.subheader("Probabilidades por modelo")
    probs_base = {}
    probs_mod = {}
    for name, model in models.items():
        try:
            probs_base[name] = float(model.predict_proba(base_row)[:,1][0])
            probs_mod[name]  = float(model.predict_proba(row2)[:,1][0])
        except Exception as e:
            probs_base[name], probs_mod[name] = np.nan, np.nan
    dfp = pd.DataFrame({"base": probs_base, "mod": probs_mod})
    st.dataframe(dfp)

    st.caption("La diferencia (mod - base) ilustra el efecto contrafactual de los cambios introducidos.")

#Main
st.title("Dashboard Interactivo — Supervivencia en el Titanic")

if df.empty or TARGET not in df.columns:
    st.info("Carga un dataset con la columna objetivo `Survived` y las features conocidas.")
else:
    fitted, metrics_df, info = fit_all_models(df, seed)

    tabs = st.tabs(["Exploración", "Predicción", "Análisis de Modelos", "What-If"])
    with tabs[0]:
        eda_section(df)
    with tabs[1]:
        single_prediction_ui(fitted, info)
    with tabs[2]:
        model_analysis_section(fitted, metrics_df, info)
    with tabs[3]:
        what_if_section(fitted, info)
