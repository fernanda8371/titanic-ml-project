##run with  streamlit run dashboard/dashboard.py
import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import re

from typing import Dict, Any, List, Tuple

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance

# Optional imports for explainability
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


st.set_page_config(
    page_title="Titanic ML Dashboard 🚢",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
    }
    .feature-importance-bar {
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        height: 8px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Constantes
SAFE_FEATURES = [
    'Title', 'FamilySize', 'IsAlone', 'AgeGroup', 'FarePerPerson', 'IsMother',
    'FareAboveMedian', 'CabinKnown', 'Pclass', 'Sex', 'Embarked', 'Age', 'Fare'
]
TARGET = 'Survived'

def extract_title(name: str) -> str:
    """Extrae el título del nombre del pasajero."""
    title_search = re.search(r' ([A-Za-z]+)\.', str(name))
    return title_search.group(1) if title_search else ""

def _ohe(*, handle_unknown="ignore"):
    """
    OneHotEncoder compatible con versiones antiguas y nuevas de sklearn.
    Usa sparse_output si existe; de lo contrario usa sparse=False.
    """
    try:
        return OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown=handle_unknown, sparse=False)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Realiza feature engineering en el dataset."""
    df = df.copy()

    # Extraer título del nombre
    if 'Name' in df.columns:
        df['Name'] = df['Name'].astype(str)
        df['Title'] = df['Name'].apply(extract_title)
        df['Title'] = df['Title'].replace(
            ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major',
             'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare'
        )
        df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

    # Tamaño de la familia
    if 'SibSp' in df.columns and 'Parch' in df.columns:
        df['FamilySize'] = df['SibSp'].fillna(0) + df['Parch'].fillna(0) + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Tarifa por persona (manejo seguro de inf/NaN fila a fila)
    if 'Fare' in df.columns and 'FamilySize' in df.columns:
        df['FarePerPerson'] = df['Fare'] / df['FamilySize']
        mask = ~np.isfinite(df['FarePerPerson'])
        # Reemplazar solo esas filas por la tarifa original
        df.loc[mask, 'FarePerPerson'] = df.loc[mask, 'Fare']

    # Grupos de edad
    if 'Age' in df.columns:
        df['AgeGroup'] = pd.cut(
            df['Age'],
            bins=[0, 12, 18, 35, 60, 100],
            labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior']
        )

    # Es madre (mujer casada con hijos)
    if {'Sex', 'Parch', 'Title'}.issubset(df.columns):
        df['IsMother'] = ((df['Sex'] == 'female') & (df['Parch'] > 0) &
                          (df['Title'] == 'Mrs')).astype(int)

    # Tarifa sobre la mediana
    if 'Fare' in df.columns:
        median_fare = df['Fare'].median()
        df['FareAboveMedian'] = (df['Fare'] > median_fare).astype(int)

    # Cabina conocida
    if 'Cabin' in df.columns:
        df['CabinKnown'] = df['Cabin'].notna().astype(int)

    return df

@st.cache_data
def load_data(default_path: str = r"..\data\Titanic-Dataset.csv") -> pd.DataFrame:
    """Carga el dataset del Titanic desde la ruta especificada (si existe)."""
    if os.path.exists(default_path):
        try:
            df = pd.read_csv(default_path)
            df = engineer_features(df)
            st.sidebar.success(f"Dataset cargado: {len(df)} registros")
            return df
        except Exception as e:
            st.sidebar.error(f"Error al cargar el dataset: {e}")
            return pd.DataFrame()
    else:
        st.sidebar.warning(f"No se encontró {default_path}")
        return pd.DataFrame()

def basic_clean(X: pd.DataFrame) -> pd.DataFrame:
    """Limpieza básica del dataset."""
    X = X.copy()

    # Imputación de valores numéricos
    numeric_cols = ['Age', 'Fare', 'FarePerPerson']
    for col in numeric_cols:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].median())

    # Imputación de valores categóricos
    categorical_cols = ['AgeGroup', 'Title', 'Embarked', 'Sex']
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype(object).fillna("Unknown")

    return X

def build_preprocessors(X: pd.DataFrame) -> Tuple[ColumnTransformer, ColumnTransformer, List[str], List[str]]:
    """Construye preprocesadores para modelos tree-based y lineales."""
    cat_feats = [c for c in X.columns if X[c].dtype == 'object']
    if 'Pclass' in X.columns:
        cat_feats.append('Pclass')  # tratar Pclass como categórica
    cat_feats = list(dict.fromkeys(cat_feats))

    num_feats = [c for c in X.columns if (X[c].dtype != 'object') and (c not in cat_feats)]

    pre_tree = ColumnTransformer([
        ("cat", _ohe(), cat_feats),
        ("num", "passthrough", num_feats)
    ])

    pre_linear = ColumnTransformer([
        ("cat", _ohe(), cat_feats),
        ("num", Pipeline(steps=[("scaler", StandardScaler())]), num_feats)
    ])

    return pre_tree, pre_linear, cat_feats, num_feats

def make_models(seed: int, for_tree_pre: ColumnTransformer, for_linear_pre: ColumnTransformer) -> Dict[str, Pipeline]:
    """Crea y configura los modelos de machine learning."""
    models: Dict[str, Pipeline] = {}

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=7, min_samples_split=2, min_samples_leaf=1,
        max_features='sqrt', class_weight='balanced', random_state=seed,
        bootstrap=True, oob_score=False
    )
    models['Random Forest'] = Pipeline([("pre", for_tree_pre), ("clf", rf)])

    lr = LogisticRegression(
        max_iter=2000, class_weight='balanced', solver='saga', random_state=seed
    )
    models['Regresión Logística'] = Pipeline([("pre", for_linear_pre), ("clf", lr)])

    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, reg_lambda=1.0, random_state=seed, eval_metric="logloss"
        )
        models['XGBoost'] = Pipeline([("pre", for_tree_pre), ("clf", xgb)])
    except Exception:
        gb = GradientBoostingClassifier(random_state=seed)
        models['Gradient Boosting'] = Pipeline([("pre", for_tree_pre), ("clf", gb)])

    try:
        from lightgbm import LGBMClassifier
        lgb = LGBMClassifier(
            n_estimators=600, max_depth=-1, num_leaves=31, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=seed
        )
        models['LightGBM'] = Pipeline([("pre", for_tree_pre), ("clf", lgb)])
    except Exception:
        hgb = HistGradientBoostingClassifier(random_state=seed)
        models['Hist Gradient Boosting'] = Pipeline([("pre", for_tree_pre), ("clf", hgb)])

    svm = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=seed)
    models['SVM'] = Pipeline([("pre", for_linear_pre), ("clf", svm)])

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation='relu', alpha=1e-4,
        learning_rate_init=1e-3, max_iter=150, random_state=seed
    )
    models['Red Neuronal'] = Pipeline([("pre", for_linear_pre), ("clf", mlp)])

    return models


def split_xy(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide el dataset en características y objetivo."""
    assert TARGET in df.columns, "Columna 'Survived' no encontrada en el dataset"

    df_engineered = engineer_features(df)

    cols = [c for c in SAFE_FEATURES if c in df_engineered.columns]
    X = basic_clean(df_engineered[cols].copy())
    y = df_engineered[TARGET].astype(int).copy()

    return train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)

@st.cache_resource
def fit_all_models(df: pd.DataFrame, seed: int) -> Tuple[Dict[str, Pipeline], pd.DataFrame, Dict[str, Any]]:
    """Entrena todos los modelos y calcula métricas."""
    X_train, X_test, y_train, y_test = split_xy(df, seed)
    pre_tree, pre_linear, cat_feats, num_feats = build_preprocessors(X_train)
    models = make_models(seed, pre_tree, pre_linear)

    fitted_models: Dict[str, Pipeline] = {}
    metrics_results: Dict[str, Any] = {}

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (name, pipe) in enumerate(models.items()):
        status_text.text(f"Entrenando {name}... ({i+1}/{len(models)})")
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics_results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "pr_auc": average_precision_score(y_test, y_proba),
            "mcc": matthews_corrcoef(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        }

        fitted_models[name] = pipe
        progress_bar.progress((i + 1) / len(models))

    status_text.text("Todos los modelos entrenados exitosamente!")

    info = {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "cat_feats": cat_feats, "num_feats": num_feats
    }

    metrics_df = pd.DataFrame(metrics_results).T.sort_values("roc_auc", ascending=False)
    return fitted_models, metrics_df, info

def render_sidebar() -> Tuple[pd.DataFrame, int]:
    """Renderiza la barra lateral y devuelve los datos configurados."""
    st.sidebar.title("Configuración")

    # Upload de datos
    st.sidebar.header("Datos")
    uploaded_file = st.sidebar.file_uploader("Subir dataset CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = engineer_features(df)
        st.sidebar.success(f"Dataset cargado: {len(df)} registros")
    else:
        df = load_data()

    # Configuración del modelo
    st.sidebar.header("Configuración del Modelo")
    seed = st.sidebar.number_input(
        "Semilla aleatoria", min_value=0, value=42, step=1,
        help="Seed para reproducibilidad de resultados"
    )

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Modelos incluidos:**
    - Random Forest
    - Regresión Logística
    - XGBoost / Gradient Boosting
    - LightGBM / Hist Gradient Boosting
    - SVM
    - Red Neuronal
    """)

    return df, seed

def eda_section(df: pd.DataFrame):
    """Sección de Análisis Exploratorio de Datos."""
    st.header("Exploración de Datos (EDA)")

    if df.empty:
        st.warning("Carga un dataset para comenzar el análisis")
        return

    df_eda = engineer_features(df)

    # Filtros interactivos
    with st.expander("Filtros Demográficos", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        filters = {}
        if "Sex" in df_eda.columns:
            filters['Sex'] = col1.multiselect("Sexo", options=sorted(df_eda["Sex"].dropna().unique()))
        if "Pclass" in df_eda.columns:
            filters['Pclass'] = col2.multiselect("Clase", options=sorted(df_eda["Pclass"].dropna().unique()))
        if "AgeGroup" in df_eda.columns:
            filters['AgeGroup'] = col3.multiselect("Grupo Edad", options=sorted(df_eda["AgeGroup"].dropna().unique()))
        if "CabinKnown" in df_eda.columns:
            filters['CabinKnown'] = col4.multiselect("Cabina Conocida", options=sorted(df_eda["CabinKnown"].dropna().unique()))

    filtered_df = df_eda.copy()
    for column, values in filters.items():
        if values:
            filtered_df = filtered_df[filtered_df[column].isin(values)]

    st.success(f"Registros filtrados: **{len(filtered_df)}** de {len(df_eda)}")

    # Métricas clave
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if TARGET in filtered_df.columns:
            survival_rate = filtered_df[TARGET].mean()
            st.metric("Tasa de Supervivencia", f"{survival_rate:.2%}")

    with col2:
        if "Fare" in filtered_df.columns:
            median_fare = filtered_df["Fare"].median()
            st.metric("Tarifa Mediana", f"${median_fare:.2f}")

    with col3:
        if "Age" in filtered_df.columns:
            median_age = filtered_df["Age"].median()
            st.metric("Edad Mediana", f"{median_age:.1f} años")

    with col4:
        if "FamilySize" in filtered_df.columns:
            avg_family = filtered_df["FamilySize"].mean()
            st.metric("Tamaño Familiar Promedio", f"{avg_family:.1f}")

    # Visualizaciones
    st.subheader("Distribuciones Clave")
    fig_col1, fig_col2 = st.columns(2)

    with fig_col1:
        if {"Sex", TARGET}.issubset(filtered_df.columns):
            sex_survival = filtered_df.groupby("Sex")[TARGET].mean().reset_index()
            st.bar_chart(sex_survival.set_index("Sex"), use_container_width=True)
            st.caption("Tasa de Supervivencia por Sexo")

    with fig_col2:
        if {"Pclass", TARGET}.issubset(filtered_df.columns):
            class_survival = filtered_df.groupby("Pclass")[TARGET].mean().reset_index()
            st.bar_chart(class_survival.set_index("Pclass"), use_container_width=True)
            st.caption("Tasa de Supervivencia por Clase")

    with st.expander("Vista Previa de los Datos"):
        st.dataframe(filtered_df.head(20), use_container_width=True)

def prediction_section(models: Dict[str, Pipeline], info: Dict[str, Any]):
    """Sección de Predicciones en Tiempo Real."""
    st.header("Predicción en Tiempo Real")

    X_train = info["X_train"]

    required_columns = [
        'Sex', 'Pclass', 'Embarked', 'Title', 'AgeGroup', 'CabinKnown',
        'Age', 'Fare', 'FarePerPerson', 'FamilySize', 'IsAlone', 'IsMother',
        'FareAboveMedian'
    ]
    for col in required_columns:
        if col not in X_train.columns:
            st.error(f"Columna requerida '{col}' no encontrada en los datos de entrenamiento")
            st.write(f"Columnas disponibles: {list(X_train.columns)}")
            return

    with st.form("passenger_form"):
        st.subheader("Información del Pasajero")

        col1, col2, col3 = st.columns(3)
        inputs = {}
        inputs['Sex'] = col1.selectbox("Sexo", options=sorted(X_train['Sex'].unique()))
        inputs['Pclass'] = col2.selectbox("Clase", options=sorted(X_train['Pclass'].unique()))
        inputs['Embarked'] = col3.selectbox("Puerto de Embarque", options=sorted(X_train['Embarked'].unique()))

        col4, col5, col6 = st.columns(3)
        inputs['Title'] = col4.selectbox("Título", options=sorted(X_train['Title'].unique()))
        inputs['AgeGroup'] = col5.selectbox("Grupo de Edad", options=sorted(X_train['AgeGroup'].unique()))
        inputs['CabinKnown'] = col6.selectbox("Cabina Conocida", options=sorted(X_train['CabinKnown'].unique()))

        col7, col8, col9 = st.columns(3)
        inputs['Age'] = col7.slider("Edad", 0, 80, int(X_train['Age'].median()))
        inputs['Fare'] = col8.slider("Tarifa", 0.0, float(X_train['Fare'].max()), float(X_train['Fare'].median()))
        inputs['FarePerPerson'] = col9.slider("Tarifa por Persona", 0.0, float(X_train['FarePerPerson'].max()),
                                              float(X_train['FarePerPerson'].median()))

        col10, col11, col12 = st.columns(3)
        inputs['FamilySize'] = col10.slider("Tamaño Familiar", 0, 11, int(X_train['FamilySize'].median()))
        inputs['IsAlone'] = col11.selectbox("Viaja Solo", options=[0, 1])
        inputs['IsMother'] = col12.selectbox("Es Madre", options=[0, 1])

        inputs['FareAboveMedian'] = st.selectbox("Tarifa sobre la Mediana", options=[0, 1])

        submitted = st.form_submit_button("Predecir Supervivencia")

    if submitted:
        row = pd.DataFrame([inputs])
        predictions: Dict[str, float] = {}

        with st.spinner("Realizando predicciones..."):
            for name, model in models.items():
                try:
                    proba = model.predict_proba(row)[:, 1][0]
                    predictions[name] = float(proba)
                except Exception as e:
                    predictions[name] = np.nan
                    st.error(f"Error con {name}: {e}")

        st.subheader("Resultados de Predicción")

        pred_df = pd.DataFrame.from_dict(predictions, orient='index',
                                         columns=['Probabilidad de Supervivencia']).sort_values(
            'Probabilidad de Supervivencia', ascending=False
        )

        st.dataframe(
            pred_df.style.format('{:.3f}').background_gradient(cmap='RdYlGn_r', vmin=0, vmax=1),
            use_container_width=True
        )

        best_model = pred_df.idxmax()[0]
        best_prob = pred_df.max()[0]
        st.success(f"**Mejor modelo**: {best_model} con {best_prob:.3f} de probabilidad")

def model_analysis_section(models: Dict[str, Pipeline], metrics_df: pd.DataFrame, info: Dict[str, Any]):
    """Sección de Análisis de Modelos."""
    st.header("Análisis de Modelos")

    st.subheader("Comparación de Métricas")
    formatted_metrics = metrics_df.style.format({
        'accuracy': '{:.3f}', 'precision': '{:.3f}', 'recall': '{:.3f}',
        'f1': '{:.3f}', 'roc_auc': '{:.3f}', 'pr_auc': '{:.3f}',
        'mcc': '{:.3f}', 'balanced_accuracy': '{:.3f}'
    }).background_gradient(cmap='YlGnBu')
    st.dataframe(formatted_metrics, use_container_width=True)

    st.subheader("Importancia de Características")
    model_choice = st.selectbox("Seleccionar modelo para análisis:", options=list(models.keys()))
    selected_model = models[model_choice]

    with st.spinner("Calculando importancia de características..."):
        try:
            result = permutation_importance(
                selected_model, info["X_test"], info["y_test"],
                n_repeats=10, random_state=42, scoring='roc_auc'
            )
            importances = pd.DataFrame({
                'feature': info["X_test"].columns,
                'importance': result.importances_mean,
                'std': result.importances_std
            }).sort_values('importance', ascending=False).head(15)

            fig, ax = plt.subplots(figsize=(10, 8))
            y_pos = np.arange(len(importances))
            ax.barh(y_pos, importances['importance'], xerr=importances['std'], align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(importances['feature'])
            ax.invert_yaxis()
            ax.set_xlabel('Importancia')
            ax.set_title(f'Importancia de Características - {model_choice}')
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error calculando importancia: {e}")

def what_if_section(models: Dict[str, Pipeline], info: Dict[str, Any]):
    """Sección de Análisis What-If."""
    st.header("Análisis What-If")

    X_test = info["X_test"]

    st.subheader("Seleccionar Caso Base")
    case_idx = st.selectbox(
        "Seleccionar caso del conjunto de prueba:",
        options=range(len(X_test)),
        format_func=lambda x: f"Caso {x}: {dict(X_test.iloc[x].items())}"
    )

    base_case = X_test.iloc[[case_idx]].copy()

    with st.expander("Ver caso base completo"):
        st.json(base_case.to_dict(orient='records')[0])

    st.subheader("Modificar Características")
    modifications = {}
    numeric_features = [col for col in base_case.columns
                        if base_case[col].dtype != 'object' and col != 'Pclass']

    for feature in numeric_features:
        current_val = float(base_case[feature].iloc[0])
        min_v = float(max(0, current_val * 0.1))
        max_v = float(current_val * 3.0) if current_val > 0 else float(1.0)
        modifications[feature] = st.slider(f"{feature}", min_value=min_v, max_value=max_v,
                                           value=current_val, step=0.1)

    modified_case = base_case.copy()
    for feature, new_value in modifications.items():
        modified_case[feature] = new_value

    if st.button("Calcular Impacto"):
        impacts = {}
        with st.spinner("Calculando impactos..."):
            for model_name, model in models.items():
                try:
                    orig_prob = model.predict_proba(base_case)[:, 1][0]
                    new_prob = model.predict_proba(modified_case)[:, 1][0]
                    impacts[model_name] = {
                        'original': orig_prob,
                        'nuevo': new_prob,
                        'diferencia': new_prob - orig_prob
                    }
                except Exception as e:
                    impacts[model_name] = {'original': np.nan, 'nuevo': np.nan, 'diferencia': np.nan}
                    st.error(f"Error con {model_name}: {e}")

        impact_df = pd.DataFrame.from_dict(impacts, orient='index')
        st.dataframe(
            impact_df.style.format('{:.3f}').background_gradient(
                subset=['diferencia'], cmap='RdYlGn', vmin=-1, vmax=1
            ),
            use_container_width=True
        )

def main():
    """Función principal de la aplicación."""
    st.markdown('<h1 class="main-header">🚢 Titanic ML Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")

    df, seed = render_sidebar()

    if df.empty or TARGET not in df.columns:
        st.warning("""
        **Por favor carga un dataset válido:**
        - Debe contener la columna 'Survived' como variable objetivo
        - Puedes subir un archivo CSV usando el panel lateral
        """)
        return

    with st.spinner("Entrenando modelos de machine learning..."):
        try:
            models, metrics, info = fit_all_models(df, seed)
        except Exception as e:
            st.error(f"Error entrenando modelos: {e}")
            return

    tab1, tab2, tab3, tab4 = st.tabs([
        "Exploración de Datos",
        "Predicciones",
        "Análisis de Modelos",
        "What-If Analysis"
    ])

    with tab1:
        eda_section(df)

    with tab2:
        prediction_section(models, info)

    with tab3:
        model_analysis_section(models, metrics, info)

    with tab4:
        what_if_section(models, info)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
