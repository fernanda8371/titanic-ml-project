# Data Directory

Este directorio contiene los datasets utilizados en el proyecto **Titanic ML Project**.  
Cada archivo corresponde a una etapa distinta del *pipeline* de datos.  

---

## Archivos disponibles

### 1. `Titanic-Dataset.csv`  
- **Descripción**: Dataset limpio del Titanic, con variables originales procesadas (sin duplicados ni valores faltantes críticos).  
- **Contenido**:  
  - Variables originales del dataset (PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked).  
  - Valores nulos imputados en *Age* y *Embarked*.  
  - La columna *Cabin* fue transformada en binaria (*CabinKnown*).  
- **Uso**: Punto de partida para exploración (EDA) y validación de imputaciones.  

---

### 2. `processed.csv`  
- **Descripción**: Dataset con **features derivadas** creadas durante la ingeniería de características.  
- **Contenido**:  
  - Features adicionales:  
    - `Title` (título honorífico)  
    - `FamilySize`  
    - `IsAlone`  
    - `AgeGroup`  
    - `FarePerPerson`  
    - `IsMother`  
    - `FareAboveMedian`  
    - `FamilySurvivalID`  
    - `LastName`  
    - `CabinKnown`  
  - Variables categóricas codificadas (ej. *Pclass*, *Embarked*, *Sex*).  
- **Uso**: Entrenamiento y comparación de modelos en notebooks (`03_Modeling.ipynb`).  

---

### 3. `dashboard.csv`  
- **Descripción**: Dataset final integrado, con todas las features derivadas y datos listos para visualización.  
- **Contenido**:  
  - Combina las columnas originales limpias con las nuevas *features*.  
  - Optimizado para visualización en el **dashboard interactivo**.  
- **Uso**: Exclusivo para alimentar `dashboard/app.py`.  

---

## Notas importantes
- El dataset a usar en el **dashboard** es únicamente `dashboard.csv`.  
- Para entrenamiento reproducible de modelos usar **`processed.csv`**.  
- Para exploración o validación de imputaciones se recomienda **`Titanic-Dataset.csv`**.  

---

## Fuente original
El dataset proviene de la competencia pública de Kaggle:  
[Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)  

