import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

### Modelo final: Random Forest

df = pd.read_csv('../data/processed.csv')
features = df.columns.drop('Survived')

Y = df["Survived"]
X = df.drop(columns=["Survived"])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
b_p = [100, 7, 2, 2, 'sqrt']

model = RandomForestClassifier(
    n_estimators=b_p[0],
    max_depth=b_p[1],
    min_samples_split=b_p[2],
    min_samples_leaf=b_p[3],
    max_features=b_p[4],
    random_state=42)

model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1] 