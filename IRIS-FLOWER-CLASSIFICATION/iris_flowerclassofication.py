import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("IRIS.csv")

print(df.shape)
print(df.head())
print(df['species'].value_counts())


feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
species_list = df['species'].unique()
colors = {s: c for s, c in zip(species_list, ['steelblue', 'coral', 'green'])}


# how each feature looks for each species
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, feat in zip(axes.flatten(), feature_cols):
    for sp, color in colors.items():
        data = df[df['species'] == sp]
        ax.hist(data[feat], bins=15, alpha=0.6, color=color, label=sp, edgecolor='white')
    ax.set_title(feat)
    ax.set_xlabel(feat)
    ax.set_ylabel('count')
    ax.legend()

plt.tight_layout()
plt.savefig('eda.png', dpi=150, bbox_inches='tight')
plt.show()


# sepal vs petal — are the species visually separable?
plt.figure(figsize=(8, 5))
for sp, color in colors.items():
    data = df[df['species'] == sp]
    plt.scatter(data['sepal_length'], data['petal_length'],
                label=sp, color=color, alpha=0.7, edgecolors='white')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.title('sepal vs petal length')
plt.legend()
plt.tight_layout()
plt.savefig('scatter.png', dpi=150, bbox_inches='tight')
plt.show()


X = df[feature_cols]
y = LabelEncoder().fit_transform(df['species'])
label_names = sorted(df['species'].unique())

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=5),
    'SVM':                 SVC(kernel='rbf', random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cv  = cross_val_score(model, X, y, cv=5).mean()
    results[name] = {'Accuracy': round(acc, 4), 'CV Accuracy': round(cv, 4)}
    print(f'\n{name}  —  acc: {acc:.4f}  cv: {cv:.4f}')
    print(classification_report(y_test, preds, target_names=label_names))

results_df = pd.DataFrame(results).T
print(results_df)

best_name = results_df['CV Accuracy'].idxmax()
best_model = models[best_name]
print(f'\nbest model: {best_name}')


cm = confusion_matrix(y_test, best_model.predict(X_test))
fig, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay(cm, display_labels=label_names).plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'confusion matrix — {best_name}')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()


if hasattr(best_model, 'feature_importances_'):
    pd.Series(best_model.feature_importances_, index=feature_cols).sort_values().plot(
        kind='barh', color='steelblue', figsize=(8, 4)
    )
    plt.title('feature importances')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()


sample = np.array([[5.1, 3.5, 1.4, 0.2]])
pred = best_model.predict(sample)[0]
print(f'\nsample flower → {label_names[pred]}')