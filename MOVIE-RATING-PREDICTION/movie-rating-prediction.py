import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('Movies_Information_India.csv', encoding='latin-1')

print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nRating range: {df['Rating'].min()} - {df['Rating'].max()}")


# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean_data(df):
    df = df.copy()

    df['Year'] = pd.to_numeric(
        df['Year'].astype(str).str.extract(r'(\d{4})')[0], errors='coerce'
    )
    df['Duration'] = pd.to_numeric(
        df['Duration'].astype(str).str.extract(r'(\d+)')[0], errors='coerce'
    )
    df['Votes'] = pd.to_numeric(
        df['Votes'].astype(str).str.replace(',', ''), errors='coerce'
    )

    df = df.dropna(subset=['Rating'])
    return df

df = clean_data(df)


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(df):
    df = df.copy()

    genres = ['Drama', 'Comedy', 'Action', 'Romance', 'Thriller',
              'Crime', 'Horror', 'Biography', 'Adventure', 'Mystery']
    for g in genres:
        df[f'genre_{g}'] = df['Genre'].fillna('').str.contains(g).astype(int)

    top_directors = df['Director'].value_counts().head(50).index
    df['Director_enc'] = df['Director'].apply(
        lambda x: x if x in top_directors else 'Other'
    )
    df['Director_enc'] = LabelEncoder().fit_transform(df['Director_enc'].fillna('Unknown'))

    for col in ['Actor 1', 'Actor 2', 'Actor 3']:
        if col in df.columns:
            top = df[col].value_counts().head(100).index
            df[f'{col}_enc'] = LabelEncoder().fit_transform(
                df[col].apply(lambda x: x if x in top else 'Other').fillna('Unknown')
            )

    df['log_votes'] = np.log1p(df['Votes'].fillna(0))
    df['movie_age'] = 2024 - df['Year'].fillna(df['Year'].median())

    return df

df = build_features(df)


# ── EDA plots ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

df['Rating'].hist(bins=30, color='steelblue', edgecolor='white', ax=axes[0, 0])
axes[0, 0].set_title('Rating distribution')
axes[0, 0].set_xlabel('Rating')
axes[0, 0].set_ylabel('Count')

genre_counts = df['Genre'].str.split(', ').explode().value_counts().head(10)
genre_counts.plot(kind='bar', color='coral', edgecolor='white', ax=axes[0, 1])
axes[0, 1].set_title('Top 10 genres')
axes[0, 1].tick_params(axis='x', rotation=45)

sample = df.dropna(subset=['log_votes', 'Rating']).sample(min(500, len(df)))
axes[1, 0].scatter(sample['log_votes'], sample['Rating'], alpha=0.4, color='purple')
axes[1, 0].set_title('Log votes vs Rating')
axes[1, 0].set_xlabel('Log votes')
axes[1, 0].set_ylabel('Rating')

yearly = df.dropna(subset=['Year', 'Rating'])
yearly.groupby('Year')['Rating'].mean().loc[lambda x: x.index >= 1970].plot(
    color='green', ax=axes[1, 1]
)
axes[1, 1].set_title('Avg rating by year')
axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('Avg rating')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()


# ── Prepare X and y ───────────────────────────────────────────────────────────

feature_cols = (
    ['Duration', 'log_votes', 'movie_age', 'Director_enc']
    + [c for c in df.columns if c.startswith('genre_')]
    + [c for c in df.columns if c.endswith('_enc') and c != 'Director_enc']
)

X = df[feature_cols].copy()
y = df['Rating'].copy()

X_imputed = SimpleImputer(strategy='median').fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42
)

print(f"\nTrain: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")


# ── Train and evaluate ────────────────────────────────────────────────────────

models = {
    'Linear Regression':   LinearRegression(),
    'Ridge':               Ridge(alpha=1.0),
    'Lasso':               Lasso(alpha=0.01),
    'Random Forest':       RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    results[name] = {
        'RMSE':  round(np.sqrt(mean_squared_error(y_test, preds)), 4),
        'MAE':   round(mean_absolute_error(y_test, preds), 4),
        'R2':    round(r2_score(y_test, preds), 4),
        'CV R2': round(cross_val_score(model, X_imputed, y, cv=5, scoring='r2').mean(), 4),
    }

results_df = pd.DataFrame(results).T
print(f"\n{results_df}")

best_name = results_df['R2'].idxmax()
best_model = models[best_name]
print(f"\nBest model: {best_name} (R² = {results_df.loc[best_name, 'R2']})")


# ── Feature importance ────────────────────────────────────────────────────────

if hasattr(best_model, 'feature_importances_'):
    importances = pd.Series(best_model.feature_importances_, index=feature_cols)
    importances.sort_values().tail(15).plot(kind='barh', color='steelblue', figsize=(10, 6))
    plt.title(f'Top 15 feature importances ({best_name})')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()


# ── Actual vs predicted ───────────────────────────────────────────────────────

y_pred = best_model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.4, color='teal', edgecolors='white', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual rating')
plt.ylabel('Predicted rating')
plt.title(f'Actual vs predicted — {best_name}')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()


# ── Predict a new movie ───────────────────────────────────────────────────────

new_movie = np.zeros((1, len(feature_cols)))
idx = {f: i for i, f in enumerate(feature_cols)}

new_movie[0, idx['Duration']]    = 180
new_movie[0, idx['log_votes']]   = np.log1p(50000)
new_movie[0, idx['movie_age']]   = 8
if 'genre_Drama' in idx:
    new_movie[0, idx['genre_Drama']] = 1

predicted = best_model.predict(new_movie)[0]
print(f"\nSample prediction → Drama | 180 min | 50k votes | 2020")
print(f"Predicted rating: {predicted:.2f} / 10")