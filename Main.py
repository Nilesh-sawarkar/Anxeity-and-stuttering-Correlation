# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats


# Load Dataset
df = pd.read_csv(r"E:\Statistics\anxiety_stuttering_dataset.csv")


# Data Preprocessing
df.dropna(inplace=True)             # Removing Missing Values
df.drop_duplicates(inplace=True)    # Removing duplicate rows


# Exploratory Data Analysis: Distribution of Anxiety Scores
sns.histplot(df['Anxiety_Score'], kde=True, color='skyblue')
plt.title("Distribution of Anxiety Score")
plt.show()


# Visualizing how Anxiety scores can vary across Age Groups

sns.boxplot(x='Age_Group', y='Anxiety_Score', hue = 'Age_Group', data=df, palette='Oranges')
plt.title("Anxiety Score across Age Groups")
plt.show()


# Visualizing how Stuttering vary across Age Groups
sns.boxplot(x='Age_Group', y='Stuttering_Severity', hue = 'Age_Group', data=df, palette='Blues')
plt.title("Stuttering Severity across Age Groups")
plt.show()


# Correlation Heatmap
correlation = df.corr(numeric_only=True)
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


# Statistical Tests: Anova Test
groups = [group['Anxiety_Score'].values for name, group in df.groupby('Age_Group') if len(group) > 0]
if len(groups) >= 2:
    f_stat, p_value = stats.f_oneway(*groups)
    print(f"ANOVA F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")


# Splitting the data into High and Low Anxiety based on the median Anxiety Score
median_anxiety = df['Anxiety_Score'].median()
high_anxiety = df[df['Anxiety_Score'] >= median_anxiety]
low_anxiety = df[df['Anxiety_Score'] < median_anxiety]
ttest_result = stats.ttest_ind(high_anxiety['Stuttering_Severity'], low_anxiety['Stuttering_Severity'])
print(f"T-test: T = {ttest_result.statistic:.3f}, p = {ttest_result.pvalue:.4f}")


# Random Forest Regression
X = df[['Anxiety_Score']]       # Predicator variable
y = df['Stuttering_Severity']   # Target Variable


# Intializing the Random Forest with 100 Trees
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)
y_rf_pred = rf_model.predict(X)


# Make Predications
y_rf_pred = rf_model.predict(X)


#Evaluate the Model Performance
r2_rf = r2_score(y, y_rf_pred)
mse_rf = mean_squared_error(y, y_rf_pred)
print(f"Random Forest R²: {r2_rf:.3f}, MSE: {mse_rf:.3f}")


# Actual Vs Predicted Values
plt.scatter(X, y, alpha=0.6, label="Actual")
plt.scatter(X, y_rf_pred, color='purple', label= "Predicted", alpha=0.6)
plt.title("Random Forest Regression")
plt.xlabel("Anxiety Score")
plt.ylabel("Stuttering Severity")
plt.legend()
plt.show() 
