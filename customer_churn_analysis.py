import pandas as pd
import matplotlib.pyplot as plt

# Load customer data
df = pd.read_csv("customer_churn.csv")

# Display data
print("CUSTOMER CHURN DATA")
print(df)

# Data information
print("\nDATA INFORMATION")
df.info()

# Summary statistics
print("\nSUMMARY STATISTICS")
print(df.describe())

# Average monthly charge
average_charge = df["Monthly_Charge"].mean()
print(f"\nAverage Monthly Charge: {average_charge:.2f}")

# Average customer age
average_age = df["Age"].mean()
print(f"\nAverage Customer Age: {average_age:.2f}")

# Total customers who left
churned = df[df["Churn"] == "Yes"]
print("\nCUSTOMERS WHO LEFT")
print(churned)

print(f"\nTotal Customers Who Left: {len(churned)}")

# Average support calls
average_calls = df["Support_Calls"].mean()
print(f"\nAverage Support Calls: {average_calls:.2f}")

# Bar chart
churn_counts = df["Churn"].value_counts()
churn_counts.plot(kind="bar")

plt.xlabel("Churn Status")
plt.ylabel("Number of Customers")
plt.title("Customer Churn Analysis")
plt.tight_layout()
plt.show()

# W04: Training and Testing Sets

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Predict monthly charge using subscription length
X = df[["Subscription_Months"]]
y = df["Monthly_Charge"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTRAINING SET")
print(X_train)

print("\nTESTING SET")
print(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

print("\nTEST PREDICTIONS")
print(y_pred)

# Calculate test score
test_score = r2_score(y_test, y_pred)

print(f"\nTEST SET R2 SCORE: {test_score:.2f}")