import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Loading the Dataset
df = pd.read_csv("yield_df.csv")
print(df.head())
print(df.info())

# Handling missing values
df.dropna(inplace=True)

# Encode Categorical Columns
le_area = LabelEncoder()
le_item = LabelEncoder()
df['Area_original'] = df['Area']
df['Area'] = le_area.fit_transform(df['Area'])
df['Item'] = le_item.fit_transform(df['Item'])

# Feature Selection
x = df[['Area', 'Item', 'Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']]
y = df['hg/ha_yield']

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model Selection and Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

# Prediction and Evaluation
y_pred = model.predict(X_test)

# Evaluation Metrics
print("R2 Score:", r2_score(Y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(Y_test, y_pred)))

# Visualization
plt.figure(figsize=(10, 5))
plt.scatter(Y_test, y_pred, alpha=0.5, color='green')
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual Crop Yield vs Predicted Crop Yeild")
plt.grid(True)
plt.show()

# Interface
def get_crop_by_place(place_name, df):
    place_data = df[df['Area_original'].str.lower() == place_name.lower()]

    if place_data.empty:
        print("No Data found.....")

    crop_yield = place_data.groupby('Item')['hg/ha_yield'].mean().sort_values(ascending = False)
    top_crop = crop_yield.idxmax()
    top_crop = le_item.inverse_transform([top_crop])[0]
    top_yield = crop_yield.max()

    return f"Top crop in {place_name} is '{top_crop}' with an average yield of {top_yield:.2f} hg/ha."

def show_result():
    place = entry.get()
    if not place:
        messagebox.showwarning("Input Error", "Please enter a place name.")
        return
    result = get_crop_by_place(place, df)
    output_label.config(text=result)

# Initialize main window
root = tk.Tk()
root.title("Crop Yield Finder")
root.geometry("400x200")
root.resizable(False, False)

# GUI Widgets
tk.Label(root, text="Enter the name of the place:", font=("Arial", 12)).pack(pady=10)
entry = tk.Entry(root, width=30, font=("Arial", 12))
entry.pack()

tk.Button(root, text="Get Top Crop", command=show_result, font=("Arial", 12), bg="green", fg="white").pack(pady=10)

output_label = tk.Label(root, text="", wraplength=350, font=("Arial", 11), fg="blue")
output_label.pack(pady=5)

# Run the GUI loop
root.mainloop()