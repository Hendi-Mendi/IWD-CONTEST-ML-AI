import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier




data = pd.read_csv('/content/data.csv')
# Data cleaning
data = data.drop("Patient_Number", axis=1)
missing_values = data.isnull().sum()
basic_stats = data.describe()

# Pairplot
pairplot_fig = sns.pairplot(data.sample(200, random_state=42), hue="Blood_Pressure_Abnormality")


#Correlation heatmap
plt.figure(figsize=(14,10))
corr_matrix = data.corr()
heatmap_fig = sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()





#TRAINING AND SAVING THE MODEL


# Selected features after plotting the pairplot and correlation heatmap
selected_features = ["Level_of_Hemoglobin","Genetic_Pedigree_Coefficient","Physical_activity","Chronic_kidney_disease","Adrenal_and_thyroid_disorders","BMI","Sex"]
x = data[selected_features]
y = data["Blood_Pressure_Abnormality"]


x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(criterion='gini' ,max_depth=None , min_samples_split=2)
model.fit(x_train, y_train)


y_pred = model.predict(x_test)
print(y_pred)


accuracy = model.score(x_test, y_test)
print(f"The accuracy of the model is {accuracy}")


with open("BP_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("The model is trained and saved as BP_model")

