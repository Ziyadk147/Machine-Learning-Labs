from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier;
from sklearn import metrics;
import pandas as pd;
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris();

# Task 2: Exploratory Data Analysis (EDA)
#     • Display dataset information, shape, and statistical summary.
#     • Plot distributions of features (histograms / pairplot).
#     • Visualize correlations
df = pd.DataFrame(data=iris.data , columns=iris.feature_names);
missingValues = df.isnull().sum()
print(f"Missing Values Count \n {missingValues}");

shape = df.shape
print(f"SHAPE {shape}");



sns.pairplot(df);



sns.histplot(df.corr())
plt.show();


# Task 3: Data Preprocessing
#     • Split dataset into training and testing sets (e.g., 70% train, 30% test).


x_train , x_test , y_train , y_test = train_test_split(iris.data , iris.target , test_size=0.3 , random_state=42);


# Task 4: Build Decision Tree Classifier
    # • Train the model using DecisionTreeClassifier.

classifier = DecisionTreeClassifier(criterion="entropy");
classifier = classifier.fit(x_train , y_train);
y_prediction = classifier.predict(x_test);

# Task 5: Model Evaluation
#     • Make predictions on the test data.
#     • Evaluate using:
#         ◦ Accuracy Score
#         ◦ Confusion Matrix
#         ◦ Classification Report
print("Accuracy:",metrics.accuracy_score(y_test, y_prediction));

confusion = metrics.confusion_matrix(y_true=y_test , y_pred=y_prediction);

sns.heatmap(confusion , annot=True , fmt='d' , cmap="Blues");
plt.title("confusion wala matrix");
plt.ylabel("True Values")
plt.xlabel("Predicted Vlaues");
plt.show();

classification_report = metrics.classification_report(y_test , y_prediction);
print(f'Classification Report {classification_report}'  )


