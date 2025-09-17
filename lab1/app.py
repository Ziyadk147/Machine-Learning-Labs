import pandas as pd 
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.model_selection import train_test_split

#Load the Titanic dataset (train.csv).
df = sns.load_dataset('titanic');
# Display the first 10 rows.
first10 = df.head(10);

print(f"first 10 rows \n{first10}");

#Check for missing values in each column.
missingValues = df.isnull().sum()
print(f"Missing Values Count \n {missingValues}");


# Fill missing values in the "Age" column with the mean age.
meanAge = df['age'].mean();
df['age'].fillna(meanAge , inplace=True);

print("Cleaned Age Column\n",df)

#Check for missing values in each column.
missingValues = df.isnull().sum()
print(f"Missing Values Count \n {missingValues}");

# Drop rows where the "Embarked" column is missing.

df = df.dropna(subset=['embarked']);
missingValues = df.isnull().sum()
print(f"Missing Values Count after removing embarked \n {missingValues}");


#Task 2 – Encoding Categorical Data
#Convert the "Sex" column into numeric (0 = Male, 1 = Female).
df['sex'] = df['sex'].map({"female": 1 , "male" : 0})

print(f"Mapped Male and Female to 0 and 1 \n{df}");

# Apply One-Hot Encoding on the "Embarked" column.
# This stores all the value including the false and true values 
df_pandas_encoded = pd.get_dummies(df, columns=['embarked'])
print(f"ENCODED\n {df_pandas_encoded}");


encoder = OneHotEncoder();
oneHot = encoder.fit_transform(df[['embarked']]);

#will return data in the following format
#(row , column) = value
#(0  , 2) = 1.0 means first row column 3 has value of True or  1 i.e [0 , 0 , 1];

#this store only the value of the non zero values
print(f"ONEHOTENCODER\n{oneHot}");


# Task 3 – Feature Scaling & Splitting
#Select features: Age, Fare, Sex, Pclass.
selectedColumns = df[['age' ,'fare', 'sex' , 'pclass'] ]
print(f"SELECTED\n{selectedColumns}" )

# Apply StandardScaler to normalize them.

scaler = StandardScaler();
model = scaler.fit(selectedColumns);
scaledData = model.transform(selectedColumns);
print(f"SCALED\n{scaledData}" )


# Split data into 80% training and 20% testing.
xValues = selectedColumns[['age' , 'sex' , 'pclass']];
yValues = selectedColumns[['fare']];

xTrain , xTest , yTrain, yTest = train_test_split(xValues , yValues , test_size=0.2 , random_state=20 )
