# Task 1: Simple Linear Regression
#     • Load dataset: Use the Diabetes dataset from sklearn.datasets. Select one feature (bmi) to predict the target (disease progression).
from sklearn.datasets import load_diabetes;
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression;
from sklearn.metrics import mean_squared_error, r2_score;
import pandas as pd
import matplotlib.pyplot as plt;
import seaborn as sns;
import numpy as np
dataset = load_diabetes()
def linear():
    df = pd.DataFrame(data = dataset.data , columns=dataset.feature_names);
    df['target'] = dataset.target


        # • Perform Exploratory Data Analysis (EDA):
        #     ◦ Plot scatter plot of BMI vs. Disease Progression.
        #     ◦ Check correlation.
    print(df.head(10));

    missingValue = df.isnull().sum();
    print(f"MISSING VALUES {missingValue}");

    x = df['bmi']
    y = df['target']

    plt.scatter(x=x , y=y ,);
    plt.xlabel("BMI ")
    plt.ylabel("Target (DISEASE PROGRESSION)")
    # plt.show(); 
    #in the graph it shows that the higher the BMI index the higher the disease progression

        # • Implement Simple Linear Regression using sklearn.linear_model.LinearRegression:
        #     ◦ Split data into training and testing sets.
        



    xValues = df['bmi']
    yValues = df['target'];

    XTrain , XTest , YTrain, YTest = train_test_split(xValues , yValues , test_size=0.3 , random_state=20);

    XTrainReShaped = XTrain.values.reshape(-1 , 1);
    YTrainReShaped = YTrain.values.reshape(-1 , 1);
    XTestReShaped = XTest.values.reshape(-1 ,1 );
    YTestReShaped = YTest.values.reshape(-1 ,1 );




    #     ◦ Fit the model and predict disease progression
    regressor = LinearRegression();
    regressor.fit(XTrainReShaped , YTrainReShaped);
    print(regressor.score(XTestReShaped , YTestReShaped))
        #     ◦ Plot the regression line on the scatter plot.
    predictedValues = regressor.predict(XTestReShaped);
    plt.scatter(XTestReShaped , YTestReShaped , color="blue");
    plt.plot(XTestReShaped , predictedValues , color="red");

    plt.show()


    MSE = mean_squared_error(y_true = YTestReShaped , y_pred= predictedValues);
    R2 = r2_score(y_true= YTestReShaped , y_pred=predictedValues );
    print(f"MEANSQUAREERROR {MSE}")
    print(f"R2 {R2}")


# Task 2: Multivariate Linear Regression
    # • Load dataset: Use the same Diabetes dataset, but include all 10 features to predict disease progression.
        # • Perform EDA:
        # ◦ Generate a correlation heatmap between features and the target.
        # ◦ Create pair plots for selected features vs. target.
    
def multiVariable():
    # Features
    features = dataset.feature_names

    # Create DataFrame with features
    heatmapData = pd.DataFrame(data=dataset.data, columns=features)
    heatmapData["target"] = dataset.target

    corrMatrix = heatmapData.corr()
    # plt.figure(figsize=(12, 8))
    # sns.heatmap(corrMatrix, annot=True, cmap="coolwarm", center=0)
    # plt.title("Correlation Heatmap (Diabetes Dataset)")
    # plt.show()
    #  sns.heatmap(corrMatrix , annot=True);
    # sns.pairplot(heatmapDataTarget  , hue="target");
    # plt.show();

    XValues = heatmapData.drop(columns=["target"])
    YValues = heatmapData["target"]

    XTrain, XTest, YTrain, YTest = train_test_split(
        XValues, YValues, test_size=0.3, random_state=40
    )

    model = LinearRegression()
    model.fit(XTrain, YTrain)
    predictedValues = model.predict(XTest);
    # print("Features:", features)
    print("Model score on test data:", model.score(XTest, YTest))

    plt.scatter(YTest, predictedValues, alpha=0.7)
    plt.plot([YTest.min(), YTest.max()], [YTest.min(), YTest.max()], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()

    residuals = YTest - predictedValues
    plt.scatter(predictedValues, residuals, alpha=0.7)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()

    mse = mean_squared_error(YTest, predictedValues)
    rmse = np.sqrt(mse)
    r2 = r2_score(YTest, predictedValues)

    print(" Model Performance:")
    print(f" MSE  = {mse}")
    print(f" RMSE = {rmse}")
    print(f" R²   = {r2}")

multiVariable()