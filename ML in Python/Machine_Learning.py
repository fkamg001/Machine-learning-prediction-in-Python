
# First, I will start off by loading and viewing the dataset.</li>
# I will see that the dataset has a mixture of both numerical and non-numerical features, that it contains values from different ranges
# Plus it contains a number of missing entries
# I will have to preprocess the dataset to ensure the machine learning model we choose can make good predictions
# After our data is in good shape, I will do some exploratory data analysis to build our intuitions
# Finally, I will build a machine learning model that can predict if an individual's application for a credit card will be accepted

# First, loading and viewing the dataset

# Importing libraries
import pandas as pd 
# Loading dataset
df = pd.read_csv("datasets/cc_approvals.data", header=None)
# Inspect data
print(df.head())

# 2. Inspecting the data set
# The output may appear a bit confusing at its first sight, but let's try to figure out the most important features
# As one can see from our first glance at the data, the dataset has a mixture of numerical and non-numerical features
# This can be fixed with some preprocessing,
# But before let's learn about the dataset a bit more to see if there are other dataset issues that need to be fixed

# Printing summary statistics
df_description = df.describe()
print(df_description)
print("\n")

# Printing DataFrame information
df_info = df.info()
print(df_info)
print("\n")

# Inspecting missing values in the dataset
print(df.isnull().values.sum())

# 3. Handling the missing values (part i)
# I've uncovered some issues that will affect the performance of our machine learning model(s) if they go unchanged:
# The dataset contains both numeric and non-numeric data (specifically.
# Specifically, the features 2, 7, 10 and 14 contain numeric values (of types float64, float64, int64 and int64 respectively) and all the other features contain non-numeric values
# The dataset also contains values from several ranges. Some features have a value range of 0 - 28, some have a range of 2 - 67, and some have a range of 1017 - 100000
# Finally, the dataset has missing values, which it'll rectified in this task
# The missing values in the dataset are labeled with '?', which can be seen in the last cell's output
# These missing value with question marks are replaced with NaN

# Import numpy
import numpy as np 

# Inspect missing values in the dataset
print(df.isnull().values.any())
print(df.tail(17))

# Replace the '?'s with NaN
df = df.replace('?', np.NaN)

# Inspect the missing values again
print(df.tail(17))

# 4. Handling the missing values (part ii)
# I replaced all the question marks with NaNs. This is going to help us in the next missing value treatment
# An important question that gets raised here is why are we giving so much importance to missing values? Can't they be just ignored?
# Ignoring missing values can affect the performance of a machine learning model heavily
# While ignoring the missing values our machine learning model may miss out on information about the dataset that may be useful for its training.
# Then, there are many models which cannot handle missing values implicitly such as LDA
# So, to avoid this problem, i will impute the missing values with a strategy called mean imputation

# Impute the missing values with mean imputation
mean = df.mean()
df.fillna(mean, inplace=True)
# Count the number of NaNs in the dataset to verify
print(df.isnull().sum().sum())

# 5. Handling the missing values (part iii)
# I have successfully taken care of the missing values present in the numeric columns. There are still some missing values to be imputed for columns 0, 1, 3, 4, 5, 6 and 13
# All of these columns contain non-numeric data and this why the mean imputation strategy would not work here. This needs a different treatment
# I am going to impute these missing values with the most frequent values as present in the respective columns

# Iterate over each column of cc_apps
for col in df:
    # Check if the column is of object type
    if df[col].dtypes == 'object':
        # Impute with the most frequent value
        df = df.fillna(df[col].value_counts().index[0])

# Count the number of NaNs in the dataset and print the counts to verify
print(df.isnull().sum().sum())

# 6. Preprocessing the data (part i)
# The missing values are now successfully handled
# There is still some minor but essential data preprocessing needed before we proceed towards building our machine learning model
# I am going to divide these remaining preprocessing steps into three main tasks:
# Convert the non-numeric data into numeric
# Split the data into train and test sets
# Scale the feature values to a uniform range

# First, I will be converting all the non-numeric values into numeric ones
# I do this because not only it results in a faster computation but also models such as XGBoost require the data to be in a strictly numeric format
# I will do this by using a technique called label encoding

# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Instantiate LabelEncoder
le = LabelEncoder()

# Iterate over all the values of each column and extract their dtypes
for col in df:
    # Compare if the dtype is object
    if df[col].dtypes =='object':
    # Use LabelEncoder to do the numeric transformation
        df[col]=le.fit_transform(df[col])

# 7. Splitting the dataset into train and test sets
# So all the non-numeric values successfully have been converted to numeric ones
# Now, I will split our data into train set and test set to prepare our data for two different phases of machine learning modeling: training and testing
# Ideally, no information from the test data should be used to scale the training data or should be used to direct the training process of a machine learning model
# Hence, I first split the data and then apply the scaling
# Also, features such as () and () are not as important as the other features in the dataset for predicting credit card approvals
# I should drop them to design our machine learning model with the best set of features
# In Data Science literature, this is often referred to as feature selection

# Import train_test_split
from sklearn.model_selection import train_test_split 

# Drop the features 11 and 13 and convert the DataFrame to a NumPy array
df = df.drop([11, 13], axis=1)
df = dff.values

# Segregate features and labels into separate variables
X,y = df[:,0:12] , df[:,13]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                y,
                                test_size= 0.33,
                                random_state=42)

# 8. Preprocessing the data (part ii)
# The data is now split into two separate sets - train and test sets respectively
# Now, let's try to understand what these scaled values mean in the real world.

# Import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# Instantiate MinMaxScaler and use it to rescale X_train and X_test
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)

# 9. Fitting a logistic regression model to the train set
# The dataset contains more instances that correspond to "Denied" status than instances corresponding to "Approved" status.
# A question to ask is: Which model should be used? The features that are mostl correlated with each other?
# I therefore measure correlation
# They indeed are correlated for
# Let's start our machine learning modeling with a Logistic Regression model (a generalized linear model)

# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Instantiate a LogisticRegression classifier with default parameter values
logreg = LogisticRegression()

# Fit logreg to the train set
logreg = logreg.fit(rescaledX_train, y_train)

# 10. Making predictions and evaluating performance
# But how well does our model perform? </p>
# I will now evaluate our model on the test set with respect to the classification accuracy
# But I will also take a look the model's confusion matrix
# If our model is not performing well in this aspect, then it might end up approving the application that should have been approved
# The confusion matrix helps us to view our model's performance from these aspects

# Import confusion_matrix
from sklearn.metrics import confusion_matrix

# Use logreg to predict instances from the test set and store it
y_pred = logreg.predict(rescaledX_test)

# Get the accuracy score of logreg model and print it
print("Accuracy of logistic regression classifier: ", logreg.score(rescaledX_test, y_test))

# Print the confusion matrix of the logreg model
print(confusion_matrix(y_test, y_pred))

# 11. Grid searching and making the model perform better
# Our model seems to be pretty good! It was able to yield an accuracy score of almost 84%
# For the confusion matrix, the first element of the of the first row of the confusion matrix denotes the true negatives meaning the number of negative instances predicted by the model correctly
# And the last element of the second row of the confusion matrix denotes the true positives meaning the number of positive instances predicted by the model correctly
# Let's see if we can do better. One can perform a grid search of the model parameters to improve the model's ability using different hyperparameters:
# 1- tol
# 2- max_iter

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Define the grid of values for tol and max_iter
tol = [0.01,0.001,0.0001]
max_iter = [100,150,200]

# Create a dictionary where tol and max_iter are keys and the lists of their values are corresponding values
param_grid = dict(tol=tol, max_iter=max_iter)

# 12. Finding the best performing model
# I have defined the grid of hyperparameter values and converted them into a single dictionary format which expects as one of its parameters
# Now, I will begin the grid search to see which values perform best
# I will instantiate GridSearchCV() with our earlier model with all the data we have
# Instead of passing train and test sets separately, I will supply X (scaled version) and y
# I will also instruct GridSearchCV() to perform a cross-validation of five folds
# I'll end the notebook by storing the best-achieved score and the respective best parameters
# <p>While building model, I tackled some of the most widely-known preprocessing steps such as scaling, label encoding, and missing value imputation
# We finished with some machine learning to predict if a person's application for a credit card would get approved or not given some information about that person

# Instantiate GridSearchCV with the required parameters
grid_model = GridSearchCV(estimator=logreg, param_grid = param_grid, cv=5)

# Use scaler to rescale X and assign it to rescaledX
rescaledX = scaler.fit_transform(X)

# Fit data to grid_model
grid_model_result = grid_model.fit(X, y)
start_time = time.time()

# Summarize results
print("Best: %f using %s" % (grid_model_result.best_score_, grid_model_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')

# 13. A gradient boosting model
# Now I'll fit a gradient boosting (GB) model. GB is similar to random forest models
# With each iteration, the next tree fits the residual errors from the previous tree in order to improve the fit

# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Create GB model -- hyperparameters have already been searched for
gbr = GradientBoostingRegressor(max_features=4,
                                learning_rate=0.01,
                                n_estimators=200,
                                subsample=0.6,
                                random_state=42)
# Fitting gbr to the train set
gbr.fit(X_train, y_train)
print(gbr.score(X_train, y_train))
print(gbr.score(X_test, y_test))

# 14. Gradient boosting feature importances
# One can extract feature importances from gradient boosting models to understand which features are the best predictors
# This can help average out any peculiarities that may arise from one particular model
# The feature importances are stored as a numpy array in the .feature_importances_ property of the gradient boosting model
# I'll need to get the sorted indices of the feature importances, using np.argsort(), in order to make a plot
# I want the features from largest to smallest, so I will use Python's indexing to reverse the sorted importances like feat_importances[::-1]

# Extract feature importances from the fitted gradient boosting model
feature_importances = gbr.feature_importances_
# Get the indices of the largest to smallest feature importances
sorted_index = np.argsort(feature_importances)[::-1]
x = range(features.shape[1])
# Create tick labels
labels = np.array(feature_names)[sorted_index]
plt.bar(x, feature_importances[sorted_index], tick_label=labels)
# Set the tick lables to be the feature names, according to the sorted feature_idx
plt.xticks(rotation=90)
plt.show()

# Import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesClassifier
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, y)
print(model.feature_importances_)


# # Remove unimportant features (weekdays)
# X_train = X_train.iloc[:, :-4]
# X_test = X_test.iloc[:, :-4]

# 15. Building and fitting a neural net(ANN)
# Neural nets can capture complex interactions between variables, but are difficult to set up and understand
# To build our nets I'll use the keras library. This is a high-level API that allows us to quickly make neural nets, yet still exercise a lot of control over the design
# The first thing I'll do is create almost the simplest net possible -- a 3-layer net that takes our inputs and predicts a single value
# Much like the sklearn models, keras models have a .fit() method that takes arguments of (features, targets)

# Import Sequential
from keras.models import Sequential
from keras.layers import Dense

# Creating the model
model_1 = Sequential()
model_1.add(Dense(100, input_dim=rescaledX_train.shape[1], activation='relu'))
model_1.add(Dense(20, activation= 'relu'))
model_1.add(Dense(1, activation='linear'))

# Fitting the model
model_1.compile(optimizer='adam', loss='mse')
history = model_1.fit(rescaledX_train, y_train, epochs=25)

# 16. Plot losses
# Once I've fit a model, I usually check the training loss curve to make sure it's flattened out
# The history returned from model.fit() is a dictionary that has an entry, 'loss', which is the training loss
# I want to ensure this has more or less flattened out at the end of our training
# Plot the losses from the fit
plt.plot(history.history['loss'])
# Use the last loss as the title
plt.title('loss:' + str(round(history.history['loss'][-1], 6)))
plt.show()

# 17. Measuring performance of the final model
# Now that I've fit our neural net, let's check performance to see how well our model is predicting new values
# There's not a built-in .score() method like with sklearn models, so I'll use the r2_score() function from sklearn.metrics
# This calculates the R22 score given arguments (y_true, y_predicted)
# I'll also plot our predictions versus actual values again. This will yield some interesting results.
# Import r2_score
from sklearn.metrics import r2_score

# Calculate R^2 score
train_preds = model_1.predict(rescaledX_train)
test_preds = model_1.predict(rescaledX_test)
print(r2_score(train_targets, train_preds))
print(r2_score(test_targets, test_preds))

# Plot predictions vs actual
plt.scatter(train_preds, train_targets, label='train')
plt.scatter(test_preds, test_targets, label = 'test')
plt.legend()
plt.show()

# 18. Combatting overfitting with dropout
# A common problem with neural networks is they tend to overfit to training data
# What this means is the scoring metric, like R22 or accuracy, is high for the training set, but low for testing and validation sets
# The model is fitting to noise in the training data. One can avoid overfitting by using dropout


