# ML-algorithms
#  LINEAR REGRESSION      ######################################
A linear regression is a good tool for quick predictive analysis: 
For Example, we want to find Employee yearly income Our Yearly income is Dependent Variable or test set
and employee work experience, employee age, employee occupation,employee last year income etc and these are our Indepent variable.
we need to build a linear regression, which predicts the line of best fit between them and can help conclude whether or not these two factors have a positive or negative relationship. 
This help us to find employee yearly income is increase or decrease.
In this linear regression the formula of the equation is Y = a + bX. Here X is INDEPENDENT VARIABLE(TRAIN OR EXPLANATORY) and X is DEPENDENT VARIABLE(TEST)
We want to plot how much of the variation in the y variable (in this case test scores) can be explained by variation in the x variable.

HOW WE FOUND THAT DATA IS MOST SUITABLE FOR LINEAR REGRESSION?????
Count data with higher means tend to be normally distributed and you can often use OLS(Ordinary Least Squares regression). However, count data with smaller means can be skewed, and linear regression might have a hard time fitting these data. So if data is not too much in right or left skewed then we use linear regression for that data.
ORDINARY LEAST SQUARES REGRESSION : USE THIS OLS METHOD TO FIND OUT OBSERVED VALUES AND PREDICT VALUES IN LINEAR REGRESSION.

# RANDOM FOREST ALGORITHM
Random Forest is a popular machine learning algorithm that belongs to the supervised learning technique. It can be used for both Classification and Regression problems in ML. It is based on the concept of ensemble learning, which is a process of combining multiple classifiers to solve a complex problem and to improve the performance of the model.

As the name suggests, "Random Forest is a classifier that contains a number of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset." Instead of relying on one decision tree, the random forest takes the prediction from each tree and based on the majority votes of predictions, and it predicts the final output.

### The greater number of trees in the forest leads to higher accuracy and prevents the problem of overfitting.
### To better understand the Random Forest Algorithm, you should have knowledge of the Decision Tree Algorithm.

# Assumptions for Random Forest
Since the random forest combines multiple trees to predict the class of the dataset, it is possible that some decision trees may predict the correct output, while others may not. But together, all the trees predict the correct output. Therefore, below are two assumptions for a better Random forest classifier:

There should be some actual values in the feature variable of the dataset so that the classifier can predict accurate results rather than a guessed result.
The predictions from each tree must have very low correlations.

# Why use Random Forest?
Below are some points that explain why we should use the Random Forest algorithm:

It takes less training time as compared to other algorithms.
It predicts output with high accuracy, even for the large dataset it runs efficiently.
It can also maintain accuracy when a large proportion of data is missing.

# How does Random Forest algorithm work?
Random Forest works in two-phase first is to create the random forest by combining N decision tree, and second is to make predictions for each tree created in the first phase.

Step-1: Select random K data points from the training set.

Step-2: Build the decision trees associated with the selected data points (Subsets).

Step-3: Choose the number N for decision trees that you want to build.

Step-4: Repeat Step 1 & 2.

Step-5: For new data points, find the predictions of each decision tree, and assign the new data points to the category that wins the majority votes.

# Applications of Random Forest
There are mainly four sectors where Random forest mostly used:

Banking: Banking sector mostly uses this algorithm for the identification of loan risk.

Medicine: With the help of this algorithm, disease trends and risks of the disease can be identified.

Land Use: We can identify the areas of similar land use by this algorithm.

Marketing: Marketing trends can be identified using this algorithm.

# Advantages of Random Forest
Random Forest is capable of performing both Classification and Regression tasks.
It is capable of handling large datasets with high dimensionality.
It enhances the accuracy of the model and prevents the overfitting issue.

# Disadvantages of Random Forest
Although random forest can be used for both classification and regression tasks, it is not more suitable for Regression tasks.



