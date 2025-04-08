# Machine Learning Project
Locating roads to target for future concrete projects.
This is a proof-of-concept project for showcasing how ML can be leveraged in different ways.

## Data Attribution
Dataset originally published by Iowa DOT GIS Team, licensed under CC BY 4.0. Modifications may have been made.

Iowa Department of Transportation - Bureau of Research & Analytics

## Results
Easy-to-read results will be posted ASAP. This is a new project.

As of right now, I'm wrestling with the tricky data set. Many data measures had missing values, and poor documentation.

### Correlation Matrix:
In an effort to consolidate the features used for predictions/targets, I'm experimenting with the Correlation 
(or Covariance) matrix. Simply, this shows whether two variables tend to change together (+), or opposite each other(-).
This analysis helps to find a subset of features, or to transform them for simplicity (like PCA).
Current strategy is to implement a composite score and PCA for dimensionality reduction.

## TL;DR
Using real DOT road data such as traffic volume, construction/resurface date, traffic size to predict the condition of the roads in the future.
This uses the simple machine learning model of Multiple Regression. More models will be added in the future.
