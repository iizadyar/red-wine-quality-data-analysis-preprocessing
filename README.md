# Red Wine Quality Data Analysis and Preprocessing

This project analyzes the UCI Red Wine Quality Dataset and compares the effect of preprocessing on two machine learning classification models: Random Forest and K-Nearest Neighbors (KNN).

The project was developed in Google Colab.

## Dataset

The dataset used in this project is the Red Wine Quality Dataset from the UCI Machine Learning Repository.

The dataset contains physicochemical properties of red wine samples. The goal is to predict the wine quality score based on these features.

The dataset includes:

- 1,599 samples
- 11 input features
- 1 target variable

## Features

The input features are:

- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol

The target variable is:

- Quality

## Project Objective

The objective of this project is to examine how preprocessing affects the performance of different machine learning models.

The following models were compared:

1. Random Forest without preprocessing
2. Random Forest with preprocessing
3. KNN without preprocessing
4. KNN with preprocessing

## Data Preprocessing

The preprocessing pipeline includes:

- Median imputation
- Yeo-Johnson power transformation
- Standard scaling

The preprocessing was implemented using Scikit-learn `Pipeline` and `ColumnTransformer`.

## Models

### Random Forest Classifier

Random Forest is a tree-based model. It is generally not very sensitive to feature scaling or transformation.

### K-Nearest Neighbors Classifier

KNN is a distance-based model. It is sensitive to feature scaling because features with larger numerical ranges can dominate the distance calculation.

## Evaluation Method

The models were evaluated using 10-fold Stratified Cross-Validation.

The evaluation metric was accuracy.

## Results

| Model | Preprocessing | Mean Accuracy | Standard Deviation |
|---|---|---:|---:|
| Random Forest | No | 0.7123 | 0.0436 |
| Random Forest | Yes | 0.7123 | 0.0432 |
| KNN | No | 0.5016 | 0.0353 |
| KNN | Yes | 0.6173 | 0.0386 |

## Discussion

The Random Forest model showed almost no change after preprocessing. This is expected because tree-based models are usually not strongly affected by scaling or monotonic feature transformations.

The KNN model improved noticeably after preprocessing. Its accuracy increased from 0.5016 to 0.6173. This improvement occurred because KNN depends on distance calculations, and preprocessing helps place features on a more comparable scale.

## Visualizations

The notebook includes the following visualizations:

- Model accuracy comparison
- Accuracy comparison by model
- Cross-validation accuracy distribution
- Feature distributions before and after preprocessing
- Wine quality class distribution
