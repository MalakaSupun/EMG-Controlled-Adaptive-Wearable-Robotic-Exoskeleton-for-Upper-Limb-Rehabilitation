#
<p align="center">
    <img width="350" src='Image/Logo_01.png' alt="Logo">
</p>

<h1 align="center">
  EMG Controlled Adaptive Wearable Robotic Exoskeleton for Upper Limb Rehabilitation AI
</h1>

# Results..
The methodology employed in this research focuses on leveraging advanced Deep Learning and Machine Learning techniques to achieve accurate predictions of the Range of Motion (ROM) angle. Specifically, the study integrates Gated Recurrent Units (GRUs) and Long Short-Term Memory (LSTM) networks as the core deep learning models. These architectures are chosen for their proven ability to capture temporal dependencies and handle sequential data effectively, making them particularly suitable for time-series prediction tasks like ROM angle estimation.  

In addition to deep learning approaches, several machine learning algorithms have been utilized to complement the predictive analysis. These include the K-Nearest Neighbors (KNN) Regressor, Support Vector Regression (SVR), and Random Forest Regressor. Each of these models contributes unique strengths: KNN excels in capturing local patterns within the data, SVR provides robust predictions with its capacity to handle non-linear relationships, and Random Forest delivers strong performance through ensemble learning, reducing the risk of overfitting. 

Given that the objective of this research is to predict a continuous variable the ROM angle the study employs regression-based artificial intelligence models. The integration of both deep learning and traditional machine learning algorithms ensures a comprehensive approach to the prediction task, combining the strengths of sequential modelling with feature-focused regressors to enhance overall accuracy and reliability.
We have graded these algorithms according to the  First let’s discuss the machine learning algorithms. R2 Score, MSE and MAE results.

#

## 1. Support Vector Regression (SVR):
The results summarise the performance of the Support Vector Regression (SVR) model when predicting the target variable, based on metrics such as R² (coefficient of determination), MSE (Mean Squared Error), MAE (Mean Absolute Error), and RMSE (Root Mean Squared Error). These metrics were evaluated using a 10-fold cross-validation approach, where the dataset was split into ten subsets, and the model was trained and tested on different combinations of these subsets. Here's a detailed breakdown:
| Cross-Validation Split | R2 Score  | MSE       | MAE       | RMSE     |
|:-----------------------:|-----------|-----------|-----------|----------|
| 1                       | 0.656371  | 66.381072 | 6.840942  | 8.147458 |
| 2                       | 0.313708  | 32.001188 | 5.052802  | 5.656959 |
| 3                       | 0.685755  | 14.807428 | 3.424892  | 3.848042 |
| 4                       | 0.942447  | 7.558939  | 1.945472  | 2.749352 |
| 5                       | 0.899167  | 11.497509 | 2.797446  | 3.390798 |
| 6                       | 0.913922  | 9.490316  | 2.573259  | 3.080636 |
| 7                       | 0.905561  | 7.999321  | 2.096159  | 2.828307 |
| 8                       | 0.903377  | 8.672771  | 2.515685  | 2.944957 |
| 9                       | 0.914370  | 13.314672 | 3.272796  | 3.648928 |
| 10                      | 0.902287  | 13.368730 | 3.374741  | 3.656327 |
| **Averages**            | 0.803696  | 18.509195 | 3.389419  | 3.995176 |

The cross-validation table summarises the performance of the Support Vector Regression (SVR) model across ten different data splits. Key metrics such as R² (coefficient of determination), MSE (Mean Squared Error), MAE (Mean Absolute Error), and RMSE (Root Mean Squared Error) are reported for each fold, providing insights into the model's accuracy and reliability. On average, the SVR model achieves an R² of 0.8037, indicating that it explains approximately 80% of the variance in the target variable, with an average MSE of 18.51 and an MAE of 3.39. However, the results vary significantly across folds, with R² scores ranging from 0.31 to 0.94 and MSE values fluctuating from 7.56 to 66.38. This variability highlights the sensitivity of the model to the data distribution in each fold, suggesting that while the model generally performs well, it may require further optimisation to achieve more consistent results. The best performance is observed in fold 4, which stands out with the lowest error metrics, offering a benchmark for improvement.

**Overall Metrics:**

•	**Average R²: 0.803696**
This indicates that, on average, the SVR model explains approximately 80.37% of the variance in the target variable, suggesting a strong overall fit.

•	**Average MSE: 18.509195**
The average MSE measures the average squared differences between predicted and actual values, highlighting the overall error magnitude. A lower value is better, and while 18.51 is acceptable, there is room for improvement.

•	**Average MAE: 3.389419**
This indicates that the average absolute difference between predicted and actual values is about 3.39 units. It is a straightforward measure of prediction accuracy.

•	**Average RMSE: 3.995176**
The RMSE reflects the standard deviation of prediction errors and provides a clearer interpretation of prediction reliability. An RMSE of 3.99 means predictions typically deviate by approximately 4 units from actual values.


## 2. Random Forest Regressor (RFR):
The table presents the cross-validation performance metrics for the Random Forest Regressor (RFR) model across ten data splits. Key evaluation metrics include R² (coefficient of determination), MSE (Mean Squared Error), MAE (Mean Absolute Error), and RMSE (Root Mean Squared Error), providing a comprehensive view of the model's prediction capabilities.

| Cross-Validation Split | R2 Score  | MSE       | MAE       | RMSE     |
|:-----------------------:|-----------|-----------|-----------|----------|
| 1                       | 0.823812  | 34.035417 | 4.658643  | 5.833988 |
| 2                       | 0.697794  | 14.091627 | 2.840062  | 3.753882 |
| 3                       | 0.971930  | 1.742091  | 0.913886  | 1.319883 |
| 4                       | 0.943781  | 4.190285  | 1.382041  | 2.047018 |
| 5                       | 0.948241  | 3.883101  | 1.321665  | 1.970558 |
| 6                       | 0.893160  | 4.867509  | 1.611156  | 2.206243 |
| 7                       | 0.991379  | 1.426605  | 0.818089  | 1.194406 |
| 8                       | 0.767034  | 16.429321 | 3.215079  | 4.053310 |
| 9                       | 0.986421  | 1.053367  | 0.700370  | 1.026337 |
| 10                      | 0.987504  | 2.166532  | 1.014344  | 1.471915 |
| **Average**             | 0.901105  | 8.388585  | 1.847534  | 2.487754 |

On average, the RFR model demonstrates strong performance, with an R² score of 0.9011, meaning it explains over 90% of the variance in the target variable. The average error metrics MSE at 8.39, MAE at 1.85, and RMSE at 2.49 further underline the model's accuracy and reliability. Performance across individual folds is consistent, with R² scores ranging from 0.698 to 0.991, showcasing a robust ability to generalise across data splits.
The best performance is observed in folds 7 and 9, where the R² scores are above 0.986 and the error metrics are the lowest, particularly an MSE as low as 1.05 in fold 9. This indicates the model's exceptional accuracy in these cases. In contrast, fold 1 displays a relatively higher MSE of 34.03 and MAE of 4.66, showing that while the model generally performs well, there may be some sensitivity to the distribution of data in certain splits.

**Overall Metrics:**

•	**Average R²: 0.901105**

This indicates that the Random Forest Regressor (RFR) model explains approximately 90.11% of the variance in the target variable, demonstrating a very strong fit and reliable predictive capability. 

•	**Average MSE: 8.388585**

The Mean Squared Error of 8.39 reflects the average squared differences between predicted and actual values. This relatively low value indicates that the model maintains a high degree of accuracy, with minimal large deviations.

•	**Average MAE: 1.847534**

The Mean Absolute Error signifies that, on average, predictions are off by only 1.85 units. This low MAE highlights the model's ability to deliver highly precise predictions.

•	**Average RMSE: 2.487754**

The Root Mean Squared Error of 2.49 quantifies the standard deviation of prediction errors, showing that predictions typically deviate by approximately 2.49 units from actual values. This further supports the model's robust performance and reliability.

## 3. K-Nearest Neighbours (KNN):
The table presents the cross-validation performance metrics for the Random Forest Regressor (RFR) model across ten data splits. Key evaluation metrics include R² (coefficient of determination), MSE (Mean Squared Error), MAE (Mean Absolute Error), and RMSE (Root Mean Squared Error), providing a comprehensive view of the model's prediction capabilities. The table presents the cross-validation performance metrics for the K-Nearest Neighbours (KNN)model across ten data splits. Key evaluation metrics include R² (coefficient of determination), MSE (Mean Squared Error), MAE (Mean Absolute Error), and RMSE (Root Mean Squared Error), providing a comprehensive view of the model's prediction capabilities.
| Cross-Validation Split | R2 Score  | MSE       | MAE       | RMSE     |
|:-----------------------:|-----------|-----------|-----------|----------|
| 1                       | 0.693911  | 59.129093 | 4.430968  | 7.689544 |
| 2                       | 0.923459  | 3.569030  | 1.501328  | 1.889188 |
| 3                       | 0.979217  | 0.979315  | 0.801429  | 0.989604 |
| 4                       | 0.980338  | 2.582316  | 1.138268  | 1.606959 |
| 5                       | 0.994687  | 0.605803  | 0.558870  | 0.778334 |
| 6                       | 0.969648  | 3.346397  | 1.352539  | 1.829316 |
| 7                       | 0.987076  | 1.094700  | 0.728928  | 1.046279 |
| 8                       | 0.988542  | 1.028425  | 0.743954  | 1.014113 |
| 9                       | 0.987118  | 2.003031  | 1.216768  | 1.415285 |
| 10                      | 0.976144  | 3.263933  | 1.344803  | 1.806636 |
| **Averages**            | 0.948014  | 7.760204  | 1.381786  | 2.006526 |

**Overall Metrics:**

•	**Average R²: 0.948014**

The average R² of 0.948 suggests that the KNN model explains approximately 94.8% of the variance in the target variable. This indicates an excellent fit, and the model captures most of the underlying patterns in the data.
•	**Average MSE: 7.760204**

The Mean Squared Error (MSE) of 7.76 indicates that, on average, the squared difference between predicted and actual values is relatively low. Although the value is acceptable, there is potential for improvement, especially in reducing larger error values.
•	**Average MAE: 1.381786**

The Mean Absolute Error of 1.38 shows that, on average, the KNN model’s predictions deviate by approximately 1.38 units from the actual values. This is a good measure of prediction accuracy, indicating a relatively small error in the model's forecasts.
•	**Average RMSE: 2.006526**

The Root Mean Squared Error (RMSE) of 2.01 indicates that the predictions typically deviate by about 2 units from the actual values. This shows a strong performance, with the model providing reliable predictions with a manageable degree of error.

LSTM and GRU are the deep learning models that were chosen.

## 4. Gated Recurrent Units (GRU):
The dataset was split into training and testing sets with a 90% to 10% ratio. Out of the total available data, 114 samples were allocated for training, while the remaining 13 samples were reserved for testing and validation. 

This split ensures that the model is trained on a substantial portion of the data while still being evaluated on an independent set to gauge its generalization performance. 
The model achieved an impressive R² score of 0.9796, indicating that 97.96% of the variance in the target variable was successfully explained by the model. Furthermore, the Mean Absolute Error (MAE) was recorded at 1.10, signifying that the average absolute difference between the predicted and actual values was only 1.10 units. The Mean Squared Error (MSE) of 1.94 highlights the squared differences' minimal magnitude, while the Root Mean Squared Error (RMSE) of 1.39 underscores the model’s strong predictive accuracy, as lower RMSE values indicate better performance. 
These metrics demonstrate the model's robust learning ability and its potential to make highly reliable predictions, even when tested on unseen data. The low error values and high R² score are indicative of a well-tuned and effective model, suitable for tackling real-world regression tasks.
|Matrix |Score|
|:-----:|:-------:|
|R2 score| 0.97961025|
|MAE | 1.104778|
|MSE | 1.937539|
|RMSE | 1.391955|

**Explanation of the MSE and MAE Graphs:**
MSE Graph (Left Panel): The Mean Squared Error (MSE) graph illustrates the loss during the training and validation phases over 300 epochs. Initially, both the training and validation loss values are high, but they decrease rapidly as the model learns from the data. By around 50 epochs, the MSE values stabilize at very low levels, indicating effective learning and minimal overfitting. The closeness of the training and validation MSE curves demonstrates that the model generalizes well to unseen data.

**MAE Graph (Right Panel):** The Mean Absolute Error (MAE) graph similarly tracks the error reduction during training and validation. Both curves show a sharp decline in error during the initial epochs, eventually stabilizing at values close to zero. The smooth convergence of the training and validation MAE curves further confirms the model's robustness and its ability to handle temporal dependencies in the dataset effectively.
These graphs highlight the GRU model’s strong learning capacity and the consistency between training and validation performance, reflecting the model’s reliability in making accurate predictions.

## 5. Long Short-Term Memory (LSTM):

The dataset was divided into a 90%-10% split, with 114 samples used for training and 13 samples for testing and validation.

The Long Short-Term Memory (LSTM) algorithm exhibited exceptional performance, achieving an R² score of 0.9892, which indicates that 98.92% of the variance in the target variable was explained by the model. The Mean Absolute Error (MAE) was measured at 0.5823, and the Mean Squared Error (MSE) was 0.7019, reflecting the model's high accuracy. Furthermore, the Root Mean Squared Error (RMSE) was recorded as 0.8378, reinforcing the LSTM model's ability to make precise and reliable predictions.
|Matrix |Score|
|:-----:|:-------:|
|R2 score| 0.989230|
|MAE     | 0.582275|
|MSE     | 0.701869|
|RMSE    | 0.837776|

**Explanation of the MSE and MAE Graphs:**
**MSE Graph (Left Panel):** The MSE graph demonstrates the progression of the model’s loss over 300 epochs for both training and validation datasets. Initially, the MSE values are high due to the random initialization of weights, but they decrease rapidly as the model learns patterns from the data. By approximately 50 epochs, both training and validation loss stabilize at very low values. The validation curve closely follows the training curve, showing minimal overfitting and a strong generalization ability to unseen data.

**MAE Graph (Right Panel):** The MAE graph also displays the reduction in error during training and validation. Similar to the MSE graph, the MAE values start high and decrease significantly during the initial epochs. Both training and validation MAE values converge smoothly, stabilizing at low levels by the 50th epoch. This indicates the model's capacity to make highly accurate predictions with minimal error.
The graphs provide clear evidence of the LSTM model's excellent learning capability. The convergence of training and validation losses without significant divergence confirms the model's robustness, and the low final values of MSE and MAE highlight its high prediction accuracy.

## Final Results Analysis 
The performance of the different AI algorithms is evaluated based on their R², MAE, MSE, and RMSE scores. 
| AI Algorithm | R2 score  | MAE       | MSE       | RMSE     |
|:------------:|-----------|-----------|-----------|----------|
| SVR          | 0.803696  | 18.509195 | 3.389419  | 3.995176 |
| RFR          | 0.901105  | 8.388585  | 1.847534  | 2.487754 |
| KNN          | 0.948014  | 7.760204  | 1.381786  | 2.006526 |
| GRU          | 0.979610  | 1.104778  | 1.937539  | 1.391955 |
| LSTM         | 0.989230  | 0.582275  | 0.701869  | 0.837776 |

Support Vector Regression (SVR) achieves an R² score of 0.803696, meaning it explains about 80.37% of the variance in the target variable, with an average MAE of 3.39 units and MSE of 18.51. While the model's predictions are fairly accurate, there is room for improvement. Random Forest Regressor (RFR), on the other hand, performs better with an R² score of 0.901105, indicating that it explains 90.11% of the variance in the data. It also achieves a lower MAE of 1.85 and a much-reduced MSE of 8.39, reflecting better prediction accuracy. The K-Nearest Neighbors (KNN) model further improves performance with an impressive R² score of 0.948014, demonstrating 94.8% of the variance explained, along with an MAE of 1.38 and an MSE of 7.76, showing excellent predictive ability. The GRU and LSTM models outperform all others, with LSTM achieving the highest R² of 0.989230, meaning it explains 98.92% of the variance. Its MAE of 0.58, MSE of 0.70, and RMSE of 0.84 indicate highly accurate predictions with minimal errors. Overall, LSTM stands out as the best-performing model, followed by GRU, KNN, RFR, and SVR.

