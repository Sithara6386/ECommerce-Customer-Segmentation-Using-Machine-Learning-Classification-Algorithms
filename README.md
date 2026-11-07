# ECommerce-Customer-Segmentation-Using-Machine-Learning-Classification-Algorithms
a) Problem Statement: 

The objective of this project is to develop a machine-learning classification solution that predicts an e-commerce customer's segment from their purchasing behavior and transaction history. This is a multiclass classification problem. Five machine-learning algorithms are trained and evaluated using a common set of classification metrics.


b) Dataset Description: 

1) Dataset name and source (Kaggle / UCI link):-	E-Commerce Sales Analytics Dataset
https://www.kaggle.com/datasets/datascikhan/e-commerce-sales-and-customer-analytics

2) Type of problem (binary / multi-class):-	Multi Class 

3) Number of instances (minimum 500):-	130,000+

4) Number of features (minimum 12):-	46

5) Target variable and its classes	Target Variable:- Customer Segment 

6) Classes:- Consumer, Premium, VIP, Business 

7) Missing values / duplicates and how handled:-	Missing values found in 10 columns. Not in the target variable. For the selected feature columns imputed values. Checked for duplicated. None found. Code written to double check and drop duplicates if found. 

8) Preprocessing (scaling, encoding, etc.):- 	Data set is order level data with 138116. Number of Unique Customers: 24911. Performed Customer level aggregation. Aggregated customer data, Numerical transaction data and Customer behaviour and delivery experience and categorical purchase preferences. Dropped columns such as customer id’s, customer name, customer postal code etc. and other columns determined as having correlation such as customer type and customer lifetime value etc. to prevent leakage. Performed standard scaling and encoding to complete preprocessing. 

9) Train / test split ratio, stratification and random seed:

    80: 20 Stratified split,
    Training Features: (19928, 120),
    Testing Features: (4983, 120),
    Random seed: 42

10) Class distribution (full, train, test): 

Full Customer Segment Distribution:-

     Consumer:   75636
     Premium:    34746
     VIP :      13957
     Business :  13777

Training Target Distribution:-

    Consumer:   10867
    Premium:     5020
    VIP:      2029
    Business:      2012

Testing Target Distribution:- 

    Consumer:   2718
    Premium:   1255
    VIP :      507
    Business:    503



c) GitHub Repository Link:

https://github.com/Sithara6386/ECommerce-Customer-Segmentation-Using-Machine-Learning-Classification-Algorithms


d) Models used: Model Comparison Table

	ML Model Name	Accuracy	AUC	Precision	Recall	F1 Weighted	MCC
	
      Support Vector Machine 0.6623	0.8083	0.5831	0.6623	0.6189	0.4213
      Decision Tree	         0.6129	0.7079	0.5785	0.6129	0.5927	0.3434
      Logistic Regression	 0.4529	0.8085	0.5979	0.4529	0.4936	0.2596
      Gaussian Naive Bayes	 0.4337	0.6551	0.5034	0.4337	0.4586	0.1643
      KNN	                 0.5410	0.6344	0.4748	0.5410	0.4953	0.1734


e) Add your observations on the performance of each model on the chosen dataset: 

ML Model Name and Observation about model performance

1) Support Vector Machine:-
The Support Vector Machine achieved the highest accuracy (66.23%) and Weighted F1-score (61.89%). It also obtained the highest MCC (0.4213), indicating the strongest overall agreement between predicted and actual customer segments among the evaluated models. Its AUC of 0.8083 also indicates good class-separation capability. Therefore, SVM is the best-performing model based on the primary evaluation metric.

2) Decision Tree:-	
The Decision Tree achieved an accuracy of 61.29% and a Weighted F1-score of 59.27%. Its performance was relatively close to SVM, suggesting that it can capture nonlinear relationships in customer purchasing behaviour. However, its lower AUC of 0.7079 indicates comparatively weaker class-separation capability.

3) Logistic Regression:- 	
Logistic Regression achieved a Weighted F1-score of 49.36% and accuracy of 45.29%. Although its classification accuracy was relatively low, it obtained the highest AUC of 0.8085, marginally above SVM. This suggests that it can distinguish between customer segments reasonably well based on its prediction scores, even though its final class predictions are less effective.

4) Naive Bayes:-	
Gaussian Naive Bayes achieved a Weighted F1-score of 45.86% and accuracy of 43.37%. Its relatively weak performance may be due to the assumption that features are conditionally independent, which may not hold for correlated e-commerce purchasing variables

5) KNN:-	
KNN achieved a Weighted F1-score of 49.53% and accuracy of 54.10%. Its relatively low MCC (0.1734) and AUC (0.6344) indicate weaker classification quality compared with SVM and Decision Tree. High-dimensional customer features may also reduce the effectiveness of distance-based classification.

6) Overall winner for the data set:-	
The Support Vector Machine is the preferred model for this customer segmentation problem. It achieved the highest Weighted F1-score of 0.6189, accuracy of 66.23%, and MCC of 0.4213 among the five models evaluated.
Weighted F1 is used as the primary metric because the objective is to achieve good overall classification performance, while Macro F1 provides additional insight into performance across individual customer segments.
Although SVM performs best overall, its Macro F1-score of 0.4174 indicates that classification performance is not yet consistent across all four segments. Further improvements should focus on correctly identifying minority customer segments, particularly VIP and Business customers.
Overall recommendation: Select Support Vector Machine as the baseline model for further optimisation, while monitoring both Weighted F1 and Macro F1 to ensure that improvements in overall performance do not come at the expense of minority customer segments.
