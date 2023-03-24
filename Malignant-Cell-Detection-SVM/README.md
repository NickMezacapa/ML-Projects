# An SVM-based model for cell classification as benign or malignant

This project contains an SVM-based model for cell classification as benign or malignant. The model is trained on the [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).<br>
<br>

[Support Vector Machines](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47) (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection. The advantages of support vector machines are:<br>
* Effective in high dimensional spaces.<br>
* Still effective in cases where number of dimensions is greater than the number of samples.<br>
* Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.<br>
* Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.<br>
<br>

The disadvantages of support vector machines include:<br>
* If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.<br>
* SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).<br>
<br>

By using an SVM model, we can successfully classify cells as benign or malignant with an accuracy of 98.2%, proving that SVMs are a powerful tool for classification problems.<br>

