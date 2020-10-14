# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt

import pandas as pd
iris = pd.read_csv("data/iris.csv")
print(iris.head(10))

# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.pairplot(iris, hue='Class')
# plt.show()

class_dict = {"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2}
iris["Class"] = iris["Class"].map(class_dict)

print(iris["Class"].value_counts())

from sklearn.model_selection import train_test_split
X = iris[["SepalLength","SepalWidth","PetalLength","PetalWidth"]]
y = iris["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

print(y_train.value_counts())
print(y_test.value_counts())

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C=1e3,solver='lbfgs')
classifier.fit(X_train, y_train)

from sklearn import metrics
predict_y = classifier.predict(X_test)
print(metrics.classification_report(y_test,predict_y))

print("分类正确率: ", metrics.accuracy_score(y_test,predict_y))

# import seaborn as sns
# colorMetrics = metrics.confusion_matrix(y_test,predict_y)
# sns.heatmap(colorMetrics,annot=True,fmt='d')
#
# plt.show()

coef_df = pd.DataFrame(classifier.coef_, columns=iris.columns[0:4])
print(coef_df.round(2))

coef_df["intercept"] = classifier.intercept_
print(coef_df.round(2))