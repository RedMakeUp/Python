from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

iris = sns.load_dataset('iris')
X_iris = iris.drop(columns=['species'])
y_iris = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, random_state=1, test_size=0.25)

model = GaussianNB()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(accuracy_score(y_test, y_predict))
