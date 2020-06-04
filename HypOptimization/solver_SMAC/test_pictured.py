from sklearn.datasets import load_iris
import random
from sklearn import tree
import solver_base_for_SMAC as sb
import solver_SMAC as sm
import matplotlib.pyplot as plt
import numpy

algo = tree.DecisionTreeClassifier
hypparams = sb.decision_tree_params_c()

def testIris():

    dataset = load_iris(return_X_y=True)
    X, y = dataset
    size = len(X)

    testDataIndices = random.sample(range(size), size // 2)
    teachDataIndicies = list(set(range(size)) - set(testDataIndices))

    X_test, y_test = [], []
    X_teach, y_teach = [], []

    for i in testDataIndices:
        X_test += [X[i]]
        y_test += [y[i]]

    for i in teachDataIndicies:
        X_teach += [X[i]]
        y_teach += [y[i]]

    algo = tree.DecisionTreeClassifier
    hypparams = sb.decision_tree_params_c()

    defaultClf = tree.DecisionTreeClassifier().fit(X_teach, y_teach)

    SMAC = sm.SMAC_solver(algo(), hypparams).fit(X_teach, y_teach)
    randomSearch = sb.Random_solver(algo(), hypparams).fit(X_teach, y_teach)
    optiNames = ["Реальные данные","По умолчанию","Random_solver","SMAC_solver"]
    optimizedY = [y_test, defaultClf.predict(X_test),randomSearch.predict(X_test),SMAC.predict(X_test)]
    X_to_show = numpy.asarray(X_test)
    for i in range(4):
        plt.subplot(1,4,i+1)
        plt.title(optiNames[i])
        plt.scatter(X_to_show[:, 0], X_to_show[:, 1], c=optimizedY[i], cmap=plt.cm.Set1, edgecolor='k')


testIris()