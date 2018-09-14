from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import RandomizedSearchCV
from sklearn import tree, neighbors, linear_model
import solver_base
import solver_SMAC

import solver_base_for_SMAC as sb

import scoring_variation as sv

def testClassification(paramsFinder, scorer):
    algos = [tree.DecisionTreeClassifier, neighbors.KNeighborsClassifier, linear_model.Perceptron]
    params = [solver_base.decision_tree_params(), solver_base.kNN_params(), solver_base.percaption_params()]
    names = ["DecisionTreeClassifier", "kNN", "Perceptron"]

    datasets = [load_breast_cancer, load_digits]
    datasets_name = ["Breast", "Digits"]

    for j in range(0, len(datasets)):
        print("\tDataset is", datasets_name[j])
        loader = datasets[j]
        X, Y = loader(return_X_y=True)
        size = len(X) // 2
        teach_X = X[0:size]
        test_X = X[size:len(Y)]
        teach_Y = Y[0:size]
        test_Y = Y[size:len(Y)]
        print("Size is", size)
        print("Testing with inserted scoring algos")

        def printData(defaultClf):
            print("teach:")
            print(scorer(defaultClf, teach_X, teach_Y))
            print("test:")
            print(scorer(defaultClf, test_X, test_Y))

        for i in range(0, len(algos)):

            print("\nTesting on", names[i])
            defaultClf = algos[i]()
            defaultClf.fit(teach_X, teach_Y)
            print("Score before search params : ")
            printData(defaultClf)

            grid_params = solver_base.get_grid_params(params[i])

            standartClf = RandomizedSearchCV(algos[i](), grid_params, scoring=scorer).fit(teach_X, teach_Y)
            print("Score after RandomSearchCV : ")
            printData(standartClf)
            ourClf = paramsFinder(algos[i](), params[i], scoring=scorer).fit(teach_X, teach_Y)
            print("Score after our solver : ")
            printData(ourClf)


testClassification(solver_base.Base_solver, sv.SCORERS['average_precision'])


def testClassificationS(paramsFinder, scorer):
    algos = [tree.DecisionTreeClassifier, neighbors.KNeighborsClassifier, linear_model.Perceptron]
    params1 = [solver_base.decision_tree_params(), solver_base.kNN_params(), solver_base.percaption_params()]
    params = [sb.decision_tree_params(), sb.kNN_params(), sb.percaption_params()]
    names = ["DecisionTreeClassifier", "kNN", "Perceptron"]

    datasets = [load_breast_cancer, load_digits]
    datasets_name = ["Breast", "Digits"]

    for j in range(0, len(datasets)):
        print("\tDataset is", datasets_name[j])
        loader = datasets[j]
        X, Y = loader(return_X_y=True)
        size = len(X) // 2
        teach_X = X[0:size]
        test_X = X[size:len(Y)]
        teach_Y = Y[0:size]
        test_Y = Y[size:len(Y)]
        print("Size is", size)
        print("Testing with inserted scoring algos")

        def printData(defaultClf):
            print("teach:")
            print(scorer(defaultClf, teach_X, teach_Y))
            print("test:")
            print(scorer(defaultClf, test_X, test_Y))

        for i in range(0, len(algos)):

            print("\nTesting on", names[i])
            defaultClf = algos[i]()
            defaultClf.fit(teach_X, teach_Y)
            print("Score before search params : ")
            printData(defaultClf)

            grid_params = solver_base.get_grid_params(params1[i])

            standartClf = RandomizedSearchCV(algos[i](), grid_params, scoring=scorer).fit(teach_X, teach_Y)
            print("Score after RandomSearchCV : ")
            printData(standartClf)
            ourClf = paramsFinder(algos[i](), params[i], scoring=scorer).fit(teach_X, teach_Y)
            print("Score after our solver : ")
            printData(ourClf)


#testClassificationS(solver_SMAC.SMAC_solver, sv.SCORERS['average_precision'])
