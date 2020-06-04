from sklearn.datasets import load_breast_cancer, load_digits, load_diabetes, load_boston
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import tree, neighbors, linear_model
from sklearn import cluster, mixture

import warnings
warnings.filterwarnings("ignore")

import solver_SMAC as sm
import solver_base_for_SMAC as sb
import scoring_variation as sv
import random

def test(algos, params, names, datasets, datasets_name, scorer):

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
            print("teach:", scorer(defaultClf, teach_X, teach_Y),"test:",scorer(defaultClf, test_X, test_Y))
            print()


        for i in range(0, len(algos)):

            print("\nTesting on", names[i])
            defaultClf = algos[i]()
            defaultClf.fit(teach_X, teach_Y)
            print("---Score before search params : ")
            printData(defaultClf)

            grid_params = sb.get_grid_params(params[i])

            randomizedClf = RandomizedSearchCV(algos[i](), grid_params, scoring=scorer).fit(teach_X, teach_Y)
            print("---Score after RandomSearchCV : ")
            printData(randomizedClf)

            reduced_params = {}
            REDUCED_VALUE = 4
            for param in grid_params.keys():
                if len(grid_params[param])> REDUCED_VALUE:
                    reduced_params[param] = random.sample(list(grid_params[param]), REDUCED_VALUE)
                else:
                    reduced_params[param] = grid_params[param]


            gridSearchClf = GridSearchCV(algos[i](), reduced_params, scoring=scorer).fit(teach_X, teach_Y)
            print("---Score after GridSearchCV : ")
            printData(gridSearchClf)

            ourRandomedClf = sb.Random_solver(algos[i](), params[i], scoring=scorer).fit(teach_X, teach_Y)
            print("---Score after our Random_solver: ")
            printData(ourRandomedClf)

            ourSMACClf = sm.SMAC_solver(algos[i](), params[i], scoring=scorer).fit(teach_X, teach_Y)
            print("---Score after our SMAC_solver : ")
            printData(ourSMACClf)

print("--testing classification algorithms")
algos = [tree.DecisionTreeClassifier, neighbors.KNeighborsClassifier, linear_model.Perceptron]
params = [sb.decision_tree_params_c(), sb.kNN_params_c(), sb.percaption_params()]
names = ["DecisionTreeClassifier", "kNN", "Perceptron"]
datasets = [load_breast_cancer]
datasets_name = ["Breast"]

print("-testing with f1_macro scorer:")
test(algos,params,names, datasets,datasets_name,sv.SCORERS['f1_macro'])

print("-testing with average_precision scorer:")
test(algos,params,names, datasets,datasets_name, sv.SCORERS['average_precision'])


print("--testing classification algorithms")
algos = [mixture.GaussianMixture]
params = [sb.gaussian_mixture()]
names = ["Gaussian_mixture"]

datasets = [load_diabetes,load_boston]
datasets_name = ["Diabetes", "Boston"]

print("-testing with f1_macro scorer:")
test(algos,params,names, datasets,datasets_name, sv.SCORERS['completeness_score'])

print("-testing with average_precision scorer:")
test(algos,params,names, datasets,datasets_name, sv.SCORERS['v_measure_score'])


