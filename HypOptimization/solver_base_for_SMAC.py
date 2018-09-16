import random
import scoring_variation as sv

class Hyperparameter(object):
    def __init__(self, name: str):
        self.name = name

    @classmethod
    def get_random_copy(self): raise NotImplementedError

    @classmethod
    def get_nearby_copy(self, range: float): raise NotImplementedError

    @classmethod
    def get_named_value(self): raise NotImplementedError

    @classmethod
    def get_value(self): raise NotImplementedError


class CategoricalHyperparameter(Hyperparameter):
    def __init__(self, name: str, arr: [str]):
        self.arr = arr
        self.value = random.randint(0, len(arr)-1) #self.value = random.choice(self.arr)
        Hyperparameter.__init__(self, name)

    def get_random_copy(self):
        return type(self)(self.name,self.arr)

    def get_nearby_copy(self,range: float): #range=[0,1]
        new_copy = self.get_random_copy()
        if random.random() < range:
            new_copy.value=self.value
        return new_copy

    def get_named_value(self):
        return {self.name: self.arr[self.value]}

    def get_value(self):
        return self.value


class UniformIntegerHyperparameter(Hyperparameter):
    def __init__(self, name: str, first: int, second: int):
        self.first = first
        self.second = second
        self.value = random.randint(self.first, self.second)
        Hyperparameter.__init__(self, name)

    def get_random_copy(self):
        return type(self)(self.name, self.first, self.second)

    def get_nearby_copy(self, range: float):
        new_copy = self.get_random_copy()

        range= int(round((self.second - self.first) * range))
        near = random.randint(max(self.first , self.value - range), min(self.second,self.value + range))

        new_copy.value=near
        return new_copy

    def get_named_value(self):
        return {self.name: self.value}

    def get_value(self):
        return self.value


class UniformFloatHyperparameter(Hyperparameter):
    def __init__(self, name: str, first: float, second: float):
        self.first = first
        self.second = second
        self.value= random.uniform(self.first, self.second)
        Hyperparameter.__init__(self, name)

    def get_random_copy(self):
        return type(self)(self.name, self.first, self.second)

    def get_nearby_copy(self, range: float):
        new_copy = self.get_random_copy()

        range = (self.second - self.first) * range
        near = random.uniform(max(self.first, self.value - range), min(self.second, self.value + range))

        new_copy.value=near
        return new_copy

    def get_named_value(self):
        return {self.name: self.value}

    def get_value(self):
        return self.value


class Solver:
    def __init__(self, estimator, params: [Hyperparameter], scoring=None):
        self.estimator = estimator
        self.conf_space = params

        # scorer(estimator, *args) returns score calculated on estimator
        if scoring is None:
            self.scorer = sv.call_internal_scorer
        elif callable(scoring):
            self.scorer = scoring
        elif isinstance(scoring, str): #is_string
            self.scorer = sv.SCORERS[scoring]
        else:
            raise ValueError("incompatible scoring value")

    @classmethod
    def fit(self, *args):raise NotImplementedError

class Base_solver(Solver):
    def fit(self, *args):
        maxfited = self.estimator.fit(*args)
        print('maxfited')
        print(self.scorer(maxfited, *args))
        for k in range(0, 1000):
            all_params = dict()
            for i in self.conf_space:
                all_params.update(i.get_parameter())
            new_estimator = type(self.estimator)(**all_params)
            fited = new_estimator.fit(*args)
            # elif maxfited.score(*args) < fited.score(*args):
            if self.scorer(maxfited, *args) < self.scorer(fited, *args):
                # print("NEW_PARAMS")
                # print(self.scorer(fited, *args))
                maxfited = fited
        return maxfited

# classification

def decision_tree_params():
    criterion = CategoricalHyperparameter("criterion", ["gini", "entropy"])
    splitter = CategoricalHyperparameter("splitter", ["best", "random"])

    min_samples_split = UniformIntegerHyperparameter("min_samples_split", 2, 20)
    max_depth = UniformIntegerHyperparameter("max_depth", 100, 1000)
    min_samples_leaf = UniformIntegerHyperparameter("min_samples_leaf", 1, 100)

    min_impurity_decrease = UniformFloatHyperparameter("min_impurity_decrease", 0.0, 0.7)
    return [criterion, splitter, min_samples_split, max_depth, min_samples_leaf, min_impurity_decrease]


def kNN_params():
    algorithm = CategoricalHyperparameter("algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
    metric = CategoricalHyperparameter("metric", ["cityblock", "euclidean", "l1", "l2",
                                                  "manhattan", "minkowski"])

    n_neighbors = UniformIntegerHyperparameter("n_neighbors", 2, 15)
    leaf_size = UniformIntegerHyperparameter("leaf_size", 5, 50)

    return [algorithm, metric, n_neighbors, leaf_size]


def percaption_params():
    penalty = CategoricalHyperparameter("penalty", ["l1", "l2", "elasticnet"])

    max_iter = UniformIntegerHyperparameter("max_iter", 5, 1000)
    tol = UniformFloatHyperparameter("tol", 1e-6, 1e-2)
    alpha = UniformFloatHyperparameter("alpha", 1e-6, 1e-2)
    eta0 = UniformFloatHyperparameter("eta0", 1e-2, 10.0)

    return [penalty, tol, alpha, eta0, max_iter]


# clustering

def dbscan_params():
    algorithm = CategoricalHyperparameter("algorithm", ["auto", "ball_tree", "kd_tree", "brute"])

    eps = UniformFloatHyperparameter("eps", 0.1, 0.9)

    min_samples = UniformIntegerHyperparameter("min_samples", 2, 10)
    leaf_size = UniformIntegerHyperparameter("leaf_size", 5, 100)
    return [algorithm, eps, min_samples, leaf_size]


def k_means_params():
    algorithm = CategoricalHyperparameter("algorithm", ["auto", "full", "elkan"])

    tol = UniformFloatHyperparameter("tol", 1e-6, 1e-2)

    n_clusters = UniformIntegerHyperparameter("n_clusters", 2, 15)
    n_init = UniformIntegerHyperparameter("n_init", 2, 15)
    max_iter = UniformIntegerHyperparameter("max_iter", 50, 1500)
    verbose = UniformIntegerHyperparameter("verbose", 0, 10)
    return [algorithm, tol, n_clusters, n_init, max_iter, verbose]


def gaussian_mixture():
    cov_t = CategoricalHyperparameter("covariance_type", ["full", "tied", "diag", "spherical"])

    tol = UniformFloatHyperparameter("tol", 1e-6, 0.1)
    reg_c = UniformFloatHyperparameter("reg_covar", 1e-10, 0.1)

    n_com = UniformIntegerHyperparameter("n_components", 2, 15)
    max_iter = UniformIntegerHyperparameter("max_iter", 10, 1000)
    return [cov_t, tol, reg_c, n_com, max_iter]
