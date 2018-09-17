import random
import numpy as np
import scoring_variation as sv
import datetime

now = datetime.datetime.now


class Hyperparameter(object):
    """Интерфейс описывающий работу с пространством гипрепараметров алгоритмов

        Поля
        ---------
        name : string
            Задает название параметра, значение которого он хранит
    """
    def __init__(self, name: str):
        self.name = name

    @classmethod
    def get_random_copy(self):
        """Создать класс, аналогичного типа, но с другим случайным значением внутри"""
        raise NotImplementedError

    @classmethod
    def get_nearby_copy(self, range: float):
        """Создать класс, аналогичного типа, но со случайным значением внутри шара радиусом range
        по отнормированной метрике пространства параметра
        """
        raise NotImplementedError

    @classmethod
    def get_named_value(self):
        """Вернуть словарь, в котором есть единственный элемент {название значения : само значение}"""
        raise NotImplementedError

    @classmethod
    def get_value(self):
        """Вернуть хранящееся значение"""
        raise NotImplementedError

    def get_grid_params(self, N):
        """Вернуть список длины N элементов, равномерно распределённыx по всему пространству параметра"""
        raise NotImplementedError


class CategoricalHyperparameter(Hyperparameter):
    """Класс, хранящий в себе пространство категориальных признаков и значение из него

        NOTE: реализация метода get_nearby_copy:
                если случайно сгенерированное число от 0 до 1 попадет в интервал range, то
                выберется случайный параметр из всех
    """

    def __init__(self, name: str, arr: [str]):
        self.arr = arr
        self.value = random.randint(0, len(arr) - 1)
        Hyperparameter.__init__(self, name)

    def get_random_copy(self):
        return type(self)(self.name, self.arr)

    def get_nearby_copy(self, range: float):
        new_copy = self.get_random_copy()
        if random.random() < range:
            new_copy.value = self.value
        return new_copy

    def get_named_value(self):
        return {self.name: self.arr[self.value]}

    def get_value(self):
        return self.value

    def get_grid_param(self):
        return self.arr


class UniformIntegerHyperparameter(Hyperparameter):
    """Класс, хранящий в себе равномерное пространство целых чисел и значение из него"""

    def __init__(self, name: str, first: int, second: int):
        self.first = first
        self.second = second
        self.value = random.randint(self.first, self.second)
        Hyperparameter.__init__(self, name)

    def get_random_copy(self):
        return type(self)(self.name, self.first, self.second)

    def get_nearby_copy(self, range: float):
        new_copy = self.get_random_copy()

        range = int(round((self.second - self.first) * range))
        near = random.randint(max(self.first, self.value - range), min(self.second, self.value + range))

        new_copy.value = near
        return new_copy

    def get_named_value(self):
        return {self.name: self.value}

    def get_value(self):
        return self.value

    def get_grid_param(self):
        return range(self.first, self.second)


class UniformFloatHyperparameter(Hyperparameter):
    """Класс, хранящий в себе равномерное пространство дробных чисел и значение из него"""

    def __init__(self, name: str, first: float, second: float):
        self.first = first
        self.second = second
        self.value = random.uniform(self.first, self.second)
        Hyperparameter.__init__(self, name)

    def get_random_copy(self):
        return type(self)(self.name, self.first, self.second)

    def get_nearby_copy(self, range: float):
        new_copy = self.get_random_copy()

        range = (self.second - self.first) * range
        near = random.uniform(max(self.first, self.value - range), min(self.second, self.value + range))

        new_copy.value = near
        return new_copy

    def get_named_value(self):
        return {self.name: self.value}

    def get_value(self):
        return self.value

    def get_grid_param(self, delta=100):
        step = (self.second - self.first) / delta
        return np.arange(self.first, self.second, step)


def get_grid_params(parametres: [Hyperparameter]):
    """Функция для генерации сетки из пространства гиперпараметров"""
    ans = {}
    for par in parametres:
        ans[par.name] = par.get_grid_param()
    return ans


class Solver:
    """Интерфейс описывающий работу с алгоритмами, оптимизирующими гиперпараметры

    Параметры
    ---------
     estimator : estimator object
        Предполагается реализация scikit-learn estimator интерфейса.
        Если у estimator не определена функция ``score``, то параметр ``scoring``
        должен быть передан

    params: [Hyperparameter]
        Список объектов, отвечающих за пространства гиперпараметров, которые будут
        оптимизироваться для estimatora.
        Для каждого объекта предполагается реализация solver_base_for_SMAC.Hyperparameter интерфейса

    scoring: string, callable или None, по умолчанию: None
        Функция, оценивающая эффективность алгоритма estimator на тестовых данных,
        которая возвращает единственное число
        Если же передается строка, она определяет один из преустановленных алгоритмов
        подсчета эффективности из scoring_variation

        Если None - в качестве scoring используется метод estimator.score
    """

    def __init__(self, estimator, params: [Hyperparameter], scoring=None, time_to_evaluate=datetime.timedelta(0, 30)):
        self.estimator = estimator
        self.conf_space = params
        self.work_time = time_to_evaluate

        # scorer(estimator, *args) returns score calculated on estimator
        if scoring is None:
            self.scorer = sv.call_internal_scorer
        elif callable(scoring):
            self.scorer = scoring
        elif isinstance(scoring, str):  # is_string
            self.scorer = sv.SCORERS[scoring]
        else:
            raise ValueError("incompatible scoring value")

    @classmethod
    def fit(self, *args):
        """Запускает оптимизацию estimator, для набора параметров args"""
        raise NotImplementedError


class Random_solver(Solver):
    """Класс для поиска указанных значений параметров для estimator, реализующий интерфейс Solver
    Основной метод - fit
    """

    def fit(self, *args):
        """Запуск оптимизации estimator для набора параметров args

        Параметры
        ---------
        args : аргументы для передачи в метод estimator.fit
        """
        maxfited = self.estimator.fit(*args)
        start_time = now()
        while start_time + self.work_time > now():
            all_params = dict()
            for i in self.conf_space:
                all_params.update(i.get_random_copy().get_named_value())
            new_estimator = type(self.estimator)(**all_params)
            fited = new_estimator.fit(*args)
            if self.scorer(maxfited, *args) < self.scorer(fited, *args):
                maxfited = fited
        return maxfited


"""Ниже представлены пространства гиперпараметров для различных алгоритмов машинного обучения"""

"""Пространства алгоритмов классификации"""
def decision_tree_params_c():
    criterion = CategoricalHyperparameter("criterion", ["gini", "entropy"])
    splitter = CategoricalHyperparameter("splitter", ["best", "random"])

    min_samples_split = UniformIntegerHyperparameter("min_samples_split", 2, 20)
    max_depth = UniformIntegerHyperparameter("max_depth", 100, 1000)
    min_samples_leaf = UniformIntegerHyperparameter("min_samples_leaf", 1, 100)

    min_impurity_decrease = UniformFloatHyperparameter("min_impurity_decrease", 0.0, 0.7)
    return [criterion, splitter, min_samples_split, max_depth, min_samples_leaf, min_impurity_decrease]


def kNN_params_c():
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


"""Пространства алгоритмов кластеризации"""
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


"""Пространства алгоритмов регрессии"""
def decision_tree_params_r():
    criterion = CategoricalHyperparameter("criterion", ["mse", "friedman_mse", "mae"])
    splitter = CategoricalHyperparameter("splitter", ["best", "random"])

    min_samples_split = UniformIntegerHyperparameter("min_samples_split", 2, 20)
    max_depth = UniformIntegerHyperparameter("max_depth", 100, 1000)
    min_samples_leaf = UniformIntegerHyperparameter("min_samples_leaf", 1, 100)

    min_impurity_decrease = UniformFloatHyperparameter("min_impurity_decrease", 0.0, 0.7)
    return [criterion, splitter, min_samples_split, max_depth, min_samples_leaf, min_impurity_decrease]


def kNN_params_r():
    algorithm = CategoricalHyperparameter("algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
    metric = CategoricalHyperparameter("metric", ["cityblock", "euclidean", "l1", "l2", "manhattan", "minkowski"])

    n_neighbors = UniformIntegerHyperparameter("n_neighbors", 2, 15)
    leaf_size = UniformIntegerHyperparameter("leaf_size", 5, 50)
    return [algorithm, metric, n_neighbors, leaf_size]


def linear_regression_params():
    fit_intercept = UniformIntegerHyperparameter("fit_intercept", 0, 1)
    return [fit_intercept]


def elastic_net_params():
    selection = CategoricalHyperparameter("selection", ["cyclic", "random"])

    l1_ratio = UniformFloatHyperparameter("l1_ratio", 0.0, 1.0)
    tol = UniformFloatHyperparameter("tol", 1e-6, 1e-2)

    fit_intercept = UniformIntegerHyperparameter("fit_intercept", 0, 1)
    max_iter = UniformIntegerHyperparameter("max_iter", 50, 500)
    return [selection, l1_ratio, tol, fit_intercept, max_iter]


def lasso_params():
    selection = CategoricalHyperparameter("selection", ["cyclic", "random"])

    tol = UniformFloatHyperparameter("tol", 1e-6, 1e-2)

    fit_intercept = UniformIntegerHyperparameter("fit_intercept", 0, 1)
    max_iter = UniformIntegerHyperparameter("max_iter", 50, 500)
    return [selection, tol, fit_intercept, max_iter]


def ridge_params():
    solver = CategoricalHyperparameter("solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"])

    tol = UniformFloatHyperparameter("tol", 1e-6, 1e-2)

    fit_intercept = UniformIntegerHyperparameter("fit_intercept", 0, 1)
    max_iter = UniformIntegerHyperparameter("max_iter", 50, 500)
    return [solver, tol, fit_intercept, max_iter]


def MLP_params():
    activation = CategoricalHyperparameter("activation", ["identity", "logistic", "tanh", "relu"])
    solver = CategoricalHyperparameter("solver", ["lbfgs", "sgd", "adam"])

    tol = UniformFloatHyperparameter("tol", 1e-6, 1e-2)
    alpha = UniformFloatHyperparameter("alpha", 1e-6, 1e-2)

    max_iter = UniformIntegerHyperparameter("max_iter", 50, 500)
    return [activation, solver, tol, alpha, max_iter]


def SGD_params():
    loss = CategoricalHyperparameter("loss",
                                     ["squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"])
    penalty = CategoricalHyperparameter("penalty", ["l1", "l2", "elasticnet"])

    alpha = UniformFloatHyperparameter("alpha", 1e-6, 1e-2)
    l1_ratio = UniformFloatHyperparameter("l1_ratio", 0.0, 1.0)
    tol = UniformFloatHyperparameter("tol", 1e-6, 1e-2)
    power_t = UniformFloatHyperparameter("power_t", 0.1, 1.0)

    max_iter = UniformIntegerHyperparameter("max_iter", 5, 1500)
    return [loss, penalty, alpha, l1_ratio, tol, power_t, max_iter]
