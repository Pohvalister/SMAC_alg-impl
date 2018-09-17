import solver_base_for_SMAC as sb
import random as rng
import datetime

from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict

now = datetime.datetime.now
TIME_TO_WORK2 = datetime.timedelta(0, 0, 0, 0, 5)
TIME_TO_WORK = datetime.timedelta(0, 20)

""" Здесь и далее:
    Конфигурация для объекта - набор параметров, которые передаются при его создании
    Аргументы для объекта - набор параметров, которые передаются в его метод fit (если существует) 
"""


class ArgumentSpaceHandler:
    """ Класс реализующий работу над пространством аргументов

        Параметры
        ---------
        args : tuple([object])
            Аргументы, надо которыми производятся методы класса

        Поля
        ---------
        listed_args : [[object]]
            Отвечает за хранение переданных аргументов
    """

    def __init__(self, *args):
        listed_args = []
        for data in list(args):
            listed_args += [list(data)]
        self.listed_args = listed_args

    def give_arg_space(self, template):
        """По заданному шаблону [number] возвращает список - подпространство, listed_args
            в которое включаются только позиции с номерами из template
        """
        answer = []
        for arg in self.listed_args:
            data = []
            for place in template:
                data += [arg[place]]
            answer += [data]
        return answer

    def give_full_template(self):
        """Возвращает шаблон [number], покрывающий все пространство аргументов"""
        answ = []
        for i in range(len(self.listed_args[0])):
            answ += [i]
        return answ


class AlgorithmTrialsTracker:
    """ Класс реализующий работу со всеми запусками алгоритма

        Поля
        ---------
        runs_list : [(Hyperparameter,number)]
            Хранит данные о конфигурации и их средней эффективности

        runs_info : {Hyperparameter : {[number] : (number, number)}}
            Для каждой конфигуации и списка аргументов хранит количество их запусков
            и среднюю эффективность
    """

    def __init__(self):
        self.runs_list = []
        self.runs_info = defaultdict(dict)

    def get_instance_for_diff(self, conf_not_runned, conf_runned):
        """Для двух конфигураций возвращает подпространство аргументов таких, что уже были запуски
         estimator для 2 ой комбинации (конфигурация, аргументы), но не было для 1 ой
        """
        inst_conf_nR = set(self.runs_info[tuple(conf_not_runned)].keys())
        inst_conf_R = set(self.runs_info[tuple(conf_runned)].keys())

        diff_inst_runs = list(inst_conf_R - inst_conf_nR)
        return diff_inst_runs

    def get_instance_for_equl(self, conf_runned1, conf_runned2):
        """Для двух конфигураций возвращает подпространство аргументов таких, что уже были запуски
        estimator для 2 ой комбинации (конфигурация, аргументы), и были для 1 ой
        """
        inst_conf_R1 = list(self.runs_info[tuple(conf_runned1)].keys())
        inst_conf_R2 = list(self.runs_info[tuple(conf_runned2)].keys())

        equl_inst_runs = set(inst_conf_R1).intersection(inst_conf_R2)
        return equl_inst_runs

    def transform(self, c):
        """Для конфигурации переводит её список значений : [Hyperparameter] -> [values]"""
        params = []
        for param in c:
            params += [param.get_value()]
        return params

    def get_N_random_trials(self, N):
        """Возвращает списоки : [значения конфигурации], [эффективность], такой что в нем
        содержится N случайно выбранных (с повторениями) значений из всех запусков
        """
        X, y = [], []
        for i in range(N):
            (conf, perf) = rng.choice(self.runs_list)
            X += [self.transform(conf)]
            y += [perf]
        return X, y

    def check_confing_runs(self, conf):
        """Для конфигурации conf возварщает количество запусков алгоритма для неё"""
        return len(self.runs_info[tuple(conf)])

    def summarize_performance(self, conf, args_list):
        """Для конфигурации и списка параметров выводит суммарную эффективность запуска алгоритма на них"""
        sum = 0
        for args in args_list:
            (count, perf) = self.runs_info[tuple(conf)][tuple(args)]
            sum += perf
        return sum

    def get_conf_list(self):
        return self.runs_list


class SMAC_solver(sb.Solver):
    """ Класс для поиска указанных значений параметров для estimator.
    Основные методы - fit, predict

    Параметры estimatora передаваемы в эти методы оптимизируются с помощью
    реализации алгоритма SMAC (Sequential Model-based Algorithm Configuration)

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

    Поля
    ---------
    initial_conf: [Hyperparameter]
        Хранит изначальную, переданное пространство конфигураций параметров

    algo_trials: AlgorithmTrialsTracker
        Хранит в себе информацию о всех запусках estimator на конфигурациях и подпространствах аргументов

    args_keeper: ArgumentSpaceHandler
        Хранит в себе информацию о всех переданных аргументах для проверок estimator

    Пример
    ---------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn import neighbors
    >>> import solver_SMAC as sm
    >>> import solver_base_for_SMAC as sb
    >>> import scoring_variation as sv
    >>> algo = neighbors.KNeighborsClassifier
    >>> params = sb.kNN_params()
    >>> dataset = load_breast_cancer
    >>> scorer = sv.SCORERS['average_precision']
    >>> X, y = dataset(return_X_y=True)
    >>> clf = sm.SMAC_solver(algo,params,scorer).fit(X[0,len(X)-1],y[0,len(X)-1])
    >>> print(scorer(clf, X, y))

    """

    def fit(self, *args):
        """ Запускает оптимизацию estimator, для набора параметров args

        Параметры
        ---------

        args : набор агрументов для метода estimator.fit, каждый из которых представлен в виде списка
    одинаковой длины

            NOTE : Важно, что для любого подпространства номеров, если элементы с этим номером убрать из
            списков в args, он останется валидным для передачи в метод estimator.fit
                Например, SMAC_solver подходит для обтимизации алгоритмов обучения, так как если
                в качестве args передать наборы тестов для обучения, то любая выдержка из них останется
                наботом тестов для обучения:

                (X : [[1,2],[2,3],[3,4]], y : [1, 2, 3]) - подходит для обучения
                (X': [[1,2],[3,4]],       y': [1,3]    ) - подходит для обучения

        Возвращает
        ---------
        estimator object - экземпляр estimator с оптимизированной конфигурацией
        """
        self.initial_conf = self.conf_space
        self.algo_trials = AlgorithmTrialsTracker()
        self.args_keeper = ArgumentSpaceHandler(*args)

        def exec_run(self, configuration, args_template):
            """Для заданной конфигурации и шаблона аргументов производит запуск
            алгоритма estimator (метод estimator.fit).

            Параметры
            ---------
            configuration : [Hyperparameters]
                Конфигурация

            args_template : [number]
                Шаблон аргументов

            Возвращает - number - эффективность полученного алгоритма, выведенную с помощью
            функции scorer
            """
            params = dict()
            for param in configuration:
                params.update(param.get_named_value())

            configured_estimator = type(self.estimator)(**params)

            real_args = self.args_keeper.give_arg_space(args_template)

            fited = configured_estimator.fit(*real_args)
            performance = self.scorer(fited, *real_args)

            self.algo_trials.runs_list += [(configuration, performance)]

            conf_as_key = tuple(configuration)
            args_as_key = tuple(args_template)

            if conf_as_key not in self.algo_trials.runs_info:
                self.algo_trials.runs_info[conf_as_key] = {}

            if args_as_key not in self.algo_trials.runs_info[conf_as_key]:
                self.algo_trials.runs_info[conf_as_key][args_as_key] = (0, 0)

            (count, perf) = self.algo_trials.runs_info[conf_as_key][args_as_key]
            self.algo_trials.runs_info[conf_as_key][args_as_key] = (
            count + 1, (perf * count + performance) / (count + 1))

            return performance

        def initialize():
            """Еденичный запуск алгоритма на случайно заданной конфигурации

            Возвращает - [Hyperparemeter] конфигурацию алгоритма
            """
            t_args = self.args_keeper.give_full_template()
            rng_conf = []
            for param in self.initial_conf:
                rng_conf += [param.get_random_copy()]
            exec_run(self, rng_conf, t_args)
            return rng_conf

        def fitModel():
            """Создает модель для последующих предсказаний эффективности конфигураций для estimator.
            В качестве модели используется алгоритм машинного обучения Random forest, и его реализация из
            sklearn.ensemble.RandomForestRegressor¶.

            Обучение модели происходит на наборах данных - конфигурация и её эффективность, выбранные случайно
            из всех предыдущих запусков алгоритмов, в количестве TRAIN_AMOUNT_OF_DATA

            Возвращает обученную модель и время, потраченное на её обучение
            """
            start_time = now()
            REGRESSION_TREE_SET_CARDINALLITY = 10
            MINIMAL_DATA_POINTS_TO_SPLIT = 10
            TRAIN_AMOUNT_OF_DATA = 30

            model = RandomForestRegressor(n_estimators=REGRESSION_TREE_SET_CARDINALLITY,
                                          min_samples_split=MINIMAL_DATA_POINTS_TO_SPLIT)
            X, y = self.algo_trials.get_N_random_trials(TRAIN_AMOUNT_OF_DATA)
            model.fit(X, y)
            return model, (now() - start_time)

        def selectConfigurations(model):
            """Функция, которая по заданной модели выводит наиболее подходящие по эффективности конфигурации

            Процесс поиска подходящих конфигураций происходит в 2 этапа:
                1) Отбираем из известных нам запусков конфигурации с наилучшей эффективностью
                    в количестве CHECK_CONF_COUNTER
                2) Пытаемся найти лучшее значение, находящееся "недалеко" от выбранных конфигураций
                    Понятие "недалеко" определяется параметром NEARBY_RANGE - отнормированное значение для
                        метрики гиперпараметра
                    Если после NEARBY_AMOUNT попыток не получается найти лучшее значение заканчиваем поиск

            После поиска добавляется ещё RANDOM_CONF_ADDITION_AMOUNT случайно выбранных конфигураций для тестирования

            Возвращает - [конфигурации] и время затраченное на свою работу
            """
            start_time = now()

            predict_list = {}
            runs_list = self.algo_trials.get_conf_list()

            for (conf, perf) in runs_list:
                conf_as_key = tuple(conf)
                predict_list[conf_as_key] = model.predict([self.algo_trials.transform(conf)])

            CHECK_CONF_COUNTER = 10
            NEARBY_RANGE = 0.2
            NEARBY_AMOUNT = 4
            RANDOM_CONF_ADDITION_AMOUNT = 1000

            counter = 0

            output_list = {}

            for (conf, pred) in sorted(predict_list.items(), key=lambda x: x[1]):
                if counter == CHECK_CONF_COUNTER:
                    break
                counter += 1

                not_found_better = False
                while not_found_better:
                    not_found_better = True
                    for i in range(NEARBY_AMOUNT):
                        new_conf = []
                        for param in conf:
                            new_conf += [param.get_nearby_copy(NEARBY_RANGE)]

                        new_conf_prediction = model.predict(new_conf)
                        if new_conf_prediction < pred:
                            output_list[new_conf] = new_conf_prediction
                            not_found_better = False
                output_list[tuple(conf)] = pred

            for i in range(RANDOM_CONF_ADDITION_AMOUNT):
                rng_conf = []
                for param in self.initial_conf:
                    rng_conf += [param.get_random_copy()]
                output_list[tuple(rng_conf)] = model.predict([self.algo_trials.transform(rng_conf)])

            return list(output_list.keys()), (now() - start_time)

        def intensify(selected_conf, best_conf, time_given):
            """Функция, отбирабщая наилучшую конфигурацию из переданного списка

            Параметры
            ---------
            selected_conf : [[Hyperparameter]]
                Список конфигураций, для выбора

            best_conf : [Hyperparemeter]
                Текущая лучшая конфигурация

            time_given : deltatime
                Время выделенное на поиск

            Идея алгоритма - выбрать лучшую конфигурацию не только для наборов аргументов,
            но и для его подмножеств, таким образом, чтобы сравнить новую конфиграцию функция
            случайно выбирает из запусков старой наборы аргументов и тестирует новую на них, если
            суммарно новая конфигурация оказаласть лучше старой, то она считается лучшей

            NOTE: изначально, если у текущей лучшей конфигурации недостаточно наборов аргументов (<MAX_RUNS),
            то они будут случайно генерироваться и тестироваться

            Возвращает - [Hyperparameter]
                Конфигурацию, которая по истечении времени функция считала лучшей
            """
            MAX_RUNS = 2000

            t_args = self.args_keeper.give_full_template()

            start_time, count = now(), 0

            for new_conf in selected_conf:
                count += 1

                if self.algo_trials.check_confing_runs(best_conf) < MAX_RUNS:
                    reduced_args = rng.sample(t_args, rng.randint(len(t_args) // 2, len(t_args) - 1))
                    exec_run(self, best_conf, reduced_args)

                N = 2
                while True:
                    conf_set_addition = self.algo_trials.get_instance_for_diff(new_conf, best_conf)
                    conf_set_toRun = rng.sample(conf_set_addition, min(N, len(conf_set_addition)))  # get random subset

                    for rArgs in conf_set_toRun:
                        exec_run(self, new_conf, rArgs)

                    conf_set_addition = list(set(conf_set_addition) - set(conf_set_toRun))  # CSA = CSA\CST

                    args_runned_for_both = self.algo_trials.get_instance_for_equl(new_conf, best_conf)

                    bc_performance = self.algo_trials.summarize_performance(best_conf, args_runned_for_both)
                    nc_performance = self.algo_trials.summarize_performance(new_conf, args_runned_for_both)

                    if nc_performance < bc_performance:
                        break
                    elif len(conf_set_addition) == 0:
                        best_conf = new_conf
                        break
                    else:
                        N = N * 2

                if now() > start_time + time_given and count >= 2:
                    break

            return best_conf

        best_conf = initialize()
        start_time = now()
        while True:
            model, time_fit = fitModel()
            selected_conf, time_select = selectConfigurations(model)
            best_conf = intensify(selected_conf, best_conf, time_fit + time_select)
            if now() > start_time + self.work_time:
                break

        answer_params = dict()
        for param in best_conf:
            answer_params.update(param.get_named_value())

        configured_estimator = type(self.estimator)(**answer_params)

        return configured_estimator.fit(*args)
