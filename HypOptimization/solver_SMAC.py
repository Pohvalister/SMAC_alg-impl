import solver_base_for_SMAC as sb
import numpy as np
import random as rng
import datetime
from collections import defaultdict  # multidimentional dict
from sklearn.ensemble import RandomForestRegressor  # based on RandomForests

now = datetime.datetime.now

TIME_TO_WORK = datetime.timedelta(0, 0, 0, 0, 5)


class ArgumentSpaceHandler():
    def __init__(self, *args):
        listed_args = []
        for data in list(args):
            listed_args += [list(data)]
        self.listed_args = listed_args

    def give_arg_space(self, template):
        answer = []
        for arg in self.listed_args:
            data = []
            for place in template:
                data.append(arg[place])
            answer.append(data)
        return answer

    def give_full_template(self):
        return range(len(self.listed_args[0]))


class AlgorithmTrialsTracker():
    def __init__(self, estimator, scorer):
        self.estimator = estimator
        self.scorer = scorer
        self.runs_list = []
        self.runs_info = dict()  # defaultdict(dict)

    # выполняет обучение estimator на заданное конф и тестах, запоминает их
    # возвращает оценку scorera на полученную обученную машину
    def exec_run(self, configuration, *args):
        params = dict()
        for conf in configuration:
            params.update(conf.get_parameter())

        configured_estimator = type(self.estimator)(**params)
        fited = configured_estimator.fit(*args)
        performance = self.scorer(fited, *args)

        self.runs_list += [(configuration, performance)]
        key_conf = tuple(configuration)
        if not key_conf in self.runs_info:
            self.runs_info[key_conf] = []

        dont_exist = True
        for (exist_args, count, exist_perf) in self.runs_info[key_conf]:
            if exist_args == args:
                exist_perf = (exist_perf * count + performance) / (count + 1)
                count += 1
                break
        if dont_exist:
            self.runs_info[key_conf] += [(args, 1, performance)]

        return performance

    # возвращает instance для которых conf1 раньше запускался, а conf2 нет
    def get_instance_seed_pairs_for(self, conf_not_runned, conf_runned):
        inst_conf_nR = []  # list(self.runs_info[conf_not_runned].keys())
        inst_conf_R = []  # list(self.runs_info[conf_runned].keys())
        for (args, amount, perf) in self.runs_info[tuple(conf_not_runned)]:
            inst_conf_nR += [args]
        for (args, amount, perf) in self.runs_info[tuple(conf_runned)]:
            inst_conf_R += [args]

        diff_inst_runs = np.setdiff1d(inst_conf_R, inst_conf_nR)  # DIR = ICR\ICN
        return diff_inst_runs

    # возвращает пары instance для которых оба conf запускались
    def get_instance_for(self, conf_runned1, conf_runned2):
        inst_conf_R1 = list(self.runs_info[conf_runned1].keys())
        inst_conf_R2 = list(self.runs_info[conf_runned2].keys())

        equl_inst_runs = set(inst_conf_R1).intersection(inst_conf_R2)
        return equl_inst_runs

    # [Hyperparameter] -> [values]
    def transform(self, c):
        params = []
        for param in c:
            params += [param.get_value()]
        return params

    # возвращает N запомненных значений для fit
    def get_N_random_trials(self, N):
        X, y = [], []
        for i in range(N):
            (conf, perf) = rng.choice(self.runs_list)
            X += [self.transform(conf)]
            y += [perf]
        return X, y

    # сколько раз запускали на доанной conf
    def check_confing_runs(self, conf):
        return len(self.runs_info[tuple(conf)])

    # суммирует performance для текущей conf и списков параметров
    def summarize_performance(self, conf, args_list):
        sum = 0
        for args in args_list:
            sum += self.runs_info[conf][args]
        return sum

    def get_conf_list(self):
        return self.runs_list


class SMAC_solver(sb.Solver):
    # estimator - алгоритм, для которого оптимизируем параметры
    # params - пространство гиперпараметров
    # scorer - оценивающая функция
    #   __init__(self, estimator, params: [Hyperparameter], scoring=None):

    # args - аргументы для которых оптимизируется работа estimator по params
    # возвращает - оптимизированную конфигурацию из params

    def fit(self, *args):
        time_to_work: [datetime] = TIME_TO_WORK
        self.initial_conf = self.conf_space
        self.algo_trials = AlgorithmTrialsTracker(self.estimator, self.scorer)
        self.args_keeper = ArgumentSpaceHandler(*args)

        def initialize(*args):
            # args - аргументы для которых оптимизируется
            # еденичный запуск estimator, на рандомных или предустановленных p из params
            # возвращает - его performance
            rng_conf = []
            for param in self.initial_conf:
                rng_conf += [param.get_random_copy()]
            self.algo_trials.exec_run(rng_conf, *args)
            return rng_conf

        def fitModel():
            # в качестве модели используем RandomForests, создаем на n выборах (с повторениями) из тестовых данных (conf + perf)
            # возвращает модель и время своей работы

            start_time = now()
            REGRESSION_TREE_SET_CARDINALLITY = 10
            MINIMAL_DATA_POINTS_TO_SPLIT = 10
            TRAIN_AMOUNT_OF_DATA = 10

            model = RandomForestRegressor(n_estimators=REGRESSION_TREE_SET_CARDINALLITY,
                                          min_samples_split=MINIMAL_DATA_POINTS_TO_SPLIT)
            X, y = self.algo_trials.get_N_random_trials(TRAIN_AMOUNT_OF_DATA)
            model.fit(X, y)
            return model, (now() - start_time)

        def selectConfigurations(model, best_conf):  # self.params
            # использует модель чтобы выбрать список перспективных conf. Исп pridictive распред модели для расчета EI(conf). Высчитываем конфигурации из
            # предыдущих runs алгоритма, берем 10 с макс EI и локально смотрим ищем рядом с ними. Чтобы искать мы нормализуем параметры до [0,1] и берм 4 рядом лежащих значения
            # останавливаемся, когда ни один из соседей не показал лучший результат. ++ ещё N-ное количество рамндомных самплов. Сортим N+10
            # возвращ время своей работы
            start_time = now()

            predict_list = {}
            runs_list = self.algo_trials.get_conf_list()

            for (conf, perf) in runs_list:
                predict_list[tuple(conf)] = model.predict([self.algo_trials.transform(conf)])

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
                output_list[tuple(conf)] = pred  # !!!

            for i in range(RANDOM_CONF_ADDITION_AMOUNT):
                rng_conf = []
                for param in self.initial_conf:
                    rng_conf += [param.get_random_copy()]
                output_list[tuple(rng_conf)] = model.predict([self.algo_trials.transform(rng_conf)])

            return sorted(output_list.items(), key=lambda x: x[1]), (now() - start_time)

        def intensify(selected_conf, best_conf, model, time_given, *args):
            # selected_conf - подпространство params для которого intensify
            # best_conf - текущая лучшая конфигурация
            # time_given - время данное на расчеты
            # возвращает updated algo_trials, best_conf

            MAX_RUNS = 2000

            start_time, count = now(), 0

            for new_conf in selected_conf:
                count += 1

                if self.algo_trials.check_confing_runs(best_conf) < MAX_RUNS:
                    reduced_args = args#get_args

                    self.algo_trials.exec_run(best_conf, *reduced_args)

                N = 1
                while True:
                    conf_set_addition = self.algo_trials.get_instance_seed_pairs_for(new_conf, best_conf)
                    conf_set_toRun = rng.sample(conf_set_addition, min(N, len(conf_set_addition)))  # get random subset

                    # for (rArgs, rSeed) in conf_set_toRun:
                    for rArgs in conf_set_toRun:
                        self.algo_trials.exec_run(new_conf, rArgs)

                    conf_set_addition = np.setdiff1d(conf_set_addition, conf_set_toRun)  # CSA = CSA\CST

                    args_runned_for_both = self.algo_trials.get_instance_for(new_conf, best_conf)

                    bc_performance = self.algo_trials.summarize_performance(best_conf, args_runned_for_both)
                    nc_performance = self.algo_trials.summarize_performance(new_conf, args_runned_for_both)

                    if nc_performance > bc_performance:  # !!!!наоборот мб
                        break
                    elif len(conf_set_addition) == 0:
                        best_conf = new_conf
                        break
                    else:
                        N = N * 2

                if now() > start_time + time_given and count >= 2:
                    break

            return best_conf

        # core
        best_conf = initialize(*args)
        start_time = now()
        while True:
            model, time_fit = fitModel()
            selected_conf, time_select = selectConfigurations(model, best_conf)
            best_conf = intensify(selected_conf, best_conf, model, time_fit + time_select, *args)
            if now() > start_time + time_to_work:
                break

        return self.algo_trials.exec_run(best_conf, args)
