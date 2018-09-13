import solver_base as sb
import numpy as np
import random as rng
from datetime import datetime
from collections import defaultdict # multidimentional dict
from sklearn.ensemble import RandomForestRegressor # based on RandomForests

now = datetime.now

MAX_RUNS = 2000
TIME_TO_WORK=datetime.min(5)

class Algorithm_Trials_Tracker():
    def __init__(self,estimator,scorer):
        self.estimator=estimator
        self.scorer = scorer
        self.runs_list=dict()

    # выполняет обучение estimator на заданное конф и тестах, запоминает их
    # возвращает оценку scorera на полученную обученную машину
    def exec_run(self,configuration,*args):
        configured_estimator = type(self.estimator)(**configuration)
        performance=self.scorer(configured_estimator.fit(*args),*args)
        self.runs_info = defaultdict(dict)
        self.runs_info[configuration][args] = performance #[conf:args:perf]
        return performance

    # возвращает пары <instance seed> для которых conf1 раньше запускался, а conf2 нет
    def get_instance_seed_pairs_for(self,conf_not_runned,conf_runned):
        inst_conf_nR= list(self.runs_info[conf_not_runned].keys())
        inst_conf_R= list(self.runs_info[conf_runned].keys())

        diff_inst_runs = np.setdiff1d(inst_conf_R, inst_conf_nR) # DIR = ICR\ICN
        return diff_inst_runs

    # возвращает пары <instance seed> для которых оба conf запускались
    def get_instance_for(self,conf_runned1,conf_runned2):
        inst_conf_R1 = list(self.runs_info[conf_runned1].keys())
        inst_conf_R2 = list(self.runs_info[conf_runned2].keys())

        equl_inst_runs = set(inst_conf_R1).intersection(inst_conf_R2)
        return equl_inst_runs

    # возвращает N запомненных значений
    def get_N_random_trials(self,N):
        return 0

    # сколько раз запускали на доанной conf
    def check_confing_runs(self, conf):
        return len(self.runs_info[conf])

    # суммирует performance для текущей conf и списков параметров
    def summarize_performance(self,conf,args_list):
        sum = 0
        for args in args_list:
            sum+=self.runs_info[conf][args]
        return sum

class SMAC_Solver(sb.Solver):
    # estimator - алгоритм, для которого оптимизируем параметры
    # params - пространство гиперпараметров
    # scorer - оценивающая функция
    #   __init__(self, estimator, params: [Hyperparameter], scoring=None):

    # args - аргументы для которых оптимизируется работа estimator по params
    # возвращает - оптимизированную конфигурацию из params

    def fit(self, time_to_work:[datetime]=TIME_TO_WORK, *args):

        self.conf_space = self.params
        self.algo_trials = Algorithm_Trials_Tracker(self.estimator,self.scorer)

        def initialize(*args):
            # args - аргументы для которых оптимизируется
            # еденичный запуск estimator, на рандомных или предустановленных p из params
            # возвращает - его performance
            rng_conf = dict()
            for conf in self.conf_space:
                rng_conf.update(conf.get_random())
            return self.algo_trials.exec_run(rng_conf,*args)

        def fitModel():
            # в качестве модели используем RandomForests, создаем на n выборах (с повторениями) из тестовых данных (conf + perf)
            # возвращает модель и время своей работы

            start_time=now()
            REGRESSION_TREE_SET_CARDINALLITY=10
            MINIMAL_DATA_POINTS_TO_SPLIT=10
            TRAIN_AMOUNT_OF_DATA=10

            model = RandomForestRegressor(n_estimators=REGRESSION_TREE_SET_CARDINALLITY,min_samples_split=MINIMAL_DATA_POINTS_TO_SPLIT)
            model.fit(self.algo_trials.get_N_random_trials(TRAIN_AMOUNT_OF_DATA))
            return model,(now() - start_time)

        def selectConfigurations(model, best_conf):#self.params
            # использует модель чтобы выбрать список перспективных conf. Исп pridictive распред модели для расчета EI(conf). Высчитываем конфигурации из
            # предыдущих runs алгоритма, берем 10 с макс EI и локально смотрим ищем рядом с ними. Чтобы искать мы нормализуем параметры до [0,1] и берм 4 рядом лежащих значения
            # останавливаемся, когда ни один из соседей не показал лучший результат. ++ ещё N-ное количество рамндомных самплов. Сортим N+10
            # возвращ время своей работы
            start_time = now()

            return smth, (now() - start_time)

        def intensify(selected_conf,best_conf,model, time_given, *args):
            # selected_conf - подпространство params для которого intensify
            # best_conf - текущая лучшая конфигурация
            # time_given - время данное на расчеты
            # возвращает updated algo_trials, best_conf
            start_time, count = now(), 0

            for new_conf in selected_conf:
                count+=1

                if self.algo_trials.check_confing_runs(best_conf) < MAX_RUNS:
                    reduced_args = args
                    #seed=
                    self.algo_trials.exec_run(best_conf,*reduced_args)

                N=1
                while True:
                    conf_set_addition = self.algo_trials.get_instance_seed_pairs_for(new_conf,best_conf)
                    conf_set_toRun =rng.sample(conf_set_addition,  min(N,len(conf_set_addition))) #get random subset

                    #for (rArgs, rSeed) in conf_set_toRun:
                    for rArgs in conf_set_toRun:
                        self.algo_trials.exec_run(new_conf,rArgs)

                    conf_set_addition=np.setdiff1d(conf_set_addition,conf_set_toRun) # CSA = CSA\CST

                    args_runned_for_both = self.algo_trials.get_instance_for(new_conf,best_conf)

                    bc_performance=self.algo_trials.summarize_performance(best_conf,args_runned_for_both)
                    nc_performance=self.algo_trials.summarize_performance(new_conf, args_runned_for_both)

                    if nc_performance > bc_performance: #!!!!наоборот мб
                        break
                    elif len(conf_set_addition) == 0:
                        best_conf=new_conf
                        break
                    else:
                        N=N*2

                if now()>start_time + time_given and count >=2:
                    break

            return best_conf

        # core algo
        # [R, oinc] <- Initialize(O, П)
        best_conf = initialize(*args)
        start_time = now()
        while True:
            # [M,tfit] <- FitModel(R)
            model, time_fit = fitModel()
            # [Onew, tselect] <- SelectConfigurations(M,oinc,O)
            selected_conf, time_select = selectConfigurations(model,best_conf)
            # [R,oinc] <- Intensify(Onew, oinc, M, R, tfit + tselect, П, c)
            best_conf = intensify(selected_conf,best_conf,model,time_fit+time_select, *args)
            if now()>start_time + time_to_work:
                break

        return best_conf





