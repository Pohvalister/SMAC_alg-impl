from sklearn.metrics import (accuracy_score,average_precision_score,f1_score,log_loss,precision_score, recall_score, roc_auc_score)
from sklearn.metrics import (adjusted_mutual_info_score,adjusted_rand_score,completeness_score, fowlkes_mallows_score, homogeneity_score, mutual_info_score,normalized_mutual_info_score,v_measure_score)
from sklearn.metrics import (explained_variance_score,mean_absolute_error,mean_squared_error,mean_squared_log_error,median_absolute_error,r2_score)

from sklearn.metrics import make_scorer

def call_internal_scorer(estimator, *args):
    return estimator.score(*args)


SCORERS = dict(#Classification
                accuracy                    = make_scorer(accuracy_score),
                average_precision           = make_scorer(average_precision_score, needs_threshold=True),
                f1                          = make_scorer(f1_score),
                f1_micro                    = make_scorer(f1_score),
                f1_macro                    = make_scorer(f1_score),
                f1_weighted                 = make_scorer(f1_score),
                f1_samples                  = make_scorer(f1_score),
                neg_log_loss                = make_scorer(log_loss, greater_is_better=True, needs_proba=True),
                precision                   = make_scorer(precision_score),
                recall                      = make_scorer(recall_score),
                roc_auc                     = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True),
              #Clustering
                adjusted_mutual_info_score  = make_scorer(adjusted_mutual_info_score),
                adjusted_rand_score         = make_scorer(adjusted_rand_score),
                completeness_score          = make_scorer(completeness_score),
                fowlkes_mallows_score       = make_scorer(fowlkes_mallows_score),
                homogeneity_score           = make_scorer(homogeneity_score),
                mutual_info_score           = make_scorer(mutual_info_score),
                normalized_mutual_info_score= make_scorer(normalized_mutual_info_score),
                v_measure_score             = make_scorer(v_measure_score),
              #Regression
                explained_variance          = make_scorer(explained_variance_score),
                neg_mean_absolute_error     = make_scorer(mean_absolute_error,greater_is_better=False),
                neg_mean_squared_error      = make_scorer(mean_squared_error, greater_is_better=False),
                neg_mean_squared_log_error  = make_scorer(mean_squared_log_error, greater_is_better=False),
                neg_median_absolute_error   = make_scorer(median_absolute_error, greater_is_better=False),
                r2                          = make_scorer(r2_score)
)
