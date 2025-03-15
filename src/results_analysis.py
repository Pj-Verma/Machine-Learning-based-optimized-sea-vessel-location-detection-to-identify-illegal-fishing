from __future__ import division
import pandas as pd
from sklearn.metrics import confusion_matrix ,matthews_corrcoef
import pickle as cPickle
import matplotlib.pyplot as plt
import numpy as np
from src.pipeline_noLatLon import keep_columns, X_y_split, models_dict, threshold_fishing
from src.utilities import get_all_data, get_group, sample_by_vessel


def make_predictions(X_train, X_cross, X_test, pickeled_model):
    with open(pickeled_model, 'rb') as f:
        model = cPickle.load(f)
    pred1 = model.predict(X_train)
    pred2 = model.predict(X_cross)
    pred3 = model.predict(X_test)
    return pred1, pred2, pred3

# commented by pj on 29/03/24

def get_scores2(y_predict, y_test):
    C = confusion_matrix(y_test, y_predict)
    accurary = (C[0][0] + C[1][1]) / len(y_test)
    precision = C[1][1] / (C[0][1] + C[1][1])
    recall = C[1][1] / (C[1][1] + C[1][0])
    F1 = 2 * (precision * recall) / (precision + recall)
    # Accuracy, Recall, F1-score
    # print(accurary)
    # print(recall)
    # print(F1)
    return accurary, recall, F1


def plot_results(metric, data_subset, metric_df):
    col_names = ['RF_longliners', 'GBC_longliners', 'RF_trawlers',
                 'GBC_trawlers', 'RF_purse_seines', 'GBC_purse_seines']

    plt.plot(metric_df[col_names])
    plt.xlabel("Time Windows(hours)")
    plt.ylabel("F1 score")
    plt.legend(col_names, loc='best')  # 7 --> right centered
    plt.title(metric + ' scores on ' + data_subset + ' subset')
    # plt.subplot(1,3,1)
    # plt.show()
    plt.savefig('model_comparison' + metric + '.png')
    # plt.savefig('../results/models_performance.png')
    plt.savefig('E:/Machine learning/Geo_areas_to_detect_illegal_fishing/results/models_performance.png')


gears = ['longliners', 'trawlers', 'purse_seines']
classifiers = ['RF', 'GBC']
models_by_columns = ['model_1', 'model_2', 'model_3',
                     'model_4', 'model_5', 'model_6']
path = '../data/labeled'
data_dict = get_all_data(path)

# Data frames for the metrics:
# each row represents a type of column model: model_1,...,model_6
# each column is a combinations: classifier + gear
# ['RF_longliners','RF_trawlers','RF_purse_seines','GBC_longliners','GBC_trawlers','GBC_purse_seines']

acc_metrics_df_train = pd.DataFrame()
acc_metrics_df_cross = pd.DataFrame()
acc_metrics_df_test = pd.DataFrame()
rec_metrics_df_train = pd.DataFrame()
rec_metrics_df_cross = pd.DataFrame()
rec_metrics_df_test = pd.DataFrame()
f1_metrics_df_train = pd.DataFrame()
f1_metrics_df_cross = pd.DataFrame()
f1_metrics_df_test = pd.DataFrame()

for gear in gears:

    df = get_group(data_dict, gear)
    df.reindex()
    df = threshold_fishing(df)

    for classy in classifiers:

        acc_train_array = []
        acc_cross_array = []
        acc_test_array = []
        rec_train_array = []
        rec_cross_array = []
        rec_test_array = []
        f1_train_array = []
        f1_cross_array = []
        f1_test_array = []

        for model_num in models_by_columns:
            col_groups = models_dict[model_num]
            df_subset = keep_columns(df, col_groups=col_groups)

            df_train, df_cross, df_test = sample_by_vessel(df_subset, size=20000, even_split=None, seed=4321)
            X_train, y_train, cols = X_y_split(df_train)
            X_cross, y_cross, cols = X_y_split(df_cross)
            X_test, y_test, cols = X_y_split(df_test)

            # model_result_file = 'results/' + gear + '/' +model_num +'_'+ classy +'_'+ gear +'.txt'
            # pickled_model_file = 'results/' + gear + '/' +model_num +'_'+ classy +'_'+ gear +'.pkl'
            #pickled_model_file = 'C:/Users/pankaj kumar/Downloads/Detecting-fishing-activity-master/results/' + model_num + '_' + classy + '_' + gear + '.pkl'
            pickled_model_file = 'E:/Machine learning/Geo_areas_to_detect_illegal_fishing/results/' + model_num + '_' + classy + '_' + gear + '.pkl'
            pred1, pred2, pred3 = make_predictions(X_train, X_cross, X_test, pickled_model_file)

            a1, r1, f1 = get_scores2(pred1, y_train)
            a2, r2, f2 = get_scores2(pred2, y_cross)
            a3, r3, f3 = get_scores2(pred3, y_test)

            acc_train_array.append(a1)
            acc_cross_array.append(a2)
            acc_test_array.append(a3)

            rec_train_array.append(r1)
            rec_cross_array.append(r2)
            rec_test_array.append(r3)

            f1_train_array.append(f1)
            f1_cross_array.append(f2)
            f1_test_array.append(f3)

        col_ = classy + '_' + gear

        print('done with ' + col_)

        acc_metrics_df_train[col_] = acc_train_array
        acc_metrics_df_cross[col_] = acc_cross_array
        acc_metrics_df_test[col_] = acc_test_array

        rec_metrics_df_train[col_] = rec_train_array
        rec_metrics_df_cross[col_] = rec_cross_array
        rec_metrics_df_test[col_] = rec_test_array

        f1_metrics_df_train[col_] = f1_train_array
        f1_metrics_df_cross[col_] = f1_cross_array
        f1_metrics_df_test[col_] = f1_test_array

plot_results('F1 score', 'test', rec_metrics_df_train)





