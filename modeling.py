import random
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy import stats
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score, accuracy_score, \
    roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from feature_extra import *
from data_processing import interpolation_data


random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)


def evaluation(y_test, y_prob, threshold=0.3):
    # Calculate recall, precision and other indicators
    y_pred = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return accuracy, recall, precision, f1score, roc_auc, sensitivity, specificity


def feature_select_ttest(feature, feature_test, label):
    # Feature selection using t-test
    w, h = np.shape(feature)
    feature_pd = pd.DataFrame(feature, columns=[f'feature_{i}' for i in range(0, h)])
    feature_test_pd = pd.DataFrame(feature_test, columns=[f'feature_{i}' for i in range(0, h)])
    t_test_results = pd.DataFrame(columns=['Feature', 't_value', 'p_value'])
    for feature in feature_pd:
        label_0 = feature_pd[label == 0][feature]
        label_1 = feature_pd[label == 1][feature]

        t_stat, p_value = stats.ttest_ind(label_0, label_1)
        t_test_results = pd.concat([t_test_results, pd.DataFrame({'Feature': [feature], 't_value': [t_stat], 'p_value': [p_value]})])

    # 筛选出显著性水平为0.05的特征
    significant_features = t_test_results[t_test_results['p_value'] < 0.05]
    feature_filtered = feature_pd[significant_features['Feature'].values]
    feature_test_filtered = feature_test_pd[significant_features['Feature'].values]

    return feature_filtered, feature_test_filtered, significant_features


def model_with_base_learner(model_name='LR'):
    # Build the model, train and test it
    data, label = interpolation_data()

    data = data.to_numpy()

    label = np.array(label)
    feature_orig = original_feature(data)
    feature_diff = original_diff(data)
    print(np.shape(feature_orig), np.shape(feature_diff))

    feature = np.concatenate([feature_orig, feature_diff], axis=1)
    w, h = np.shape(feature)
    feature_pd = pd.DataFrame(feature, columns=[f'feature_{i}' for i in range(0, h)])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    recalls = []
    precisions = []
    f1scores = []
    roc_aucs = []
    sensitivities = []
    specificities = []

    if model_name=='LR':
        classifier = LogisticRegression(C=100, max_iter=2000, solver='newton-cholesky', random_state=42)
    elif model_name=='SVM':
        classifier = SVC(C=0.1, kernel='linear', gamma=0.1, probability=True)
    elif model_name == 'KNN':
        classifier = KNeighborsClassifier(n_neighbors=40)
    elif model_name == 'XGBoost':
        classifier = XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.05)
    elif model_name == 'Random Forest':
        classifier = RandomForestClassifier(n_estimators=15, max_depth=10, max_features='sqrt', random_state=42)

    prob_all = np.array([0, 1])
    label_all = np.array([0, 1])
    prob_all_train = np.array([0, 1])
    label_all_train = np.array([0, 1])

    for train_index, test_index in kf.split(feature_pd):
        X_train, X_test = feature_pd.iloc[train_index], feature_pd.iloc[test_index]
        y_train, y_test = label[train_index], label[test_index]

        X_train_sampled, y_train_sampled = X_train, y_train

        X_train_filtered, X_test_filtered, significant_features = feature_select_ttest(X_train_sampled, X_test, y_train_sampled)
        scaler = StandardScaler()
        X_train_filtered = scaler.fit_transform(X_train_filtered)
        X_test_filtered = scaler.transform(X_test_filtered)

        selector = SelectKBest(score_func=mutual_info_classif, k=12)
        X_train_filtered = selector.fit_transform(X_train_filtered, y_train_sampled)
        X_test_filtered = selector.transform(X_test_filtered)

        classifier.fit(X_train_filtered, y_train_sampled)
        y_prob = classifier.predict_proba(X_test_filtered)[:, 1]
        y_prob_train = classifier.predict_proba(X_train_filtered)[:, 1]

        threshold = sum(y_train_sampled) / len(y_train_sampled)
        accuracy, recall, precision, f1score, roc_auc, sensitivity, specificity = evaluation(y_test, y_prob, threshold=threshold)
        accuracies.append(accuracy)
        recalls.append(recall)
        precisions.append(precision)
        f1scores.append(f1score)
        roc_aucs.append(roc_auc)
        sensitivities.append(sensitivity)
        specificities.append(specificity)

        prob_all = np.concatenate([prob_all, np.squeeze(y_prob)])
        label_all = np.concatenate([label_all, y_test])

        prob_all_train = np.concatenate([prob_all_train, np.squeeze(y_prob_train)])
        label_all_train = np.concatenate([label_all_train, y_train_sampled])

    # Calculate the average accuracy across 5 folds
    average_accuracy = sum(accuracies) / len(accuracies)
    average_recall = sum(recalls) / len(accuracies)
    average_precision = sum(precisions) / len(accuracies)
    average_f1score = sum(f1scores) / len(accuracies)
    average_roc_auc = sum(roc_aucs) / len(accuracies)
    sensitivity = sum(sensitivities) / len(accuracies)
    specificity = sum(specificities) / len(accuracies)

    prob_all, label_all = prob_all[2:], label_all[2:]
    prob_all_train, label_all_train = prob_all_train[2:], label_all_train[2:]

    print(
        f"accuracy:{round(average_accuracy, 2)}, roc_auc:{round(average_roc_auc, 2)}, sensitivity:{round(sensitivity, 2)}, specificity:{round(specificity, 2)}")

    # # save the probability and label
    # df_save = pd.DataFrame({model_name: prob_all, 'label_all': label_all})
    # excel_file = r'.../' + model_name + '_prob.xlsx'
    # df_save.to_excel(excel_file, index=False)
    #
    # df_save_train = pd.DataFrame({model_name: prob_all_train, 'label_all_train': label_all_train})
    # excel_file_train = r'.../' + model_name + '_prob_train.xlsx'
    # df_save_train.to_excel(excel_file_train, index=False)


def get_selected_feature_for_ensemble_model():
    data, label = interpolation_data()

    data = data.to_numpy()

    feature_orig = original_feature(data)
    feature_diff = original_diff(data)

    feature = np.concatenate([feature_orig, feature_diff], axis=1)
    w, h = np.shape(feature)
    feature_pd = pd.DataFrame(feature, columns=[f'feature_{i}' for i in range(0, h)])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    select_features_all = np.array(np.zeros([2, 8]))
    select_features_train_all = np.array(np.zeros([2, 8]))

    for train_index, test_index in kf.split(feature_pd):
        X_train, X_test = feature_pd.iloc[train_index], feature_pd.iloc[test_index]

        # save the selected features
        temp_X_test = deepcopy(X_test)
        temp_X_test = np.array(temp_X_test)
        select_features = []
        for sf in [20, 23, 88, 89, 91, 66, 92, 202]:
            select_features.append(temp_X_test[:, sf])
        select_features = np.array(select_features).T
        select_features_all = np.concatenate([select_features_all, select_features])

        temp_X_train = deepcopy(X_train)
        temp_X_train = np.array(temp_X_train)
        select_features_train = []
        for sf in [20, 23, 88, 89, 91, 66, 92, 202]:
            select_features_train.append(temp_X_train[:, sf])
        select_features_train = np.array(select_features_train).T
        select_features_train_all = np.concatenate([select_features_train_all, select_features_train])

    # save the selected features
    # select_features_all = select_features_all[2:, :]
    # df_features = pd.DataFrame(select_features_all)
    # df_features.to_excel(r'.../select_features.xlsx', index=False)
    #
    # select_features_train_all = select_features_train_all[2:, :]
    # df_features_train = pd.DataFrame(select_features_train_all)
    # df_features_train.to_excel(r'.../select_features_train.xlsx', index=False)


def ensemble_model():
    excel_file = r'.../select_features.xlsx'
    df = pd.read_excel(excel_file)
    label_all = df['label_all'].array
    prob_all = pd.read_excel(excel_file, usecols=['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'KNN', 'LR', 'RF', 'SVM', 'XGBoost'])
    prob_all = np.array(prob_all)

    excel_file_train = r'.../select_features_train.xlsx'
    df_train = pd.read_excel(excel_file_train)
    label_all_train = df_train['label_all'].array
    prob_all_train = pd.read_excel(excel_file_train, usecols=['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'KNN', 'LR', 'RF', 'SVM', 'XGBoost'])
    prob_all_train = np.array(prob_all_train)

    params_dict = {'colsample_bytree': 0.07, 'learning_rate': 0.99, 'max_depth': 3, 'max_leaves': 22, 'alpha': 2.7,
                   'min_child_weight': 1.8, 'n_estimators': 11}
    classifier = XGBClassifier(**params_dict)
    kf = KFold(n_splits=5, shuffle=False)
    accuracies = []
    recalls = []
    precisions = []
    f1scores = []
    roc_aucs = []
    sensitivities = []
    specificities = []

    for (_, train_index), (_, test_index) in zip(kf.split(prob_all_train), kf.split(prob_all)):
        X_train, X_test = prob_all_train[train_index, :], prob_all[test_index, :]
        y_train, y_test = label_all_train[train_index], label_all[test_index]
        classifier.fit(X_train, y_train)
        y_prob = classifier.predict_proba(X_test)[:, 1]

        threshold = sum(y_train) / len(y_train)
        accuracy, recall, precision, f1score, roc_auc, sensitivity, specificity = evaluation(y_test, y_prob,
                                                                                             threshold=threshold)
        c_index = concordance_index(y_test, y_prob)
        print(roc_auc, c_index)

        accuracies.append(accuracy)
        recalls.append(recall)
        precisions.append(precision)
        f1scores.append(f1score)
        roc_aucs.append(roc_auc)
        sensitivities.append(sensitivity)
        specificities.append(specificity)

    # Calculate the average accuracy across 5 folds
    average_accuracy = sum(accuracies) / len(accuracies)
    average_recall = sum(recalls) / len(accuracies)
    average_precision = sum(precisions) / len(accuracies)
    average_f1score = sum(f1scores) / len(accuracies)
    average_roc_auc = sum(roc_aucs) / len(accuracies)
    sensitivity = sum(sensitivities) / len(accuracies)
    specificity = sum(specificities) / len(accuracies)
    print(
        f"accuracy:{average_accuracy}, roc_auc:{average_roc_auc}, sensitivity:{sensitivity}, specificity:{specificity}")


if '__main__' == __name__:
    # model_name = ['LR', 'SVM', 'XGBoost', 'Random Forest', 'KNN']
    model_with_base_learner(model_name='LR')
