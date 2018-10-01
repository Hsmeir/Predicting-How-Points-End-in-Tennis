# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 19:51:15 2018

@author: hm2sm
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 21:32:00 2018

@author: hm2sm
"""
import pickle
import math
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.externals import joblib
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def combineResults(name):
    """ Combine mean and women predictions, and save the result to file
    Args:
        name (str): file name info
    """
    ds_m = pd.read_csv("Xtest_{}_mens.csv".format(name))
    ds_w = pd.read_csv("Xtest_{}_womens.csv".format(name))
    ds_f = pd.concat((ds_m.iloc[:,1:], ds_w.iloc[:,1:]), axis = 0)
    ds_f.to_csv("Xtest_{}_full.csv".format(name), index = False)


def save_output(model, name, model_dict, men = True):
    """output and save the predictions of a model to a csv file
    Args:
        model (str): Name of model to use
        name (str): File name
        model_dict (dict): python dictionary containes trained models
        men (bool): True men data is used
    """
    d = pd.read_csv("AUS_SubmissionFormat.csv")

    if men:
        ds_test = pd.read_csv("mens_test_file.csv")
        ds_test_id = ds_test['id'].astype(str)+"_mens"
    else:
        ds_test = pd.read_csv("womens_test_file.csv")
        ds_test_id = ds_test['id'].astype(str)+"_womens"
        
    ds_test_id.name = "submission_id"    
    ds_test = pre_processing(ds_test, True)
    ds_test = feature_engineering(ds_test)
    ds_test = ds_test.drop(['id','train', 'outcome'], axis = 1)
    pred_test = model.predict_proba(ds_test.values)
    pred_test = pd.DataFrame(data = pred_test, columns = ['UE', 'FE','W'])
    d1 = d.iloc[:2000,:]
    test_id = pd.DataFrame(ds_test_id)
    pred_test = pd.concat((test_id, pred_test), axis = 1)
    pred_test = pd.merge(d1.iloc[:,:2], pred_test, on = "submission_id")
    
    if men:
        pred_test.to_csv("Xtest_{}_mens.csv".format(name))
        joblib.dump(model_dict, "./{}_mens".format(name))
    else:
        pred_test.to_csv("Xtest_{}_womens.csv".format(name))
        joblib.dump(model_dict, "./{}_womens".format(name))
        
def save_obj(model, filename):
    """output an object to file
    Args:
        model: An object to save
        filename: A string used to name the file including directory
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
        
def load_obj(filename):
    """load an object from file
    Args:
        filename: A string represents filename inculuding directory
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

def plot_class_dist(data, labels = None):
    """ Plot the distributaion for the classes
    Args:
        data: A numpy vector containing the class data
        labels: optional labels to display for each class. if none
                numbers from 0 to num class -1 will be used
    """
    label_count = np.unique(data, return_counts = True)
    fig = plt.figure()
    ax = fig.gca()   
    ax.set_xticks(label_count[0])
    if labels:        
        ax.set_xticklabels(labels, rotation='horizontal')
    ax.set_ylabel('Percentage')
    ax.set_xlabel('Class')
    plt.title('Class Distribution')
    plt.bar(x = label_count[0], height = label_count[1]/data.shape[0])
    plt.grid()
    plt.show()
    
def pre_processing(ds, gender = False):
    """ Perfomrs label encoding for some categorical features
    Args:
        ds (pandas DataFrame): DataFrame contains dataset
        gender (bool) : True ds contains gender feature
    Returns:
        resulting pandas DataFrame  
    """
    
    hitpoint_feats = ['hitpoint','previous.hitpoint']
    hitpoint_dict = {'F':0, 'B':1, 'V':2, 'U':3}
    for i in hitpoint_feats:
        ds.loc[:,i] = ds.loc[:,i].map(hitpoint_dict)

    boolean_feats = ['outside.sideline', 'outside.baseline', 
                     'same.side','server.is.impact.player']
    boolean_dict = {True:0, False:1}
    for i in boolean_feats:
        ds.loc[:,i] = ds.loc[:,i].map(boolean_dict)

    outcome_dict = {'UE':0, 'FE':1, 'W':2}
    ds.loc[:,'outcome'] = ds.loc[:,'outcome'].map(outcome_dict)
    #rearranging features
    cat_feat = ['serve', 'hitpoint' , 'previous.hitpoint', 'outside.sideline', 
                'outside.baseline', 'same.side', 'server.is.impact.player']
    if gender:
        gender_dict = {'mens':0, 'womens':1}
        ds.loc[:,'gender'] = ds.loc[:,'gender'].map(gender_dict)
      
    return ds[cat_feat + list(ds.drop(cat_feat + ["outcome"], axis =1).columns) 
              + ["outcome"]]


def feature_engineering(ds):
    """ performs feature engineering 
    Args:
        ds (pandas DataFrame): DataFrame contains dataset samples
    """
    
    index = (ds['outside.sideline']==0) & (ds['outside.baseline']==0)
    ds['dist'] = (ds['player.impact.distance.from.center']**2 + 
                  ds['player.impact.depth'])
    index_F = ds['hitpoint'] == 0
    index_B = ds['hitpoint'] == 1
    index_t = index_F | index
    ds['speed_dist_ratio'] = 0*ds['speed']
    factor = 3
    ds.loc[index_F,'speed_dist_ratio'] = ds.loc[index_F]['speed']\
                                        /((ds.loc[index_F]['dist']))
    ds.loc[index_B,'speed_dist_ratio'] = ds.loc[index_B]['speed']\
                                        /((factor*ds.loc[index_B]['dist']))
    ds.loc[~(index_t),'speed_dist_ratio'] = ds.loc[~(index_t)]['speed']\
                                       /((3*factor*ds.loc[~(index_t)]['dist']))
    #create different speed depth ratios
    ds['speed_depth_ratio'] = ds['speed']/(ds['player.impact.depth'])
    ds['prev_speed_depth_ratio_opp'] = ds['previous.speed']\
                                        /(ds['opponent.depth'])
    ds['prev_speed_depth_ratio'] = ds['previous.speed']\
                                    /(ds['previous.depth'])
    ds['speed_dist_ratio'] = ds['speed']\
                            /(ds['depth'] + ds['distance.from.sideline'])

    ds['sideline_ratio'] = ds['previous.distance.from.sideline']\
                            /ds['distance.from.sideline']
    #time speed distance
    ds['actual_dist'] = ds['previous.time.to.net']*ds['previous.speed']
    ds['diagonal_dist'] = np.sqrt(ds['opponent.depth']**2\
                          + ds['opponent.distance.from.center']**2)

    ds['dist_ratio'] = ds['actual_dist']/ds['opponent.depth']
    ds['speed_ratio'] = ds['speed']/ds['previous.speed']
    #creating ratio features of net clearance with different depths and distances
    ds['net_ratio1'] = ds['net.clearance']/ds['opponent.depth']
    ds['net_ratio2'] = ds['net.clearance']/ds['player.depth']
    ds['net_ratio3'] = ds['net.clearance']/ds['dist']
    #indicator flag comparing speed and previous speed
    ds['speed_flag'] = 1*(ds['speed'] > ds['previous.speed'])
    #rearranging features
    return ds[list(ds.drop("outcome", axis = 1).columns) + ["outcome"]]
   
#loading submission sample file
d = pd.read_csv("AUS_SubmissionFormat.csv")
d1 = d.iloc[:2000,:]
test1 = pd.read_csv("mens_test_file.csv")
test1_id = test1['id'].astype(str)+"_mens"
test1_id.name = "submission_id"
test1_id = pd.DataFrame(test1_id)
index = d['submission_id']
result = pd.merge(d1, test1_id, on = "submission_id")

#loading men trainig data 
cat_feat = ['serve', 'hitpoint' , 'previous.hitpoint', 'outside.sideline', 
            'outside.baseline', 'same.side', 'server.is.impact.player']
ds_men = pd.read_csv("mens_train_file.csv")
ds_men = ds_men.drop(['id', 'train'], axis = 1)
#generate simple information and statistics about the data, and check missing 
#values
ds_men.info()
ds_men.describe(include = 'all')

#checking for missing values
missing = pd.isnull(ds_men).any().any()
#loading women trainig data 
ds_women = pd.read_csv("womens_train_file.csv")
ds_women = ds_women.drop(['id', 'train'], axis = 1)
#generate simple information and statistics about the data, and check missing 
#values
ds_men.info()
ds_men.describe(include = 'all')

# preproccessing and feature engineering
ds_men = pre_processing(ds_men, True)
ds_women = pre_processing(ds_women, True)
ds_men = feature_engineering(ds_men)
ds_women = feature_engineering(ds_women)
#check class distribuation fro each data
plot_class_dist(ds_men['outcome'], labels = ['UE', 'FE', 'W'])
plot_class_dist(ds_women['outcome'], labels = ['UE', 'FE', 'W'])
#create a 5 folds stratified splits
k = 5
seed = 100
skf = StratifiedKFold(n_splits = k, random_state = seed, shuffle = True)
X_tr = ds_men.iloc[:,:-1].values
y_tr = ds_men.iloc[:,-1:].values

# using xgboost on mens data
#XGBoost hyperparameters grid
num_estimators = np.arange(100,400,25)
max_depth = [4]
param = {
        'objective': 'multi:softprob',
        'max_depth':max_depth[0],
        'n_estimators': num_estimators[0],      
        'colsample_bytree' : 0.4,
        'subsample': 0.5,
        'seed': 7,
        'min_child_weight': 6,
        'gamma':0,
        'lambda' : 0,
        'alpha':0,
        'silent': 1,
        'eval_metric':'mlogloss',
        'num_class':3,
        'eta':  0.001,
        'tree_method': 'gpu_exact'
        }

loss_train = np.zeros((len(num_estimators), len(max_depth), k))
loss_val = np.zeros((len(num_estimators), len(max_depth), k))

xgb_model = dict()
for i in range(len(num_estimators)):
    for j in range(len(max_depth)):
        param.update({'n_estimators': num_estimators[i], 
                      'max_depth': max_depth[j]})
        clf = xgb.XGBClassifier(**param)
        l = 0
        for train_index, test_index in skf.split(X_tr, y_tr.squeeze()):             
            #adding women data to men data for training
            X_tr_aug = np.concatenate((X_tr[train_index], 
                                       ds_women.iloc[:,:-1].values), axis = 0)
            y_tr_aug = np.concatenate((y_tr[train_index], 
                                       ds_women.iloc[:,-1:].values), axis = 0)
            #create a weight for the samples based on their gender
            scale = 0.8
            sample_weight = np.r_[np.ones((X_tr[train_index].shape[0])), 
                                  scale*np.ones((ds_women.shape[0]))]

            clf.fit(X_tr_aug, y_tr_aug.squeeze(), sample_weight =sample_weight)
            y_score_train = clf.predict_proba(X_tr[train_index])[:,1]
            y_score_test = clf.predict_proba(X_tr[test_index])[:,1]
            #taking the loss on the men samples as an evaluation metric
            loss_train[i, j, l] = log_loss(y_true = y_tr[train_index], 
                                 y_pred = clf.predict_proba(X_tr[train_index]))
            loss_val[i, j, l] = log_loss(y_true = y_tr[test_index], 
                                y_pred = clf.predict_proba(X_tr[test_index]))
            xgb_model['{}_{}'.format(i, j)] = clf
            l += 1    

# feature importances
loss_train_avg = np.mean(loss_train, axis = 2)
loss_val_avg = np.mean(loss_val, axis = 2)
best_model = np.argwhere(np.round(loss_val_avg, 3)\
                         == np.min(np.round(loss_val_avg, 3)))[0]

feat_importance = xgb_model['{}_{}'.format(best_model[0],
                            best_model[1])].feature_importances_
feat_importance = pd.DataFrame(data = feat_importance, 
                               columns = ['feature_importance'], 
                               index = ds_men.columns[:-1])      
sns.barplot(x = feat_importance.values.squeeze(), 
            y = feat_importance.index.tolist())
feat_importance = feat_importance.sort_values(by = 'feature_importance')

print('best number of estimator = {}, best max depth = {}'
      .format(num_estimators[best_model[0]], max_depth[best_model[1]]))
print('Loss train =', loss_train_avg[best_model[0], best_model[1]])
print('Loss validation =', loss_val_avg[best_model[0], best_model[1]])

fig = plt.figure(figsize = (16, 10))
fig.suptitle("XGBoost AUC ")
for i in range(len(max_depth)):
    ax = plt.subplot(2, 2, i+1)
    ax.plot(num_estimators, loss_train_avg[:, i], c = 'b')
    plt.hold
    ax.plot(num_estimators, loss_val_avg[:, i], c= 'r')
    plt.title('num estimators = {}'.format(max_depth[i]))
    plt.xticks(num_estimators)
    plt.legend(['Train', 'Test'])
    plt.tight_layout()

#Retrain the best XGBoost Classifier on the full training set
X_tr_full = np.concatenate((X_tr, ds_women.iloc[:,:-1].values), axis = 0)
y_tr_full = np.concatenate((y_tr, ds_women.iloc[:,-1:].values), axis = 0)

xgb_model['{}_{}'.format(best_model[0], best_model[1])].fit(X_tr_full,
                                                           y_tr_full.squeeze())
# save results
model_name = "M1_seed{}".format(seed)
save_output(xgb_model['{}_{}'.format(best_model[0], best_model[1])], 
                      model_name, xgb_model, men = True )

# using xgboost on women data
X_tr = ds_women.iloc[:,:-1].values
y_tr = ds_women.iloc[:,-1:].values

#XGBoost hyperparameters grid
num_estimators = np.arange(100,400,25)
max_depth = [4]
param = {
        'objective': 'multi:softprob',
        'max_depth':max_depth[0],
        'n_estimators': num_estimators[0],        
        'colsample_bytree' : 0.5,
        'subsample': 0.5,
        'seed':77,
        'min_child_weight': 7,
        'gamma':0,
        'lambda' : 0,
        'alpha':0,
        'silent': 1,
        'eval_metric':'mlogloss',
        'num_class':3,
        'eta':  0.001,       
        'tree_method': 'gpu_exact'
        }

loss_train = np.zeros((len(num_estimators), len(max_depth), k))
loss_val = np.zeros((len(num_estimators), len(max_depth), k))

xgb_model2 = dict()
for i in range(len(num_estimators)):
    for j in range(len(max_depth)):
        param.update({'n_estimators': num_estimators[i],
                      'max_depth': max_depth[j]})
        clf = xgb.XGBClassifier(**param)
        l = 0
        for train_index, test_index in skf.split(X_tr, y_tr.squeeze()):             
            #adding women data to men data for training
            X_tr_aug = np.concatenate((X_tr[train_index], 
                                       ds_men.iloc[:,:-1].values), axis = 0)
            y_tr_aug = np.concatenate((y_tr[train_index], 
                                       ds_men.iloc[:,-1:].values), axis = 0)
            #create a weight for the samples based on their gender
            scale = 0.8
            sample_weight = np.r_[np.ones((X_tr[train_index]).shape[0]), 
                                  scale*np.ones((ds_men.shape[0]))]
            clf.fit(X_tr_aug, y_tr_aug.squeeze(), sample_weight =sample_weight)
            y_score_train = clf.predict_proba(X_tr[train_index])[:,1]
            y_score_test = clf.predict_proba(X_tr[test_index])[:,1]
            #taking the loss on the men samples as an evaluation metric
            loss_train[i, j, l] = log_loss(y_true = y_tr[train_index], 
                                 y_pred = clf.predict_proba(X_tr[train_index]))
            loss_val[i, j, l] = log_loss(y_true = y_tr[test_index], 
                                  y_pred = clf.predict_proba(X_tr[test_index]))
            xgb_model2['{}_{}'.format(i, j)] = clf
            l += 1    

# feature importances
loss_train_avg = np.mean(loss_train, axis = 2)
loss_val_avg = np.mean(loss_val, axis = 2)
best_model2 = np.argwhere(np.round(loss_val_avg, 3)\
                          == np.min(np.round(loss_val_avg, 3)))[0]
     
feat_importance = xgb_model2['{}_{}'.format(best_model2[0],
                             best_model2[1])].feature_importances_
feat_importance = pd.DataFrame(data = feat_importance, 
                               columns = ['feature_importance'],
                               index = ds_men.columns[:-1])      
sns.barplot(x = feat_importance.values.squeeze(), 
            y = feat_importance.index.tolist())
feat_importance = feat_importance.sort_values(by = 'feature_importance')

print('best number of estimator = {}, best max depth = {}'.
      format(num_estimators[best_model2[0]], max_depth[best_model2[1]]))
print('Loss train =', loss_train_avg[best_model2[0], best_model2[1]])
print('Loss validation =', loss_val_avg[best_model2[0], best_model2[1]])

fig = plt.figure(figsize = (16, 10))
fig.suptitle("XGBoost AUC ")
for i in range(len(max_depth)):
    ax = plt.subplot(2, 2, i+1)
    ax.plot(num_estimators, loss_train_avg[:, i], c = 'b')
    plt.hold
    ax.plot(num_estimators, loss_val_avg[:, i], c= 'r')
    plt.title('num estimators = {}'.format(max_depth[i]))
    plt.xticks(num_estimators)
    plt.legend(['Train', 'Test'])
    plt.tight_layout()

# save results
model_name = "M1_seed{}".format(seed)
save_output(xgb_model2['{}_{}'.format(best_model2[0], best_model2[1])],
                       model_name, xgb_model2, men = False )

combineResults("M1_seed{}".format(seed))   
