# -*- coding: utf-8 -*-

#Headers and Imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#this was the smoothest programming experience Ive had with a ML algorithm
##############################################################################
def adaboost(xtrain, ytrain, T, classifier_type):
    
    #Step 1 Initialize Weights
    m=xtrain.shape[0]
    D_weights = np.zeros(shape=(T, m))
    D_weights[0] = np.ones(shape=m) / m
    
    #Initialize weak learner vector
    stumps = np.zeros(shape=T, dtype=object)
    stump_weights = np.zeros(shape=T)
    
    #get error
    errors = np.zeros(shape=T)
    
    #Step 2-6
    for t in range(T):
        
        #Train Weak classifiers
        curr_D_weights=D_weights[t]
        if(classifier_type=='Gradient_Boost'):
            stump = trainWeakGrad(xtrain, ytrain, curr_D_weights)
        else:
            stump = trainWeakTree(xtrain, ytrain, curr_D_weights)
        
        #return the weak hypothesis
        weakH = stump.predict(xtrain)
        
        #Get the Error
        err = curr_D_weights[(weakH != ytrain)].sum()# / n
        stump_weight = np.log((1 - err) / err) / 2
        
        #Weight Update
        update=updateDt(weakH,curr_D_weights,stump_weight, ytrain)
        if t+1 < T:
             D_weights[t+1] = update
        stumps[t]=stump
        stump_weights[t]=stump_weight
        errors[t]= err
        
        
    #Step 7 Return Everything, I know there is a better way to return this other than a tuple but yolo
    return (stumps, stump_weights, errors, D_weights)

##################################################################################
def trainWeakTree(X, Y, D_weights):
    stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
    stump = stump.fit(X, Y, sample_weight=D_weights)
    return stump
##################################################################################
def trainWeakGrad(X, Y, D_weights):
    stump = GradientBoostingClassifier(random_state=0)
    stump = stump.fit(X, Y, sample_weight=D_weights)
    return stump
##################################################################################
def updateDt(h_t,D_weights,stump_weight,Y):
        new_sample_weights = (
            D_weights * np.exp(-stump_weight * Y * h_t)
        )
        new_sample_weights /= new_sample_weights.sum()
        return new_sample_weights
##################################################################################
def predict(stumps,stump_weights, X):
    """ Make predictions using already fitted model """
    stump_preds = np.array([stump.predict(X) for stump in stumps])
    return np.sign(np.dot(stump_weights, stump_preds))
##################################################################################






#Load Data
(xtrain,ytrain),(xtest,ytest)=tf.keras.datasets.mnist.load_data()
#1 is even -1 is odd
#lambda because lambda functions are cool >:)
OddEvenMapping=lambda x: 1 if x%2 ==0 else -1
xtrain=np.array([np.concatenate(xi)for xi in xtrain])
xtest = np.array([np.concatenate(xi)for xi in xtest])
#normalize Data
xtrain=xtrain/255
xtest=xtest/255
ytrain=np.array([OddEvenMapping(xi) for xi in ytrain])
ytest=np.array([OddEvenMapping(xi) for xi in ytest])

########################################################################
#Error of Decision tree vs Gradient Booster vs SkLearn
model_params=adaboost(xtrain,ytrain,7,'Decision_tree')  
pred=predict(model_params[0],model_params[1],xtest)
train_errTree = (pred != ytest).mean()
print(f'Train error Tree: {train_errTree:.1%}')

clf = AdaBoostClassifier(n_estimators=7, random_state=0)
clf.fit(xtrain, ytrain)
pred=clf.predict(xtest)
train_errSKL = (pred != ytest).mean()
print(f'Train error SKlearn: {train_errSKL:.1%}')

model_params=adaboost(xtrain,ytrain,7,'Gradient_Boost')
pred=predict(model_params[0],model_params[1],xtest)
train_errGrad = (pred != ytest).mean()

print(f'Train error GradientBoost: {train_errGrad:.1%}')

#######################################################################
#just some graphing of the boosting process
"""
#Error over boosting process
print("Tree Boosting Begining:")
train_errTrees=np.zeros(shape=8)
for x in range(1,7):
    model_params=adaboost(xtrain,ytrain,xtest,ytest,x)
    
    pred=predict(model_params[0],model_params[1],xtrain)
    train_errTrees[x] = (pred != ytrain).mean()
    print(f'Train error: {train_errTrees[x]:.1%}')
    
    plt.plot(train_errTrees[1:7])
#this takes a really long time to run but still is interesting
   
print("Gradient Boosting Begining:")  
train_errGrads=np.zeros(shape=8)
plt.plot(train_errTrees[1:7])
for x in range(1,7):
    model_params=adaboostGrad(xtrain,ytrain,xtest,ytest,x)
    
    pred=predict(model_params[0],model_params[1],xtrain)
    train_errGrads[x] = (pred != ytrain).mean()
    print(f'Train error: {train_errGrads[x]:.1%}')
    
plt.plot(train_errGrads[1:7])
"""