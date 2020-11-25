# AdaboostOnMNIST
This is an implementation of the Adaboost algorithm from scratch using two different weak learners: Decision Tree Classifiers and Gradient Boost Classifiers. Adaboost run on MNIST to tell odd vs even numbers. Tested against scikit Learn model for adaboost and got a better score. Lowest training error was %1.8 with gradient boosting on 7 iterations.

The function call is adaboost(X_train, Y_train, iterations_t, Classifier_type) there are two types of classifiers, 'Gradient_Boost' and 'Decision_tree' that can be put into the 4th input. 

adaboost returns a 4 tuple (stumps, stump_weights, errors, D_weights)

You can predict a set against a training set using predict(stumps, stump_weights, X_test). This will return an array of labels for that X_test
