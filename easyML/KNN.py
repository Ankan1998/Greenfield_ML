import numpy as np
class KNN:

    def __init__(self,k=3):
        # Need to specified the number of K default = 3
        self.k=k
    
    def fit(self,X,y):
        # Training on include giving info of training dataset
        self.X_train=X
        self.y_train=y
        
    def predict(self,X):
        y=np.zeros(len(X))
        for idx_of_test, x in enumerate(X): # iterating over test set
            distance_list_of_test_train=[] # empty list to include test_train distance
            for idx_of_train , x_t in enumerate(self.X_train): # iterating over training set data
                difference=x-x_t # next two lines of code calculate distance
                sqrd_dist=difference.dot(difference)
                distance_list_of_test_train.append(sqrd_dist) # appending sqrd distance to distance list
            k_idx=np.argsort(distance_list_of_test_train)[:self.k] #argsort sorts from asc to desc returns index in position
            k_nearest_lbl=[self.y_train[train_y_idx] for train_y_idx in k_idx] #labelling according its index of y_train
            y[idx_of_test]=max(set(k_nearest_lbl), key = k_nearest_lbl.count) # getting max occurences of classes in the list
        return y
