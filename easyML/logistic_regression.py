import numpy as np
class Logistic_Regression:
    
    
    def __init__(self,learning_rate=0.01,epochs=500):
        self.learning_rate=learning_rate
        # Epochs is the number to time the descent will take place
        self.epochs=epochs
        self.weights=None
        self.bias=None
        
    def sigmoid(self,x):
        # sigmoid function for logistic regression
        return 1/(1+np.exp(-x))
        
        
    def predicted_val(self,Xi,W,b):
        # Xi=(num_of_samples,num_of_features);W=(num_of_features,1);b=(num_of_samples,1)
        # Dimension of ypred --> (num_of_samples,1)
        linearPred=np.dot(Xi,W) + b
        # Passing the prediction through sigmoid
        # To simply explain, is to bring down value between 0 and 1
        ypred=self.sigmoid(linearPred) 
        return ypred
        
    def fit(self,X,y):
        num_of_samples,num_of_features=X.shape
        # reshape y because if y is a rank matrix -->(num_of_samples,)
        # Then there will be a issues in dimension
        # For better understanding. visit this link: https://www.kaggle.com/getting-started/176510
        y=y.reshape(num_of_samples,1)
        # weight vector is of dimension (num_of_features,1)
        self.weights= np.random.randn(num_of_features,1)
        # Bias is (1,1)
        self.bias=np.random.randn(1,1)
        # Looping over number of epochs
        for i in range(self.epochs):
            # dW is the derivative of the cost function 
            # Cost function is the binary or sigmoid cross-entropy −(ylog(p)+(1−y)log(1−p))
            # p=predicted
            dW=np.dot(X.T,(self.predicted_val(X,self.weights,self.bias)-y))/(2*(num_of_samples))
            db=np.sum(self.predicted_val(X,self.weights,self.bias)-y)/(2*(num_of_samples))
            
            # Updating weights and bias
            self.weights=self.weights-self.learning_rate*dW
            self.bias=self.bias-self.learning_rate*db
            
    def predict(self,X):
        lin_pred_y=np.dot(X,self.weights) + self.bias
        y_pred_sig=self.sigmoid(lin_pred_y) 
        # if any value greater than 0.5 in y_pred_sig then 1 else 0 
        y_predicted = [1 if i > 0.5 else 0 for i in y_pred_sig]
        return np.array(y_predicted)