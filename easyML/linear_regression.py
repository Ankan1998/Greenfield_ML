import numpy as np
class lin_reg:


	def __init__(self,learning_rate=0.001,epochs=100,alpha=0.001):
		self.learning_rate=learning_rate
		self.epochs=epochs
		self.aplha=alpha
		self.weights=None
		self.bias=None


	def predicted_val(self,X,W,b):
		ypred = np.dot(X,W) + b
		return ypred


	def fit(self,x,y):
		self.weights=np.random.randn(x.shape[1],1)
		self.weights=self.weights.reshape(x.shape[1],1)
		self.bias=np.random.randn(1,1)
		for i in range(self.epochs):
			db=sum(self.predicted_val(x,self.weights,self.bias)-y)/(2*len(y))
			dW=(x.T@(self.predicted_val(x,self.weights,self.bias)-y))/(2*len(y))
			self.bias=self.bias-self.alpha*db
			self.weights=self.weights-self.alpha*dW

	def predict(self,X):
		return np.dot(x,self.weights) + self.bias

