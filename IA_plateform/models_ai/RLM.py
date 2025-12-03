import numpy as np

class RLM:

    def __init__(self,lr,ite,scaler):
        self.lr = lr
        self.ite = ite
        self.theta = None
        self.scaler = scaler
        self.x_scaled = None

    def scaling(self,x):
        ss = self.scaler
        x = np.array(x).reshape(1,-1)
        self.x_scaled = ss.transform(x)

    def predict(self,x):
        y = x.dot(self.theta)
        return y
    
    def fonction_cout(self,x,y):
        m = len(y)
        pred = self.predict(x)
        erreur = pred - y
        cout = (1/(2*m)) * ((erreur.T).dot(erreur))
        return cout[0][0]
    
    def fit(self,x,y):
        m = len(y)
        self.theta = np.random.randn(x.shape[1],1)
        for i in range(self.ite):
            pred = self.predict(x)
            erreur = pred - y
            derive = (1/m) * (x.T.dot(erreur))
        
            self.theta = self.theta - self.lr * derive