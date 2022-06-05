# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb

class ARMA():

    def __init__(self,p = 1,q = 1):

        self.p = p
        self.q = q
        self.i = 0
        self._data = None

        self.InitParams()

    def InitParams(self):

        self._params = np.zeros(( 1 + self.p + self.q,1))

    def SetData(self, data):

        self._data = data

    @property
    def params(self):
        return self._params

    @property
    def data(self):
        return self._data

    def GetState(self,t, obs = None):

        if obs == None:
            observation = self.data[t]
        else:
            observation = obs

        self.y_pred = self._predict()
        self.predictions = np.roll(self.predictions,-1)
        self.predictions[-1] = self.y_pred

        self.errors = np.roll(self.errors,-1)
        self.errors[-1] = self.y_pred - observation

        self.state = np.concatenate((np.ones((1,1)), self.predictions[-self.q:], self.errors) )




    def InitLS(self, gamma = 0.999, delta = 0.1):

        self.order = np.max((self.q,self.p))
        self._data = np.concatenate((np.zeros((self.order,1)),self.data))

        self.predictions = np.zeros((self.order,1))
        self.errors = np.zeros((self.p,1))


        self.state = np.zeros((self.q + self.p + 1,1))


        self.GainMatrix = np.eye(1 + self.q + self.p) * delta
        self.gamma = gamma

    def _predict(self):
        return self.params.T @ self.state

    def update(self,t):

       y_hat = self._predict()
       y = self.data[t]

       alpha = y - y_hat
       g = self.GainMatrix @ self.state /(self.gamma + self.state.T@self.GainMatrix@self.state)
       self.GainMatrix = 1/self.gamma *(self.GainMatrix - g @ self.state.T @ self.GainMatrix)

       self._params = self.params + alpha*g

       self.GetState(t)

       return y_hat


    def fit(self):
        # assert self.data != None


        self.InitLS()

        self.prediction = np.zeros_like(self.data[self.order:])


        for t in range(self.order,data.shape[0] + 2 - self.i):
            y = self.update(t)
            self.prediction[t-self.order] = y.squeeze()


        return self.prediction

    def OnlineUpdate(self,obs):
        y_hat = self._predict()
        y = obs

        alpha = y - y_hat
        g = self.GainMatrix @ self.state / (self.gamma + self.state.T @ self.GainMatrix @ self.state)
        self.GainMatrix = 1 / self.gamma * (self.GainMatrix - g @ self.state.T @ self.GainMatrix)

        self._params = self.params + alpha * g

        self.GetState(t,obs)

    def OnlineNoUpdate(self,h):

        errors = np.flip(self.errors)

        predictions = np.empty((h,1))

        for t in range(h):

            horizon = self.p - np.min((t,self.p))

            error = np.flip(errors[:horizon])

            state = np.concatenate((np.ones((1, 1)), self.predictions[-self.q:], self.errors))


class ARIMA(ARMA):

    def __init__(self,p = 1,q = 1, i = 0):
        super(ARIMA, self).__init__(p,q)
        self.i = i

    def SetData(self, data):
        if self.i == 0:
            super().SetData(data)

        self.unintegratedData = data

        data = np.diff(data,self.i,axis=0)

        self._data = np.atleast_2d(data)

    def GetState(self,t, obs = None):
        if obs == None:
            observation = self.data[t]
        else:
            observation = obs

        self.y_pred = self._predict()
        self.predictions = np.roll(self.predictions,-1)
        self.predictions[-1] = self.y_pred

        update = self.cumsum
        self.cumsum += self.y_pred
        self.errors = np.roll(self.errors,-1)
        self.errors[-1] = self.cumsum - self.unintegratedData[t]



        self.state = np.concatenate((np.ones((1,1)), self.predictions[-self.q:], self.errors) )


    def fit(self):

        self.cumsum = np.atleast_2d(self.unintegratedData[:self.i]).cumsum()
        first_parts = []
        for i in range(self.i):
            first_parts.append(np.diff(self.unintegratedData[:i+1],i,axis=0))

        self.firstparts = np.flip(np.atleast_2d(np.array(first_parts).squeeze()).T)

        integrated = self.data
        for i in range(self.i):
            integrated = np.atleast_2d(np.concatenate((np.atleast_2d(self.firstparts[i]),integrated),axis=0).cumsum()).T

        y_pred = super().fit()

        if self.i > 0:
            y_pred = np.atleast_2d(np.concatenate((self.unintegratedData[:self.i], y_pred)).cumsum()).T
        return y_pred









if __name__ == '__main__':

    T = 1000
    x = 0
    data = np.empty((T,1))
    for t in range(T):
        data[t] =   t**2 + 1*np.random.randn()
        x = data[t]

    arma = ARIMA(2,2,2)
    arma.SetData(data)
    # plt.plot(arma.integratedData)
    # plt.show()
    predi = arma.fit()


    from matplotlib import pyplot as plt

    # plt.plot(predi, label='est')
    # plt.plot(arma.data, label='GT')
    # plt.plot(data)

    # plt.plot(predi, label = 'est')
    # plt.plot(data,label = 'GT')

    # plt.plot(predi-data, label = 'residual')
    # plt.legend()
    # plt.show()

    # print((data-predi).mean())


