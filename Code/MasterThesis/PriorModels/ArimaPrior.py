import matplotlib.pyplot as plt

from PriorModels.BasePrior import BasePrior
from statsmodels.tsa.arima.model import ARIMA

class ArimaPrior(BasePrior):

    def __init__(self,**kwargs):

        super(ArimaPrior, self).__init__()

        self.order = kwargs['order'] if 'order' in kwargs.keys() else (5,0,0)

    def fit(self,data):

        sample,self.channels, self.T = data.shape
        data = data[0]


        self.models = []

        for channel in range(self.channels):

            model = ARIMA(data[channel].numpy(), order=self.order)
            result = model.fit(method='yule_walker')
            self.models.append(result)



        1

    # def f(self,x,t):





    # def getSysModel(self):









