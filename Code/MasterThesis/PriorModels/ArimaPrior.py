from PriorModels.BasePrior import BasePrior
from statsmodels.tsa.arima.model import ARIMA

class ArmaPrior(BasePrior):

    def __init__(self,order = (1,1,1)):

        super(ArmaPrior, self).__init__()

        self.order = order

    def fit(self,data):

        model = ARIMA(data, order=self.order)
        self.model = model.fit()







