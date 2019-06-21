class Metric:
    def __init__(self, name):
        self.name = name


class MeanSquaredError(Metric):
    """Mean Squared Error
        """
    def __init__(self):
        super(MeanSquaredError, self).__init__('mean_squared_error')


class CategoricalAccuracy(Metric):
    """Categorical Accuracy
    """

    def __init__(self):
        super(CategoricalAccuracy, self).__init__('categorical_accuracy')
