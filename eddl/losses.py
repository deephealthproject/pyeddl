class Loss:
    def __init__(self, name):
        self.name = name


class MeanSquaredError(Loss):
    """Mean Squared Error
    """

    def __init__(self):
        super(MeanSquaredError, self).__init__('mean_squared_error')


class CategoricalCrossEntropy(Loss):
    """Categorical Cross-Entropy
    """

    def __init__(self):
        super(CategoricalCrossEntropy, self).__init__('categorical_crossentropy')


class CategoricalSoftCrossEntropy(Loss):
    """Categorical Soft Cross-Entropy
    """

    def __init__(self):
        super(CategoricalSoftCrossEntropy, self).__init__('categorical_soft_crossentropy')
