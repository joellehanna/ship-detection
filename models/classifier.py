from abc import abstractmethod


class Classifier:
    r"""
    A classifier interface for the different image classifier
    """
    @abstractmethod
    def call(self):
        pass
