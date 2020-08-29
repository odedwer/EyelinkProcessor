from abc import ABC, abstractmethod


class BaseSaccadeDetector(ABC):
    """
    This is a base class for saccade detector classes.
    One example is the microsaccade detection algorithm of Engert & Morgenthaler 2006
    """
    @classmethod
    @abstractmethod
    def detect_saccades(cls, saccade_data, sf):
        """
        detects saccades/microsaccads in the given data
        :param saccade_data: time X Position (X,Y) matrix
        :param sf: sampling frequency
        :return: dataframe of detected saccades
        """
        pass
