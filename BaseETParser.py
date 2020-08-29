from abc import ABC, abstractmethod


class BaseETParser(ABC):
    """
    Base class for ET parsers, cannot be instantiated (abstract). To use, inherit from this class and override methods
    """

    # there might be a need to define more "must have" properties such as this
    blink = False

    @property
    def TIME(self):
        """
        key string for time column in samples DF
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def get_type(self):
        """
        Return the type of the parser - Left, Right or Both eyes
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def is_blink(cls):
        """
        return True if the blink property is True
        """
        pass

    @classmethod
    def toggle_blink(cls):
        """
        Toggle the blink property - from True to False and from False to True
        """
        cls.blink = not cls.blink

    @classmethod
    @abstractmethod
    def get_empty_sample(cls, time):
        """
        :param time: time for the empty sample
        :return: dict  representing an empty sample (MISSING_VALUE in all columns), with a timestamp of the given time
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def parse_sample(cls, line):
        """
        parses a sample from the EDF, returns a dictionary that will be a line in the DataFrame
        """
        pass

    @classmethod
    @abstractmethod
    def parse_msg(cls, line):
        """
        parses a message line from the EDF, returns a dictionary that will be a line in the DataFrame
        """
        pass

    @classmethod
    @abstractmethod
    def parse_input(cls, line):
        """
        parses a trigger line from the EDF, returns a dictionary that will be a line in the DataFrame
        """
        pass

    @classmethod
    @abstractmethod
    def parse_fixation(cls, line):
        """
        parses a fixation line from the EDF, returns a dictionary that will be a line in the DataFrame
        """
        pass

    @classmethod
    @abstractmethod
    def parse_saccade(cls, line):
        """
        parses a saccade line from the EDF, returns a dictionary that will be a line in the DataFrame
        """
        pass

    @classmethod
    @abstractmethod
    def parse_blinks(cls, line):
        """
        parses a blink line from the EDF, returns a dictionary that will be a line in the DataFrame
        """
        pass

    @classmethod
    @abstractmethod
    def parse_recordings(cls, line):
        """
        parses a recording start/end line from the EDF, returns a dictionary that will be a line in the DataFrame
        """
        pass

    @classmethod
    @abstractmethod
    def is_sample(cls, line):
        """
        checks if a line is a sample line
        :param line: line to check
        :return: True if line is sample, else False
        """
        pass
