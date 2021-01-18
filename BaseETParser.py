from abc import ABC, abstractmethod

from Enums import *


class BaseETParser(ABC):
    """
    Base class for ET parsers, cannot be instantiated (abstract). To use, inherit from this class and override methods
    """

    blink = False

    @property
    def LINE_SPLIT_PATTERN(self) -> re.Pattern:
        """
        The pattern to split a line by
        """
        raise NotImplementedError

    @property
    def RIGHT_X(self) -> str:
        """
        key string for time column in samples DF
        """
        raise NotImplementedError

    @property
    def RIGHT_Y(self) -> str:
        """
        key string for time column in samples DF
        """
        raise NotImplementedError

    @property
    def LEFT_X(self) -> str:
        """
        key string for time column in samples DF
        """
        raise NotImplementedError

    @property
    def LEFT_Y(self) -> str:
        """
        key string for time column in samples DF
        """
        raise NotImplementedError

    @property
    def START_TIME(self) -> str:
        """
        key string for time column in samples DF
        """
        raise NotImplementedError

    @property
    def TIME(self) -> str:
        """
        key string for time column in samples DF
        """
        raise NotImplementedError

    @property
    def MISSING_VALUE(self) -> int:
        """
        int representing missing values
        """
        raise NotImplementedError

    @property
    def MISSING_VALUES(self) -> list:
        """
        list containing the values in an unparsed line which should be replaced
        by MISSING_VALUE
        """
        raise NotImplementedError

    @property
    def parse_line_by_token(self) -> dict:
        """
        dictionary with tokens as keys (the first token in a line) and
        corresponding parse method as the value
        """
        raise NotImplementedError

    @property
    def PARSER_EYE_TYPE(self) -> Eye:
        """
        Return the type of the parser - Eye.LEFT, Eye.RIGHT or Eye.BOTH eyes
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def is_blink(cls) -> bool:
        """
        return True if the blink property is True
        """
        pass

    @classmethod
    def toggle_blink(cls) -> None:
        """
        Toggle the blink property - from True to False and from False to True
        """
        cls.blink = not cls.blink

    @classmethod
    @abstractmethod
    def get_empty_sample(cls, time: float) -> dict:
        """
        :param time: time for the empty sample
        :return: dict representing an empty sample
        (MISSING_VALUE in all columns), with a timestamp of the given time
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def replace_missing_values(cls, line: list) -> None:
        """
        return True if the blink property is True
        """
        pass

    @classmethod
    @abstractmethod
    def parse_sample(cls, line) -> dict:
        """
        parses a sample from the EDF, returns a dictionary that will be a
        line in the DataFrame
        """
        pass

    @classmethod
    @abstractmethod
    def parse_msg(cls, line) -> dict:
        """
        parses a message line from the EDF, returns a dictionary that will be
        a line in the DataFrame
        """
        pass

    @classmethod
    @abstractmethod
    def parse_input(cls, line) -> dict:
        """
        parses a trigger line from the EDF, returns a dictionary that will be
        a line in the DataFrame
        """
        pass

    @classmethod
    @abstractmethod
    def parse_fixation(cls, line) -> dict:
        """
        parses a fixation line from the EDF, returns a dictionary that will be
        a line in the DataFrame
        """
        pass

    @classmethod
    @abstractmethod
    def parse_saccade(cls, line) -> dict:
        """
        parses a saccade line from the EDF, returns a dictionary that will be
        a line in the DataFrame
        """
        pass

    @classmethod
    @abstractmethod
    def parse_blinks(cls, line) -> dict:
        """
        parses a blink line from the EDF, returns a dictionary that will be a
        line in the DataFrame
        """
        pass

    @classmethod
    @abstractmethod
    def parse_recordings(cls, line) -> dict:
        """
        parses a recording start/end line from the EDF, returns a
        dictionary that will be a line in the DataFrame
        """
        pass

    @classmethod
    @abstractmethod
    def is_sample(cls, line) -> bool:
        """
        checks if a line is a sample line
        :param line: line to check
        :return: True if line is sample, else False
        """
        pass
