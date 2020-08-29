from enum import Enum
from SaccadeDetectors import *
from EyeTrackingParsers import *


class SaccadeDetectorType(Enum):
    """
    enum class for saccade detectiors. Values should be relevant class names.
    """
    ENGBERT_AND_MERGENTHALER = EngbertAndMergenthalerMicrosaccadeDetector


class ParserType(Enum):
    EL_BIN_NO_VELOCITY = BinocularNoVelocityParser


class Eye(Enum):
    """
    Enum for eye - left, right or both
    """
    RIGHT = 1
    LEFT = 2
    BOTH = 3
