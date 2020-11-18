import numpy as np

from BaseSaccadeDetector import BaseSaccadeDetector

NUM_OF_COLUMS_ERROR_MSG = "Saccade data should have 2 columns - X, Y positions, but has %d columns"
INPUT_SHAPE_ERROR_MSG = "Shape of saccade data is incorrect! should be a 2D matrix yet has %d dimensions"


class EngbertAndMergenthalerMicrosaccadeDetector(BaseSaccadeDetector):
    NOISE_THRESHOLD_LAMBDA = 8

    @classmethod
    def detect_saccades(cls, eye_location, sf):
        """
        see documentation in BaseSaccadeDetector
        """
        # validate input
        eye_location = np.asarray(eye_location)
        if len(eye_location.shape) != 2:
            raise Exception(INPUT_SHAPE_ERROR_MSG % len(eye_location.shape))
        if eye_location.shape[1] != 2:
            raise Exception(NUM_OF_COLUMS_ERROR_MSG % eye_location.shape[1])
        velocities = cls._get_velocities(eye_location, sf)
        msdx, msdy = cls._get_SD_thresholds(velocities)
        radius_x = cls.NOISE_THRESHOLD_LAMBDA * msdx
        radius_y = cls.NOISE_THRESHOLD_LAMBDA * msdy

        # threshold the data
        thresholded_data = (velocities[:, 0] / radius_x) ** 2 + (velocities[:, 1] / radius_y) ** 2
        thresholded_data = thresholded_data > 1

        # calculate row numbers of saccade starts
        saccade_start_indices = cls._get_saccade_start_indices(thresholded_data)
        event_vector = np.zeros((eye_location.shape[0],), dtype=int)
        event_vector[saccade_start_indices] = 1  # 1 in every saccade start, 0 elsewhere
        return event_vector

    @classmethod
    def _get_saccade_start_indices(cls, thresholded_data):
        possible_saccade_indices = np.nonzero(thresholded_data)[0]  # all non-zero indices
        consecutive_saccade_indices = np.diff(possible_saccade_indices) == 1  # only consecutive indices
        # only indices in which 2 previous indices are detected saccades
        detected_saccades_indices = np.insert((consecutive_saccade_indices[:-1:] & consecutive_saccade_indices[1::]), 0,
                                              [False, False])
        # get run length of 0 and 1's (for [1,1,1,0,0,1,1,1,1] we'll get [3,4] and [2])
        non_zero = np.nonzero(np.diff(detected_saccades_indices) != 0)[0]
        diff = np.diff(non_zero)
        saccade_run_lengths = diff[::2]  # from first element, in jumps of 2 are 1 sequences lengths
        no_saccade_run_lengths = diff[1::2]  # the rest
        first_saccade_index = np.argmax(detected_saccades_indices)  # index of first 1
        # calculate saccade starts - first_saccade_index, then cumsum for length up to next sequence,
        # which is length of 1 run + length of 0 run
        saccade_starts = np.cumsum(
            np.hstack([[first_saccade_index],
                       saccade_run_lengths + no_saccade_run_lengths]))
        saccade_start_indices = possible_saccade_indices[saccade_starts]
        return saccade_start_indices

    @staticmethod
    def _get_velocities(eye_location, sf):
        """
        calculates the velocities of the given data
        :param eye_location: X,Y locations of the eye
        :param sf: sampling frequency
        :return: 2 x (X,Y) numpy array of x,y velocities
        """
        velocities = np.zeros((eye_location.shape[0], 2))
        velocities[3:-2, :] = (sf / 6.) * (
                eye_location[5:, :] + eye_location[4:-1, :] - eye_location[2:-3, :] - eye_location[1:-4, :])
        velocities[2, :] = (sf / 2.) * (eye_location[3, :] - eye_location[1, :])
        velocities[-2, :] = (sf / 2.) * (eye_location[-1, :] - eye_location[3, :])
        return velocities

    @staticmethod
    def _get_SD_thresholds(velocities):
        """
        calculates the median based SD estimators for X,Y velocities
        :param velocities: X,Y velocities
        :return: tuple of 2 elements, median-SD-x, median-SD-y estimators
        """
        msdx = np.sqrt((np.median(velocities[:, 0] ** 2)) - (np.median(velocities[:, 0]) ** 2))
        msdy = np.sqrt((np.median(velocities[:, 1] ** 2)) - (np.median(velocities[:, 1]) ** 2))
        return msdx, msdy