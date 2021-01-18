import typing
from os.path import exists, basename
from re import split
from sys import stderr

import numpy as np
import pandas as pd
from mne import find_events
from scipy.stats import pearsonr

import Enums as e
from BaseETParser import BaseETParser


class EyeLinkProcessor:
    """
    This class loads & parses Eyelink 1000 .asc data file.
    This class assumes the recording is binocular without velocity
    This class can extract ET events from the samples and synchronize to mne.Raw file by triggers
    """
    ET_SAMPLING_RATE_CONST = 1000
    NO_SACCADE_DETECTOR_MSG = 'Did not detect saccades as no saccade detector was ' \
                              'supplied to the ET processor. ' \
                              'Please set processor._saccade_detector to an ' \
                              'object of a class that inherits ' \
                              'from BaseSaccadeDetector, before calling ' \
                              'detect_saccades()'

    def __init__(self, filename: str, parser_type: e.ParserType,
                 saccade_detector: e.SaccadeDetectorType = None,
                 max_sampling_rate=1000., block_start_trigger=254,
                 block_end_trigger=255,
                 smoothing_factor=10, blink_event_trigger=256,
                 saccade_event_trigger=257):
        """
        initialize a parser
        :param filename: path of the .asc file to parse
        """
        self.max_sampling_rate = max_sampling_rate
        self.block_start_trigger = block_start_trigger
        self.block_end_trigger = block_end_trigger
        self.smoothing_factor = smoothing_factor
        self.blink_event_trigger = blink_event_trigger
        self.saccade_event_trigger = saccade_event_trigger
        self.name = basename(filename)[:-4]
        self._is_synced = False
        self._filename = filename
        self._init_lists()
        self._parser: BaseETParser = parser_type.value  # enum, get class from enum value
        # method dictionary for parsing
        self._parse_line_by_token = {'INPUT': self._parse_input,
                                     'MSG': self._parse_msg,
                                     'ESACC': self._parse_saccade,
                                     'EFIX': self._parse_fixation,
                                     "EBLINK": self._parse_blinks}
        self._dt = None
        self._parse_et_data()  # parse the file
        # get sampling frequency
        self._sf = EyeLinkProcessor.max_sampling_rate / (
                self._samples.loc[1, self._parser.TIME] - self._samples.loc[
            0, self._parser.TIME])
        self._saccade_detector = saccade_detector.value  # enum, get class from enum value
        self._detected_saccades = None
        if self._saccade_detector:
            self.detect_saccades()
        self._eeg_index = None

    def _init_lists(self):
        self._samples = []
        self._messages = []
        self._triggers = []
        self._fixations = []
        self._saccades = []
        self._blinks = []
        self._recording_data = []

    #################################################
    #               'public' methods                #
    #################################################
    def sync_to_raw(self, raw) -> None:
        """
        Creates an ET-EEG index vector for ET samples in self._eeg_index -
        a vector with length of _samples.shape[0]
        where self._eeg_index[i] is the sample number of the i'th ET sample in
        the given eeg raw file.
        self._eeg_index[i]=np.NaN where the ET has samples and the EEG doesn't
        based on Eden Gerber's Matlab function sync_eyetracker_samples_to_eeg
        :param raw: mne.io.Raw containing EEG data whose events can be extracted
        using mne.find_events
        """
        eeg_data_trig_ch, eeg_sf, eeg_trigs, et_samples, et_sf, et_timediff, \
        et_trigs, et_trig_ch = self._prepare_et_eeg_sync_data(raw)
        eeg_block_starts, eeg_blocks, et_block_starts, et_blocks = \
            self._split_to_recording_blocks(
                eeg_trigs,
                et_samples,
                et_timediff,
                et_trig_ch)
        self._resample_eeg_to_et(eeg_blocks, eeg_sf)
        matched_eeg_blocks, matched_et_blocks = self._match_eeg_et_blocks(
            eeg_blocks, et_blocks)
        eeg_block_starts = np.hstack(
            [eeg_block_starts[matched_eeg_blocks], [eeg_data_trig_ch.size]])
        et_block_starts = np.hstack(
            [et_block_starts[matched_et_blocks], [et_samples[-1, 0]]])
        resampling_factor = self._correct_sampling_rate_discrepancies(
            eeg_blocks, et_blocks, matched_eeg_blocks,
            matched_et_blocks)
        et_block_start_in_et_sample = self._get_et_block_starts_in_et_samples(
            et_block_starts, et_samples, et_trigs)
        self._generate_eeg_index(eeg_block_starts, eeg_sf,
                                 et_block_start_in_et_sample, et_samples, et_sf,
                                 et_blocks,
                                 matched_eeg_blocks, matched_et_blocks,
                                 resampling_factor)
        self._is_synced = True

    def detect_saccades(self) -> None:
        """
        detect saccades based on implemented algorithm in self._saccade_detector
        """
        if not self._saccade_detector:
            from sys import stderr
            print(self.NO_SACCADE_DETECTOR_MSG, file=stderr)
            return
        self._detected_saccades = []
        if self._parser.PARSER_EYE_TYPE == e.Eye.BOTH or \
                self._parser.PARSER_EYE_TYPE == e.Eye.LEFT:
            left_data = self._samples[
                [self._parser.LEFT_X, self._parser.LEFT_Y]]
            self._detected_saccades = self._saccade_detector.detect_saccades(
                left_data, self._sf)
        if self._parser.PARSER_EYE_TYPE == e.Eye.BOTH or \
                self._parser.PARSER_EYE_TYPE == e.Eye.RIGHT:
            right_data = self._samples[
                [self._parser.RIGHT_X, self._parser.RIGHT_Y]]
            # choose how to stack the data - if no saccades were added hstack,
            # else left eye was parsed so vstack
            stack_function = np.vstack if self._detected_saccades is not None \
                else np.hstack
            self._detected_saccades = stack_function(
                [self._detected_saccades,
                 self._saccade_detector.detect_saccades(right_data,
                                                        self._sf)]).T
        if self._parser.PARSER_EYE_TYPE == e.Eye.BOTH:
            self._combine_monocular_saccades_to_binocular()

    def get_synced_blinks(self) -> np.array:
        """

        :return: blink event times in eeg sample number. If binocular, assumes
        blinks happen in both eyes, and returns
        The times corresponding to the earliest blink time of the two eyes for
        each blink
        """
        return self._get_synced_et_data(
            np.array(self._blinks[self._parser.START_TIME]))

    def get_synced_saccades(self) -> np.array:
        return self._get_synced_et_data(
            np.array(self._saccades[self._parser.START_TIME]))

    def get_synced_detected_saccades(self) -> [np.array, None]:
        """
        Returns an array of eeg sample indices where saccades detected by the
        saccade detector happened. If detect_saccades() was not
        called, returns None
        """
        if not self._detected_saccades:
            return None
        ret = self._eeg_index[self._detected_saccades != 0]
        ret = ret[~np.isnan(ret)]
        return ret.astype(np.int)

    def get_eye_x_y_samples(self, eye=e.Eye.BOTH) -> pd.DataFrame:
        """
        returns the samples of the given eye
        :param eye: the eye to retrieve data for
        :type eye: Eye
        :return: pandas DataFrame containing time, x & y positions and pupil size per timepoint for given eye
        """
        if eye == e.Eye.RIGHT:
            return self._samples[
                [self._parser.TIME] + [col for col in self._samples.columns if
                                       "right" in col]]
        elif eye == e.Eye.LEFT:
            return self._samples[
                [self._parser.TIME] + [col for col in self._samples.columns if
                                       "left" in col]]
        elif eye == e.Eye.BOTH:
            return self._samples
        else:
            raise NotImplementedError(
                "%d Not supported. Please choose one of the following:\n" % eye +
                "".join(["Eye.%s ," % eye.name for eye in e.Eye]))

    #################################################
    #               'private' methods               #
    #################################################

    def _combine_monocular_saccades_to_binocular(self) -> None:
        saccade_indices_l = np.nonzero(self._detected_saccades[:, 0])[0]
        saccade_indices_r = np.nonzero(self._detected_saccades[:, 1])[0]
        diff_mat = np.repeat(saccade_indices_l,
                             saccade_indices_r.size).reshape(
            (saccade_indices_r.size, saccade_indices_l.size), order='F')
        diff_mat = np.abs(diff_mat - saccade_indices_r[:, np.newaxis])
        good_diffs = np.min(diff_mat, axis=1) < 20
        monocular_saccade_indices = np.unique(
            np.argmin(diff_mat, axis=1)[good_diffs])
        saccade_indices = saccade_indices_l[monocular_saccade_indices]
        saccade_indices = np.delete(saccade_indices,
                                    np.where(np.diff(saccade_indices) < 20)[
                                        0] + 1)
        self._detected_saccades = np.zeros_like(
            self._detected_saccades[:, 0], dtype=np.int)
        self._detected_saccades[saccade_indices] = 1

    def _get_synced_et_data(self, times: np.ndarray) -> np.array:
        if self._parser.PARSER_EYE_TYPE == e.Eye.BOTH:
            times = times[np.argsort(times)]  # sort
            times = times[::2]  # get lowest of each pair
        samples = np.asarray(self._samples[self._parser.TIME])
        array_event_positions = np.searchsorted(samples, times)
        return_array = self._eeg_index[array_event_positions]
        return np.sort(return_array[~np.isnan(return_array)])

    @staticmethod
    def _get_block_correlation(eeg_block, et_block) -> float:
        min_len = min(eeg_block.size, et_block.size)
        eeg = eeg_block[:min_len]
        et_block = et_block[:min_len]
        return pearsonr(eeg, et_block)[0]

    @staticmethod
    def _find_common_sequential_elements(a, b) -> typing.Tuple[
        np.array, np.array, np.array]:
        """
        find the common elements in sequential order
        :param a: 1d vector
        :param b: 1d vector
        :return: common elements, indices of common elements in a,  indices of common elements in b
        """
        a = np.asarray(a)
        b = np.asarray(b)
        min_len_arr, max_len_arr = (a, b) if a.size < b.size else (b, a)
        j = 0
        min_arr_idx, max_arr_idx = [], []
        for i in range(min_len_arr.size):
            if min_len_arr[i] == max_len_arr[j]:  # match on i,j
                min_arr_idx.append(i)
                max_arr_idx.append(j)
                j += 1
                continue
            else:
                # check if the element is in the rest of the array
                isin = np.isin(max_len_arr[j:], min_len_arr[i])
                if np.sum(isin) > 0:  # if it is
                    idx = np.argmax(isin)
                    min_arr_idx.append(i)
                    max_arr_idx.append(j + idx)
                    j += idx + 1  # update to next sequential possible common element
            if j >= max_len_arr.size:
                break
        min_arr_idx = np.asarray(min_arr_idx)
        max_arr_idx = np.asarray(max_arr_idx)
        return (max_len_arr[max_arr_idx], min_arr_idx,
                max_arr_idx) if a.size < b.size else (
            max_len_arr[max_arr_idx], max_arr_idx, min_arr_idx)

    def _split_to_recording_blocks(self, eeg_trigs, et_samples, et_timediff,
                                   et_trig_ch) -> typing.Tuple[
        np.array, np.array, np.array, np.array]:
        # split eeg to blocks based on 254 start record value
        eeg_block_starts = np.where(eeg_trigs == self.block_start_trigger)[0]
        eeg_blocks = np.split(eeg_trigs, eeg_block_starts)
        eeg_blocks = [np.squeeze(block) for block in eeg_blocks if
                      block[0] == self.block_start_trigger and np.sum(
                          block) > 0]
        # split et to blocks based on 254 start record value and 255 stop
        et_block_starts = np.hstack(
            [np.where(et_trig_ch == self.block_start_trigger)[0],
             [len(et_trig_ch)]])
        et_block_pause = np.where(et_trig_ch == self.block_end_trigger)[0]
        et_recording_stop = np.hstack(
            [np.where(np.diff(et_samples[:, 0]) > et_timediff)[0],
             et_samples[-1, 0]])
        et_blocks = self._get_et_blocks(et_block_pause, et_block_starts,
                                        et_recording_stop, et_samples,
                                        et_trig_ch)
        return eeg_block_starts, eeg_blocks, et_block_starts, et_blocks

    def _generate_eeg_index(self, eeg_block_starts, eeg_sf,
                            et_block_start_in_et_sample, et_samples, et_sf,
                            et_blocks,
                            matched_eeg_blocks, matched_et_blocks,
                            resampling_factor) -> None:
        """
        for each matched block pair get the EEG
        sample index which corresponds to each ET sample, and record this index
        on a vector which is the same length as the ET data. NaN values indicate
        that the respective ET sample is not mapped to any recorded EEG epoch.
        """
        self._eeg_index = np.full(et_samples.shape[0], np.NaN)
        for b in range(matched_eeg_blocks.size):
            block_resample_factor = (et_sf / eeg_sf) * resampling_factor[b]
            et_start = int(et_block_start_in_et_sample[b])
            et_len = int(
                np.floor(et_blocks[matched_et_blocks[b]].size * et_sf / 1000))
            eeg_start = eeg_block_starts[b]
            eeg_len = np.floor((eeg_block_starts[b + 1] - eeg_block_starts[
                b]) * block_resample_factor)
            length = int(
                min(eeg_len,
                    et_len))  # the matched time points can only go as long as the shorter recording block
            # but in any case not longer than the length of the ET sample matrix
            length = int(min(length, et_samples.shape[0] - et_start))
            eeg_sample_indexes = np.round(
                np.arange(0, length, 1) / block_resample_factor).astype(np.int)
            self._eeg_index[
            et_start:et_start + length] = eeg_sample_indexes + eeg_start

    @staticmethod
    def _get_et_block_starts_in_et_samples(et_block_starts, et_samples,
                                           et_trigs) -> np.array:
        """
        The variable et_block_start holds the timestamp of the start of each
        block in absolute time since the beginning of the recording, but now we
        need these time points in the form of et_sample times.
        This is not the same since et_sample is not continuous but rather skips
        epochs which were not recorded.
        """
        et_block_start_in_et_sample = np.zeros(et_block_starts.size)
        for i in range(et_block_starts.size):
            ind = np.where(et_samples[:, 0] == et_block_starts[i])[0]
            if ind.size == 0 and min(et_samples[0, 0], et_trigs[0, 0]) == \
                    et_trigs[0, 0]:
                print(
                    'start recording code 254 was sent before that actual START_RECORDING of ET, need to fix... ',
                    file=stderr)
            if ind.size == 0:
                ind = np.where(et_samples[:, 0] == et_block_starts[i] + 1)[0]
            if ind.size == 0:
                print(f'check block {i}', file=stderr)
                continue
            et_block_start_in_et_sample[i] = ind[0]
        return et_block_start_in_et_sample

    def _correct_sampling_rate_discrepancies(self, eeg_blocks, et_blocks,
                                             matched_eeg_blocks,
                                             matched_et_blocks) -> np.array:
        """
        Correct sampling-rate discrepancies
        Due to different clocks, sampling rates after the initial re-sampling are
        still not identical, leading to cumulative discrepancies (can be on the
        order of 1 ms discrepancy per 10 seconds of recording).
        This is corrected by comparing the latency of matching events at the end
        of each block and re-sampling the EEG block once again.
        """
        resampling_factor = np.zeros(matched_eeg_blocks.size)
        for b in range(resampling_factor.size):
            eeg_block = eeg_blocks[matched_eeg_blocks[b]]
            et_block = et_blocks[matched_et_blocks[b]]
            eeg_latencies = np.where(eeg_block != 0)[0]
            et_latencies = np.where(et_block != 0)[0]
            block_eeg_trigs = eeg_block[eeg_latencies]
            block_et_trigs = et_block[et_latencies]
            _, eeg_common_idx, et_common_idx = self._find_common_sequential_elements(
                block_eeg_trigs, block_et_trigs)
            found = False
            i = 1
            while not found:  # need to make sure that the matched event is indeed from
                # about the same time in both blocks
                time_diff = eeg_latencies[eeg_common_idx[-i]] - et_latencies[
                    et_common_idx[-i]]
                if abs(time_diff) > 100:
                    i = i + 1  # move back one event
                else:
                    total_time1 = eeg_latencies[eeg_common_idx[-i]] - \
                                  eeg_latencies[eeg_common_idx[0]]
                    total_time2 = et_latencies[et_common_idx[-i]] - \
                                  et_latencies[et_common_idx[0]]
                    resampling_factor[b] = total_time2 / total_time1
                    found = True
                # If it's hard to find such an event, something is wrong:
                if i > np.ceil(len(eeg_common_idx) * 0.75):
                    problem = f'Matched EEG-ET recording block {b}: failed to fine-tune sampling rates of ET vs. EEG.'
                    print(problem, file=stderr)
                    break
            if found:
                eeg_latencies = np.round(
                    eeg_latencies * resampling_factor[b]).astype(np.int)
                eeg_latencies[eeg_latencies < 1] = 1
                eeg_block = np.zeros(eeg_block.size)
                eeg_block[eeg_latencies] = block_eeg_trigs
                eeg_blocks[matched_eeg_blocks[b]] = eeg_block
        return resampling_factor

    def _match_eeg_et_blocks(self, eeg_blocks, et_blocks) -> typing.Tuple[
        np.array, np.array]:
        """
        Find the best match between EEG and ET blocks
        This function can synchronize recordings where the EEG and ET can have a
        different number of blocks (e.g. if one recording started or stopped
        earlier than the other). This is done by going over all possible overlaps
        of the two block lists, and chosing the one with the highest total
        correlation between each pair of matched blocks.

        Create smoothed event time courses to decrease the correlations'
        sensitivity to event timing discrepancies. But first delete the first
        event in the block to prevent spuriously high correlations for blocks
        with only one event.
        """
        # smooth with convolution
        smoothed_eeg_blocks = list(
            map(lambda block: np.convolve(block[1:],
                                          np.ones(self.smoothing_factor),
                                          'same'), eeg_blocks))
        smoothed_et_blocks = list(
            map(lambda block: np.convolve(block[1:],
                                          np.ones(self.smoothing_factor),
                                          'same'), et_blocks))
        # finding the highest correlation sum for all eeg blocks
        larger_block_list, shorter_block_list = (
            smoothed_eeg_blocks, smoothed_et_blocks) if len(eeg_blocks) > len(
            et_blocks) else (smoothed_et_blocks, smoothed_eeg_blocks)
        best_offset = 0
        best_corr_sum = 0
        diff_in_lists_len = len(larger_block_list) - len(shorter_block_list) + 1
        for offset in range(diff_in_lists_len):
            # calculate sum of correlations, given the offset
            cur_corr = np.sum(list(
                map(lambda tup: self._get_block_correlation(tup[0], tup[1]),
                    zip(larger_block_list[
                        offset:len(
                            larger_block_list) - diff_in_lists_len + offset],
                        shorter_block_list))))
            if cur_corr > best_corr_sum:
                best_corr_sum = cur_corr
                best_offset = offset
        # create matched blocks based on eeg/et
        if len(eeg_blocks) > len(et_blocks):
            matched_eeg_blocks = np.arange(best_offset, len(
                eeg_blocks) - diff_in_lists_len + best_offset + 1, 1)
            matched_et_blocks = np.arange(0, len(et_blocks), 1)
        else:
            matched_eeg_blocks = np.arange(0, len(eeg_blocks), 1)
            matched_et_blocks = np.arange(best_offset, len(
                et_blocks) - diff_in_lists_len + best_offset + 1, 1)
        return matched_eeg_blocks, matched_et_blocks

    def _resample_eeg_to_et(self, eeg_blocks, eeg_sf) -> None:
        """
        Re-sample the EEG event blocks to ET sampling rate.
        Although the final result of the function will be the ET data re-sampled
        to the EEG time-course, this is first done by translating the latter to
        the ET time-course - so that each individual ET sample could be mapped to
        its corresponding EEG time point.
        """
        for i in range(len(eeg_blocks)):
            trig_sample_idx = np.where(eeg_blocks[i] != 0)[0]
            triggers = eeg_blocks[i][trig_sample_idx]
            new_trig_idx = np.ceil(trig_sample_idx * (
                    self.ET_SAMPLING_RATE_CONST / eeg_sf)).astype(np.int)
            new_block = np.zeros(np.ceil(eeg_blocks[i].size * (
                    self.ET_SAMPLING_RATE_CONST / eeg_sf)).astype(np.int))
            new_block[new_trig_idx] = triggers
            eeg_blocks[i] = new_block

    @staticmethod
    def _get_et_blocks(et_block_pause, et_block_starts, et_recording_stop,
                       et_samples, et_trig_ch) -> np.array:
        """
        Generate ET event time course (unlike the EEG event time course, the
        event timestamps are in absolute time (ms) and not sample indexes, so
        this timecourse will also include epochs in which there was no recording).
        """
        et_blocks = []
        for i in range(et_block_starts.size):
            is_stop = False
            cur_start = et_block_starts[i]
            if cur_start < 0 or cur_start == len(et_trig_ch):
                break
            end_conditions = (et_block_pause > cur_start) & (
                    et_block_pause <= et_block_starts[i + 1])
            possible_ends = np.where(end_conditions)[0]
            if possible_ends.size > 0:  # found an end trigger
                cur_end = et_block_pause[possible_ends[-1]] - 1
            else:  # recording was stopped without end trigger
                is_stop = True
                end_conditions = (et_recording_stop > cur_start) & (
                        et_recording_stop <= et_block_starts[i + 1])
                possible_ends = np.where(end_conditions)[0]
                if possible_ends.size > 0:  # found a discrepancy in time
                    cur_end = et_recording_stop[possible_ends[-1]]
                else:  # no recording stops
                    if et_samples[
                        -1, 0] > cur_start:  # the start is after the last et sample
                        cur_end = et_block_starts[i + 1]
                    else:  # something is wrong - just add the block till the next start
                        et_blocks.append(
                            et_trig_ch[cur_start:et_block_starts[i + 1]])
                        continue

            if np.sum(et_trig_ch[cur_start:cur_end]) > 1:
                et_blocks.append(et_trig_ch[cur_start:cur_end])
            else:
                et_block_starts[i] = -1
                (et_recording_stop if is_stop else et_block_pause)[
                    possible_ends[-1]] = -1
        return et_blocks

    def _prepare_et_eeg_sync_data(self, raw) -> typing.Tuple[
        np.array, np.array, np.array, np.array, np.array,
        np.array, np.array, np.array]:
        # get et data as np arrays
        et_samples = np.asarray(self._samples, dtype=np.int)
        et_trigs = np.asarray(self._triggers, dtype=np.int)
        et_trigs = et_trigs[np.where((et_trigs[:, 1] != 0) & (
                et_trigs[:, 0] <= np.max(et_samples[:, 0])))]

        # get sampling rates and time diffs
        eeg_sf = raw.info['sfreq']
        et_sf = self._sf
        et_timediff = int(et_samples[1, 0] - et_samples[0, 0])

        # Reset the first timestamp of the ET data to 1 (or 2 if the sampling rate
        first_ts = min(et_samples[0, 0], et_trigs[0, 0])
        et_samples[:, 0] += et_timediff - first_ts
        et_trigs[:, 0] += et_timediff - first_ts

        # Generate EEG event time course
        eeg_data_trig_ch = np.squeeze((raw['Status'])[0])
        eeg_data_trigs = find_events(raw, mask=255,
                                     mask_type="and")  # read events
        eeg_trigs = np.zeros_like(eeg_data_trig_ch, dtype=np.int)
        eeg_trigs[eeg_data_trigs[:, 0]] = eeg_data_trigs[:,
                                          2]  # transform events into eeg_data long vector
        # transform et events into et_num_samples long vector
        et_trig_ch = np.zeros(et_samples[-1, 0])
        et_trig_ch[et_trigs[:, 0]] = et_trigs[:, 1]
        return eeg_data_trig_ch, eeg_sf, eeg_trigs, et_samples, et_sf, et_timediff, et_trigs, et_trig_ch

    #################################################
    #               parsing methods                 #
    #################################################
    def _parse_sample(self, line) -> None:
        """
        parses a sample line from the EDF
        """
        s = self._parser.parse_sample(line)
        if self._parser.is_blink():
            cur_time = self._samples[-1][self._parser.TIME] + self._dt
            end_time = s[self._parser.TIME]
            while cur_time < end_time:
                self._samples.append(self._parser.get_empty_sample(cur_time))
                cur_time += self._dt
            self._parser.toggle_blink()
        self._samples.append(s)

    def _parse_msg(self, line) -> None:
        """
        parses a message line from the EDF
        """
        self._messages.append(self._parser.parse_msg(line))

    def _parse_input(self, line) -> None:
        """
        parses a trigger line from the EDF
        """
        self._triggers.append(self._parser.parse_input(line))

    def _parse_fixation(self, line) -> None:
        """
        parses a fixation line from the EDF
        """
        self._fixations.append(self._parser.parse_fixation(line))

    def _parse_saccade(self, line) -> None:
        """
        parses a saccade line from the EDF
        """
        self._saccades.append(self._parser.parse_saccade(line))

    def _parse_blinks(self, line) -> None:
        """
        parses a blink line from the EDF
        """
        self._blinks.append(self._parser.parse_blinks(line))

    def _parse_recordings(self, line) -> None:
        """
        parses a recording start/end line from the EDF
        """
        self._recording_data.append(self._parser.parse_recordings(line))

    def _parse_et_data(self) -> None:
        """
        parses the .asc file whose path is self._filename
        """
        # validity checks, raise exception in case of invalid input
        if not exists(self._filename):
            raise Exception("ET File does not exist!")
        if self._filename[-4:] != ".asc":
            raise Exception("given file is not an .asc file!")
        # read all lines
        with open(self._filename, 'r') as file:
            all_lines = file.readlines()
        # parse every line
        dt_calculated = False
        for i, line in enumerate(all_lines):
            if not dt_calculated and len(self._samples) > 1:
                self._dt = self._samples[1][self._parser.TIME] - \
                           self._samples[0][self._parser.TIME]
                dt_calculated = True
            self._parse_line(line)
        # create pandas data frames from the lists of dictionaries
        self._samples = pd.DataFrame(self._samples)
        self._messages = pd.DataFrame(self._messages)
        self._triggers = pd.DataFrame(self._triggers)
        self._fixations = pd.DataFrame(self._fixations)
        self._saccades = pd.DataFrame(self._saccades)
        self._blinks = pd.DataFrame(self._blinks)
        self._recording_data = pd.DataFrame(self._recording_data)

    def _parse_line(self, line) -> None:
        line = split(self._parser.LINE_SPLIT_PATTERN, line)
        # ignore lines that are not samples or have a
        # first token corresponding to one of the tokens in the
        # parsing methods dictionary
        if self._parser.is_sample(line):
            self._parse_sample(line)
        elif line[0] in self._parser.parse_line_by_token:
            self._parse_line_by_token[line[0]](line)
