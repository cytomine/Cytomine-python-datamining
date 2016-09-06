# -*- coding: utf-8 -*-

import timeit

import numpy as np

__author__ = "Romain Mormont <romainmormont@hotmail.com>"
__version = "0.1"


class WorkflowTiming(object):
    """A class that computes and stores execution times for various phases of the workflow.
    WorkflowTiming objects can be combined (their stored execution times are added)

    Class constants:
        - LOADING : Time required for loading image into memory (call of tile.np_image)
        - SEGMENTATION : Time for segmenting the tiles (call of segmenter.segment)
        - MERGING : Time for merging the polygons found in the tiles (call of merger.merge)
        - LOCATION : Time for locating the polygons in the segmented tiles (call of locator.locate)
        - DISPATCH : Time for dispatching the polygons (call of rule.evaluate_batch or rule.evaluate)
        - CLASSIFY : Time for classifying the polygons (call of polygon_classifier.predict_batch or
                     polygon_classifier.predict)
        - FSL : Total time for executing the fetch/segment/locate (same as FETCHING + SEGMENTATION + MERGING in case of
                sequential execution. Less then these in case of parallel execution)
    """

    LOADING = "loading"
    SEGMENTATION = "segmentation"
    MERGING = "merging"
    LOCATION = "location"
    DISPATCH = "dispatch"
    CLASSIFY = "classify"
    LSL = "load_segment_locate"
    DC = "dispatch_classify"

    def __init__(self):
        """Construct a WorkflowTiming object
        """
        self._durations = {
            WorkflowTiming.LOADING: [],
            WorkflowTiming.SEGMENTATION: [],
            WorkflowTiming.MERGING: [],
            WorkflowTiming.LOCATION: [],
            WorkflowTiming.DISPATCH: [],
            WorkflowTiming.CLASSIFY: [],
            WorkflowTiming.LSL: [],
            WorkflowTiming.DC: []
        }
        self._start_dict = dict()

    def start_loading(self):
        """Record the start for the 'loading' phase
        """
        self._record_start(WorkflowTiming.LOADING)

    def end_loading(self):
        """Record the end for the 'loading' phase
        """
        self._record_end(WorkflowTiming.LOADING)

    def start_segment(self):
        """Record the start for the 'segment' phase
        """
        self._record_start(WorkflowTiming.SEGMENTATION)

    def end_segment(self):
        """Record the end for the 'segment' phase
        """
        self._record_end(WorkflowTiming.SEGMENTATION)

    def start_location(self):
        """Record the start for the 'location' phase
        """
        self._record_start(WorkflowTiming.LOCATION)

    def end_location(self):
        """Record the end for the 'location' phase
        """
        self._record_end(WorkflowTiming.LOCATION)

    def start_dispatch(self):
        """Record the start for the 'dispatch' phase
        """
        self._record_start(WorkflowTiming.DISPATCH)

    def end_dispatch(self):
        """Record the end for the 'dispatch' phase
        """
        self._record_end(WorkflowTiming.DISPATCH)

    def start_classify(self):
        """Record the start for the 'classify' phase
        """
        self._record_start(WorkflowTiming.CLASSIFY)

    def end_classify(self):
        """Record the end for the 'classify' phase
        """
        self._record_end(WorkflowTiming.CLASSIFY)

    def start_merging(self):
        """Record the start for the 'merging' phase
        """
        self._record_start(WorkflowTiming.MERGING)

    def end_merging(self):
        """Record the end for the 'merging' phase
        """
        self._record_end(WorkflowTiming.MERGING)

    def start_lsl(self):
        """Record the start for the 'load_segment_locate' phase
        """
        self._record_start(WorkflowTiming.LSL)

    def end_lsl(self):
        """Record the end for the 'load_segment_locate' phase
        """
        self._record_end(WorkflowTiming.LSL)

    def start_dc(self):
        """Record the start for the 'dispatch_classify' phase
        """
        self._record_start(WorkflowTiming.DC)

    def end_dc(self):
        """Record the end for the 'dispatch_classify' phase
        """
        self._record_end(WorkflowTiming.DC)

    def statistics(self):
        """Compute time statistics tuples for each phase of the algorithm
        Returns
        -------
        stats: dict
            A dictionary mapping phase string with a stat tuple containing time statistics for the given phase
        """
        stats = dict()
        for key in self._durations.keys():
            stats[key] = self._stat_tuple(key)
        return stats

    def total(self):
        """Compute the total execution times of the algorithm recorded so far
        """
        total_time = 0
        for key in self._durations.keys():
            total_time += sum(self._durations[key])
        return total_time

    def sl_total_duration(self):
        """Return the total execution time for segmenting tiles and locating polygons recoreded so far
        Returns
        -------
        time: float
            The execution time in second
        """
        return self.total_duration_of([WorkflowTiming.SEGMENTATION, WorkflowTiming.LOCATION])

    def lsl_total_duration(self):
        """Return the total execution time for the loading, segment and locate phase
        Returns
        -------
        time: float
            The execution time in second
        """
        return self.total_duration_of([WorkflowTiming.LSL])

    def dc_total_duration(self):
        """Return the total execution time for dispatching and classifying polygons recoreded so far

        Returns
        -------
        time: float
            The execution time in second
        """
        return self.total_duration_of([WorkflowTiming.DC])

    def duration_of(self, phase):
        """Return the total duration of the given phase
        Parameters
        ----------
        phase: string
            The phase string

        Returns
        -------
        time: float
            Total time in seconds
        """
        if phase not in self._durations:
            return 0
        return sum(self._durations[phase])

    def total_duration_of(self, phases):
        """Return the total dûration of the given phases
        Parameters
        ----------
        phases: iterable (subtype: string)
            Iterable containing the strings of the phases to be included in the computed times
        Returns
        -------
        time: float
            Total time in seconds
        """
        if len(phases) == 0:
            return 0
        return sum([self.duration_of(phase) for phase in phases])

    def _record_start(self, phase):
        """Record a start for a given phase
        Parameters
        ----------
        phase: string
            The string of the phase that starts
        """
        self._start_dict[phase] = timeit.default_timer()

    def _record_end(self, phase):
        """Record an end for a given phase
        Parameters
        ----------
        phase: string
            The string of the phase that ends
        """
        start = self._start_dict.get(phase)
        if start is not None:
            self._durations[phase].append(timeit.default_timer() - start)
            del self._start_dict[phase]

    def _stat_tuple(self, phase):
        """Make a statistics tuple from the given phase string
        Parameters
        ----------
        phase: string
            The phase string of which statistics tuple is wanted
        Returns
        -------
        stats: tuple of (float, float, float, float, float, float)
            Tuple containing the following stats (sum, min, mean, max, std, count)
        """
        durations = np.array(self._durations[phase])
        count = durations.shape[0]
        if count == 0:
            return 0, 0, 0, 0, 0
        return round(np.sum(durations), 5), \
            round(np.min(durations), 5), \
            round(np.mean(durations), 5), \
            round(np.max(durations), 5), \
            round(np.std(durations), 5), \
            count

    @classmethod
    def merge_timings(cls, timing1, timing2):
        """Merge the two timings into a new timing object
        Parameters
        ----------
        timing1: WorkflowTiming
            The first timing object to merge
        timing2: WorkflowTiming
            The second timing object to merge
        Returns
        -------
        timing: WorkflowTiming
            A new timing object containing the merging of the passed timings
        """
        if timing1 is None and timing2 is None:
            return WorkflowTiming()
        elif timing1 is None or timing2 is None:
            return timing1 if timing1 is not None else timing2

        timing = WorkflowTiming()
        for key in timing._durations.keys():
            timing._durations[key] = timing1._durations.get(key, []) + timing2._durations.get(key, [])
        return timing

    def merge(self, other):
        """Merge the other WorkflowTiming into the current one

        Parameters
        ----------
        other: WorkflowTiming
            The WorkflowTiming object to merge
        """
        if other is None or not isinstance(other, WorkflowTiming):
            return
        for key in self._durations.keys():
            self._durations[key] += other._durations.get(key, [])

    def report(self, logger):
        """Report the execution times of the workflow phases using the given logger
        Parameters
        ----------
        logger: Logger
            The logger to which the times must be notified
        """
        to_report = "Execution times of the workflow phases."
        stats = self.statistics()
        for key in stats.keys():
            curr_stat = stats[key]
            to_report += "\n  {} : {} s (avg: {} s, std: {} s)".format(key, curr_stat[0], curr_stat[2], curr_stat[4])
        logger.info(to_report)