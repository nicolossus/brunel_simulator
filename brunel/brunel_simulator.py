#!/usr/bin/env python
# -*- coding: utf-8 -*-
from multiprocessing import Pool, Process

import elephant.statistics as es
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quantities as pq
import seaborn as sns
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import (correlation_coefficient,
                                              spike_time_tiling_coefficient)
from elephant.spike_train_synchrony import spike_contrast as spike_contrast_fn
from viziphant.rasterplot import rasterplot as vizi_rasterplot
from viziphant.spike_train_correlation import plot_corrcoef

from . import BrunelNetwork, NetworkNotSimulated

ALLOWED_STATISTICS = ["mean_firing_rate",
                      "mean_cv",
                      "fanofactor",
                      "mean_corr",
                      "mean_corr2",
                      "mean_sttc",
                      "spike_contrast"]


class BrunelSimulator(BrunelNetwork):

    def __init__(
        self,
        T=1000,
        dt=0.1,
        N_rec=100,
        threads=1,
        print_time=False,
        neuron="exc",
        stats=None,
        t_start=None,
        t_stop=None,
        rate_unit=None,
        bin_size=None,
        **kwargs
    ):
        """
        **kwargs
            Arbitrary keyword arguments are passed to the BrunelNetwork
            constructor.
        """
        # Simulation
        super().__init__(**kwargs)
        # self._bnet = BrunelNetwork(**kwargs)
        self._T = T
        self._dt = dt
        self._N_rec = N_rec
        self._threads = threads
        self._print_time = print_time
        self._check_neuron_type(neuron)
        self._neuron = neuron

        # Summary statistic extraction
        # error handling
        if stats is None:
            msg = ("'stats' must be provided as a list of statistic attributes"
                   " in order to make the instance callable.")
            raise ValueError(msg)

        self._stat_names = stats[:]
        self._stats = stats[:]

        if isinstance(self._stats, (list, tuple, np.ndarray)):
            for i, stat in enumerate(self._stats):
                if not stat in ALLOWED_STATISTICS:
                    msg = (f"Unknown statistic '{stat}' provided. Refer to "
                           "documentation for a list of available statistics.")
                    raise ValueError(msg)
                self._stats[i] = "_" + stat

        self._t_start = t_start
        self._t_stop = t_stop
        self._rate_unit = rate_unit
        self._bin_size = bin_size
        self._olkin_pratt_k = (-7 + 9 * np.sqrt(2)) / 2

    def __call__(self, eta=2.0, g=5.0, J=0.1, D=1.5):
        self.eta = eta
        self.g = g
        self.J = J
        self.D = D
        self.simulate(T=self._T,
                      dt=self._dt,
                      N_rec=self._N_rec,
                      threads=self._threads,
                      print_time=self._print_time
                      )

        if self._neuron == "exc":
            self._spiketrains = self.spiketrains_exc
        elif self._neuron == "inh":
            self._spiketrains = self.spiketrains_inh

        self._sum_stats = np.array([getattr(self, stat)
                                   for stat in self._stats])

        '''
        # Does not work, seems like all threads are owned by NEST
        # The pool function calls only result in sporadic behavior of NEST
        with Pool(processes=len(self._stats)) as pool:
            res = []
            for stat in self._stats:
                r = pool.apply_async(getattr(self, stat), ())
                res.append(r)
            sum_stats = np.array([r.get() for r in res])

        return sum_stats
        '''

        return self._sum_stats

    @property
    def spiketrains(self):
        try:
            return self._spiketrains
        except AttributeError:
            msg = ("Missing simulator call. No solution exists.")
            raise NetworkNotSimulated(msg)

    @property
    def sum_stats(self):
        try:
            return self._sum_stats
        except AttributeError:
            msg = ("Missing simulator call. No solution exists.")
            raise NetworkNotSimulated(msg)

    @property
    def _mean_firing_rate(self):
        frate = self.mean_firing_rate(self._spiketrains,
                                      self._t_start,
                                      self._t_stop,
                                      self._rate_unit
                                      )
        return frate

    @property
    def _mean_cv(self):
        return self.mean_cv(self._spiketrains, self._t_start, self._t_stop)

    @property
    def _fanofactor(self):
        return self.fanofactor(self._spiketrains, self._t_start, self._t_stop)

    @property
    def _mean_corr(self):
        mean_corr = self.mean_corr(self._spiketrains,
                                   self._t_start,
                                   self._t_stop,
                                   self._bin_size
                                   )
        return mean_corr

    @property
    def _mean_corr2(self):
        mean_corr = self.mean_corr2(self._spiketrains,
                                    self._t_start,
                                    self._t_stop,
                                    self._bin_size
                                    )
        return mean_corr

    @property
    def _mean_sttc(self):
        mean_sttc = self.mean_sttc(self._spiketrains,
                                   self._t_start,
                                   self._t_stop,
                                   self._bin_size
                                   )
        return mean_sttc

    @property
    def _spike_contrast(self):
        spike_contrast = self.spike_contrast(self._spiketrains,
                                             self._t_start,
                                             self._t_stop,
                                             )
        return spike_contrast

    def _is_empty(self, spiketrains):
        """
        Returns True if all spike trains are empty.
        """
        return all(len(st) == 0 for st in spiketrains)

    def _spiketrains_with_variance(self, spiketrains):
        """Returns a list of spike trains where each train contains at least
        two spikes. (This is needed for summary calculations using variance)."""
        return [st for st in spiketrains if len(st) > 1]

    def _check_type_quantity(self, parameter, name):
        if not isinstance(parameter, pq.Quantity):
            msg = (f"{name} must be set as a Quantity object.")
            raise TypeError(msg)

    def slice_spiketrains(self, spiketrains, t_start=None, t_stop=None):
        if t_start is not None:
            self._check_type_quantity(t_start, 't_start')
        if t_stop is not None:
            self._check_type_quantity(t_stop, 't_stop')

        spiketrains_slice = []
        for spiketrain in spiketrains:
            if t_start is None:
                t_start = spiketrain.t_start
            if t_stop is None:
                t_stop = spiketrain.t_stop

            spiketrain_slice = spiketrain[np.where(
                (spiketrain > t_start) & (spiketrain < t_stop))]
            spiketrain_slice.t_start = t_start
            spiketrain_slice.t_stop = t_stop
            spiketrains_slice.append(spiketrain_slice)

        return spiketrains_slice

    def mean_firing_rate(
        self,
        spiketrains=None,
        t_start=None,
        t_stop=None,
        units=None
    ):
        """Compute the time and population-averaged firing rate.

        The time averaged firing rate of a single spike train is calculated as
        the number of spikes in the spike train in the range [t_start, t_stop]
        divided by the time interval t_stop - t_start. The mean firing rate of
        all the provided spike trains is simply calculated by averaging over
        all the recorded neurons.

        Notes
        -----
        The function uses the most common definition of a firing rate, namely the
        temporal average. Hence, the mean firing rate :math:`\bar{\nu}` of the
        spike trains is the spike count :math:`n^{\mathrm{sp}}_i` that occur over a
        time interval of length :math:`\Delta t` in spike train :math:`i`,
        :math:`i=1, ..., N_{\mathrm{spt}}`, divided by :math:`\Delta t` and
        averaged over the number of spike trains :math:`N_{\mathrm{spt}}`:

        .. math::
            \bar{\nu} = \frac{1}{N_{\mathrm{spt}}} \sum_{i=1}^{N_\mathrm{spt}} \frac{n_i^{\mathrm{sp}}}{\Delta t}

        The statistic is computed using the open-source package Elephant
        (RRID:SCR_003833); a library for the analysis of electrophysiological
        data.

        Parameters
        ----------
        spiketrains : :term:`array_like`
            Spike trains as a list of :obj:`neo.SpikeTrain` objects.
        t_start : :obj:`float` or :obj:`pq.Quantity`, optional
            The start time to use for the time interval. If `None`, it is retrieved
            from the `t_start` attribute of :obj:`neo.SpikeTrain`. Default: `None`.
        t_stop : :obj:`float` or :obj:`pq.Quantity`, optional
            The stop time to use for the time interval. If `None`, it is retrieved
            from the `t_stop` attribute of :obj:`neo.SpikeTrain`. Default: `None`.
        units : quantities.Quantity
            quantities.Quantity object specifying the unit of the rate. Defaults
            to :math:`kHz`, i.e. `quantities.kHz`.

        Returns
        -------
        mean_firing_rate : :obj:`float`
            Firing rate averaged over input spike trains in units specified by
            the unit keyword. Returns np.inf if an empty list is specified, or
            if all spike trains are empty.
        """

        if spiketrains is None:
            spiketrains = self.spiketrains

        if self._is_empty(spiketrains):
            # if all spike trains are empty
            return np.inf

        if t_start is None:
            t_start = self._t_start

        if t_stop is None:
            t_stop = self._t_stop

        if units is None:
            units = self._rate_unit

        if units is not None:
            _units = units
        else:
            _units = pq.kHz

        firing_rates = []

        for spiketrain in spiketrains:
            firing_rate = es.mean_firing_rate(spiketrain,
                                              t_start=t_start,
                                              t_stop=t_stop
                                              )
            firing_rate.units = _units
            firing_rates.append(firing_rate.magnitude)

        return np.mean(firing_rates)

    def mean_cv(
        self,
        spiketrains=None,
        t_start=None,
        t_stop=None
    ):
        """Compute the coefficient of variation averaged over recorded spike trains.

        The coefficient of variation (CV) is a measure of spike train variablity
        and is defined as the standard deviation of ISIs divided by their mean.
        A regularly spiking neuron would have a CV of 0, since there is no variance
        in the ISIs, whereas a Poisson process has a CV of 1.

        Notes
        -----
        CHANGE THIS

        .. math::
            \mathrm{CV} = \frac{\sqrt{\mathrm{Var}(\mathrm{ISIs})}}{\mathbb{E}[\mathrm{ISIs}]}

        The statistic is computed using the open-source package Elephant
        (RRID:SCR_003833); a library for the analysis of electrophysiological
        data.

        Parameters
        ----------
        spiketrains : :term:`array_like`
            Spike trains as a list of :obj:`neo.SpikeTrain` objects.

        Returns
        -------
        mean_cv : :obj:`float`
            Mean coefficient of variation of spike trains. Returns np.inf if
            an empty list is specified, or if all spike trains are empty.
        """
        if spiketrains is None:
            spiketrains = self.spiketrains

        # end computation if all spike trains are empty
        if self._is_empty(spiketrains):
            return np.inf

        # slice spike trains?
        if t_start is None:
            t_start = self._t_start

        if t_stop is None:
            t_stop = self._t_stop

        if t_start is not None or t_stop is not None:
            spiketrains = self.slice_spiketrains(spiketrains,
                                                 t_start=t_start,
                                                 t_stop=t_stop)

        # remove spike trains with zero variance
        spiketrains = self._spiketrains_with_variance(spiketrains)

        # check if modified spike trains are empty
        if self._is_empty(spiketrains):
            return np.inf

        # compute statistic
        mean_cv = np.mean([es.cv(es.isi(spiketrain), nan_policy='omit')
                           for spiketrain in spiketrains])

        return mean_cv

    def fanofactor(
        self,
        spiketrains=None,
        t_start=None,
        t_stop=None
    ):
        """Compute the Fano factor of the spike counts.

        Parameters
        ----------
        spiketrains : :term:`array_like`
            Spike trains as a list of :obj:`neo.SpikeTrain` objects.

        Returns
        -------
        fanofactor: :obj:`float`
            The Fano factor of the spike counts of the input spike trains. Returns
            np.inf if an empty list is specified, or if all spike trains are empty.
        """

        if spiketrains is None:
            spiketrains = self.spiketrains

        # end computation if all spike trains are empty
        if self._is_empty(spiketrains):
            return np.inf

        # slice spike trains?
        if t_start is None:
            t_start = self._t_start

        if t_stop is None:
            t_stop = self._t_stop

        if t_start is not None or t_stop is not None:
            spiketrains = self.slice_spiketrains(spiketrains,
                                                 t_start=t_start,
                                                 t_stop=t_stop)

        # remove spike trains with zero variance
        spiketrains = self._spiketrains_with_variance(spiketrains)

        # check if modified spike trains are empty
        if self._is_empty(spiketrains):
            return np.inf

        # compute statistic
        fano = es.fanofactor(spiketrains)

        return fano

    def mean_corr(
        self,
        spiketrains=None,
        t_start=None,
        t_stop=None,
        bin_size=None
    ):
        """Mean pairwise correlation coefficient using Fisher Z-transformation

        https://link.springer.com/content/pdf/10.3758/BF03334037.pdf
        """

        if spiketrains is None:
            spiketrains = self.spiketrains

        # end computation if all spike trains are empty
        if self._is_empty(spiketrains):
            return np.inf

        # slice spike trains?
        if t_start is None:
            t_start = self._t_start

        if t_stop is None:
            t_stop = self._t_stop

        if t_start is not None or t_stop is not None:
            spiketrains = self.slice_spiketrains(spiketrains,
                                                 t_start=t_start,
                                                 t_stop=t_stop)

        # remove spike trains with zero variance
        spiketrains = self._spiketrains_with_variance(spiketrains)

        # check if modified spike trains are empty
        if self._is_empty(spiketrains):
            return np.inf

        # set bin size
        if bin_size is None:
            if self._bin_size is not None:
                bin_size = self._bin_size
            else:
                bin_size = 50 * spiketrains[0].t_start.units

        # bin spike trains
        binned_spiketrains = BinnedSpikeTrain(spiketrains, bin_size=bin_size)

        # compute Pearson's pairwise correlation coefficient matrix
        corrcoef_matrix = correlation_coefficient(binned_spiketrains)

        # extract upper triangle matrix without the diagonal
        m = np.arange(corrcoef_matrix.shape[0])
        mask = m[:, None] < m

        # Fisher Z-transformation
        fisher_z = np.arctanh(corrcoef_matrix[mask])

        # compute mean
        mean_z = np.mean(fisher_z)

        # inverse transformation to retrieve mean correlation coefficient
        mean_corr = np.tanh(mean_z)

        return mean_corr

    def mean_corr2(
        self,
        spiketrains=None,
        t_start=None,
        t_stop=None,
        bin_size=None
    ):
        """Mean pairwise correlation coefficient using the Olkin & Pratt (1958)
        estimator

        https://link.springer.com/content/pdf/10.3758/BF03334037.pdf
        """

        if spiketrains is None:
            spiketrains = self.spiketrains

        # end computation if all spike trains are empty
        if self._is_empty(spiketrains):
            return np.inf

        # slice spike trains?
        if t_start is None:
            t_start = self._t_start

        if t_stop is None:
            t_stop = self._t_stop

        if t_start is not None or t_stop is not None:
            spiketrains = self.slice_spiketrains(spiketrains,
                                                 t_start=t_start,
                                                 t_stop=t_stop)

        # remove spike trains with zero variance
        spiketrains = self._spiketrains_with_variance(spiketrains)

        # check if modified spike trains are empty
        if self._is_empty(spiketrains):
            return np.inf

        # set bin size
        if bin_size is None:
            if self._bin_size is not None:
                bin_size = self._bin_size
            else:
                bin_size = 50 * spiketrains[0].t_start.units

        # bin spike trains
        binned_spiketrains = BinnedSpikeTrain(spiketrains, bin_size=bin_size)

        # compute Pearson's pairwise correlation coefficient matrix
        corrcoef_matrix = correlation_coefficient(binned_spiketrains)

        # extract upper triangle matrix without the diagonal
        m = np.arange(corrcoef_matrix.shape[0])
        mask = m[:, None] < m
        r = corrcoef_matrix[mask]

        # Olkin & Pratt estimator
        n = len(r) - 1
        G = r * (1 + (1 - r**2) / (2 * (n - self._olkin_pratt_k)))

        # compute mean
        mean_corr = np.mean(G)

        return mean_corr

    def corrcoefs(
        self,
        spiketrains=None,
        t_start=None,
        t_stop=None,
        bin_size=None
    ):
        """Mean pairwise correlation coefficient using the Olkin & Pratt (1958)
        estimator

        https://link.springer.com/content/pdf/10.3758/BF03334037.pdf
        """

        if spiketrains is None:
            spiketrains = self.spiketrains

        # end computation if all spike trains are empty
        if self._is_empty(spiketrains):
            return np.inf

        # slice spike trains?
        if t_start is None:
            t_start = self._t_start

        if t_stop is None:
            t_stop = self._t_stop

        if t_start is not None or t_stop is not None:
            spiketrains = self.slice_spiketrains(spiketrains,
                                                 t_start=t_start,
                                                 t_stop=t_stop)

        # remove spike trains with zero variance
        spiketrains = self._spiketrains_with_variance(spiketrains)

        # check if modified spike trains are empty
        if self._is_empty(spiketrains):
            return np.inf

        # set bin size
        if bin_size is None:
            if self._bin_size is not None:
                bin_size = self._bin_size
            else:
                bin_size = 50 * spiketrains[0].t_start.units

        # bin spike trains
        binned_spiketrains = BinnedSpikeTrain(spiketrains, bin_size=bin_size)

        # compute Pearson's pairwise correlation coefficient matrix
        corrcoef_matrix = correlation_coefficient(binned_spiketrains)

        # extract upper triangle matrix without the diagonal
        m = np.arange(corrcoef_matrix.shape[0])
        mask = m[:, None] < m
        rs = corrcoef_matrix[mask]

        return rs

    def mean_sttc(
        self,
        spiketrains=None,
        t_start=None,
        t_stop=None,
        bin_size=None
    ):
        """Spike tile correlation coefficients
        (values in the upper triangle matrix without the diagonal)
        """
        if spiketrains is None:
            spiketrains = self.spiketrains

        # end computation if all spike trains are empty
        if self._is_empty(spiketrains):
            return np.inf

        # slice spike trains?
        if t_start is None:
            t_start = self._t_start

        if t_stop is None:
            t_stop = self._t_stop

        if t_start is not None or t_stop is not None:
            spiketrains = self.slice_spiketrains(spiketrains,
                                                 t_start=t_start,
                                                 t_stop=t_stop)

        # remove spike trains with zero variance
        spiketrains = self._spiketrains_with_variance(spiketrains)

        # check if modified spike trains are empty
        if self._is_empty(spiketrains):
            return np.inf

        # set bin size
        if bin_size is None:
            if self._bin_size is not None:
                bin_size = self._bin_size
            else:
                bin_size = 50 * spiketrains[0].t_start.units

        ind = np.triu_indices(len(spiketrains), 1)
        sttcs = np.array([spike_time_tiling_coefficient(spiketrains[i],
                                                        spiketrains[j],
                                                        dt=bin_size)
                         for i, j in zip(*ind)]
                         )

        mean_sttc = np.mean(sttcs)
        return mean_sttc

    def sttcs(
        self,
        spiketrains=None,
        t_start=None,
        t_stop=None,
        bin_size=None
    ):
        """Spike tile correlation coefficients
        (values in the upper triangle matrix without the diagonal)
        """

        if spiketrains is None:
            spiketrains = self.spiketrains

        # end computation if all spike trains are empty
        if self._is_empty(spiketrains):
            return np.inf

        # slice spike trains?
        if t_start is None:
            t_start = self._t_start

        if t_stop is None:
            t_stop = self._t_stop

        if t_start is not None or t_stop is not None:
            spiketrains = self.slice_spiketrains(spiketrains,
                                                 t_start=t_start,
                                                 t_stop=t_stop)

        # remove spike trains with zero variance
        spiketrains = self._spiketrains_with_variance(spiketrains)

        # check if modified spike trains are empty
        if self._is_empty(spiketrains):
            return np.inf

        # set bin size
        if bin_size is None:
            if self._bin_size is not None:
                bin_size = self._bin_size
            else:
                bin_size = 50 * spiketrains[0].t_start.units

        ind = np.triu_indices(len(spiketrains), 1)
        sttc = np.array([spike_time_tiling_coefficient(spiketrains[i],
                                                       spiketrains[j],
                                                       dt=bin_size)
                         for i, j in zip(*ind)]
                        )

        return sttc

    def spike_contrast(
        self,
        spiketrains=None,
        t_start=None,
        t_stop=None,
    ):
        """Spike tile correlation coefficients
        (values in the upper triangle matrix without the diagonal)
        """

        if spiketrains is None:
            spiketrains = self.spiketrains

        # end computation if all spike trains are empty
        if self._is_empty(spiketrains):
            return np.inf

        # slice spike trains?
        if t_start is None:
            t_start = self._t_start

        if t_stop is None:
            t_stop = self._t_stop

        if t_start is not None or t_stop is not None:
            spiketrains = self.slice_spiketrains(spiketrains,
                                                 t_start=t_start,
                                                 t_stop=t_stop)

        sc = spike_contrast_fn(spiketrains,
                               t_start=None,
                               t_stop=None,
                               min_bin=10 * pq.ms,
                               bin_shrink_factor=0.9
                               )

        return sc

    @property
    def frame(self):
        data = dict(zip(self._stat_names, self.sum_stats))
        df = pd.DataFrame([data])
        df.insert(0, "eta", self.eta)
        df.insert(1, "g", self.g)
        df.insert(2, "J", self.J)
        df.insert(3, "D", self.D)
        return df

    def save_frame(self, filename, index=False):
        df = self.frame
        df.to_csv(filename, index=index)

    def rasterplot(
        self,
        spiketrains=None,
        t_start=None,
        t_stop=None,
        ax=None,
        marker='.',
        s=3,
        c='black',
        **kwargs
    ):
        """
        ax: matplotlib.axes.Axes or None, optional
            Matplotlib axes handle. If None, new axes are created and returned.
            Default: None
        s: float or array-like, shape (n, ), optional
            The marker size in points**2
        **kwargs
            Arbitrary keyword arguments
        """
        if spiketrains is None:
            spiketrains = self.spiketrains

        if t_start is None:
            t_start = self._t_start

        if t_stop is None:
            t_stop = self._t_stop

        if t_start is not None or t_stop is not None:
            spiketrains = self.slice_spiketrains(spiketrains,
                                                 t_start=t_start,
                                                 t_stop=t_stop)

        if ax is None:
            ax = plt.gca()

        vizi_rasterplot(spiketrains,
                        marker=marker,
                        s=s,
                        c=c,
                        axes=ax,
                        **kwargs
                        )

        '''
        # Dirty solution for labeling the first neuron by 1 instead of 0
        # and spacing the labels such that y-axis ticks = [1, 10, 20, ...]
        yticklabels = [1]
        for st in spiketrains[9::10]:
            yticklabels.append(st.annotations['unitID'])
        yticks = np.array(yticklabels) - 1
        '''

        ax.set(xlabel=f'Time ({spiketrains[0].t_start.dimensionality})',
               ylabel='Neuron',
               # yticks=yticks,
               # yticklabels=yticklabels
               )

    def rateplot(
        self,
        spiketrains=None,
        t_start=None,
        t_stop=None,
        bin_size=None,
        units=None,
        ax=None,
        color='C0',
        edgecolor='C0',
        align='edge',
        alpha=1,
        **kwargs
    ):
        """Plot time resolved firing rate
        """

        if spiketrains is None:
            spiketrains = self.spiketrains

        if t_start is None:
            t_start = self._t_start

        if t_stop is None:
            t_stop = self._t_stop

        if t_start is not None or t_stop is not None:
            spiketrains = self.slice_spiketrains(spiketrains,
                                                 t_start=t_start,
                                                 t_stop=t_stop)

        if bin_size is None:
            if self._bin_size is not None:
                bin_size = self._bin_size
            else:
                bin_size = 10 * spiketrains[0].t_start.units

        if units is not None:
            _units = units
        else:
            _units = pq.Hz

        if ax is None:
            ax = plt.gca()

        hist = es.time_histogram(spiketrains,
                                 bin_size=bin_size,
                                 output='rate')

        hist = hist.rescale(_units)

        ax.bar(hist.times,
               hist.magnitude.flatten(),
               width=hist.sampling_period,
               color=color,
               edgecolor=edgecolor,
               align=align,
               alpha=alpha,
               **kwargs,
               )

        ax.set(xlabel=f'Time ({spiketrains[0].t_start.dimensionality})',
               ylabel=f'Spike rate ({_units.dimensionality})'
               )

    def plot_mean_frate(
        self,
        spiketrains=None,
        t_start=None,
        t_stop=None,
        bin_size=None,
        units=None,
        ax=None,
        color='C3',
        lw=1.5,
        ls='-',
        **kwargs
    ):
        """Plot mean firing rate
        """
        if spiketrains is None:
            spiketrains = self.spiketrains

        if t_start is None:
            t_start = self._t_start

        if t_stop is None:
            t_stop = self._t_stop

        if units is not None:
            _units = units
        else:
            _units = pq.Hz

        '''
        # this should be removed 
        if t_start is not None or t_stop is not None:
            spiketrains = self.slice_spiketrains(spiketrains,
                                                 t_start=t_start,
                                                 t_stop=t_stop)
        '''
        if ax is None:
            ax = plt.gca()

        mean_frate = self.mean_firing_rate(spiketrains,
                                           t_start,
                                           t_stop,
                                           units=_units)

        ax.axhline(mean_frate, color=color, lw=lw, ls=ls, **kwargs)

    def plotcorr(
        self,
        spiketrains=None,
        t_start=None,
        t_stop=None,
        bin_size=None,
        ax=None,
        colormap='bwr',
        correlation_range='full',
        **kwargs
    ):
        """Plot pairwise correlation matrix
        """

        if spiketrains is None:
            spiketrains = self.spiketrains

        if t_start is None:
            t_start = self._t_start

        if t_stop is None:
            t_stop = self._t_stop

        if t_start is not None or t_stop is not None:
            spiketrains = self.slice_spiketrains(spiketrains,
                                                 t_start=t_start,
                                                 t_stop=t_stop)
        if bin_size is None:
            if self._bin_size is not None:
                bin_size = self._bin_size
            else:
                bin_size = 50 * spiketrains[0].t_start.units

        if ax is None:
            ax = plt.gca()

        binned_spiketrains = BinnedSpikeTrain(spiketrains, bin_size=bin_size)
        corrcoef_matrix = correlation_coefficient(binned_spiketrains)

        '''
        # Dirty solution for labeling the first neuron by 1 instead of 0
        # and spacing the labels such that ticks = [1, 10, 20, ...]
        ticklabels = [1]
        for st in spiketrains[9::10]:
            ticklabels.append(st.annotations['unitID'])
        ticks = np.array(ticklabels) - 1
        '''

        with sns.axes_style("white"):
            plot_corrcoef(corrcoef_matrix,
                          colormap=colormap,
                          correlation_range=correlation_range,
                          axes=ax,
                          **kwargs
                          )

            ax.set(xlabel='Neuron',
                   ylabel='Neuron',
                   # xticks=ticks,
                   # xticklabels=ticklabels,
                   # yticks=ticks,
                   # yticklabels=ticklabels
                   )

    def plot_simulation(self, figsize=(6, 6)):
        fig, axes = plt.subplots(nrows=3,
                                 ncols=1,
                                 figsize=figsize,
                                 tight_layout=True
                                 )

        self.rasterplot(ax=axes[0])
        self.rateplot(ax=axes[1])
        self.plot_mean_frate(ax=axes[1])
        self.plotcorr(ax=axes[2])


if __name__ == "__main__":

    import time

    sum_stats = ["mean_firing_rate",
                 "mean_cv",
                 "fanofactor",
                 "mean_sttc",
                 "spike_contrast"]

    bnet = BrunelSimulator(order=2500,  # 500
                           T=1000,
                           dt=0.1,
                           N_rec=100,
                           threads=16,
                           print_time=False,
                           neuron='exc',
                           stats=sum_stats,
                           t_start=100 * pq.ms,
                           t_stop=1000 * pq.ms
                           )

    start = time.time()
    sum_stats = bnet(eta=2.0, g=2.5, J=0.35)
    end = time.time()
    print("Time:", end - start)
    print(bnet.frame)
