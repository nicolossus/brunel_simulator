#!/usr/bin/env python
# -*- coding: utf-8 -*-
import nest
import numpy as np
import quantities as pq
from neo.core import SpikeTrain

nest.set_verbosity(level="M_QUIET")


class NetworkNotSimulated(Exception):
    """Failed attempt at accessing solutions.

    A call to simulate the network must be
    carried out before the solution properties
    can be used.
    """
    pass


class BrunelNetwork:
    """Implementation of the sparsely connected recurrent network described by
    Brunel (2000).

    The Brunel network model characterizes the local cortical network as a
    network composed of 80% excitatory and 20% inhibitory leaky integrate-and-fire
    (LIF) neurons that are interconnected. Each neuron receives randomly
    connection from other neurons, both excitatory and inhibitory, in the
    network. In addition to sparse recurrent inputs from within the network,
    each neuron receives excitatory synaptic input from a population of
    randomly firing neurons outside the network with activation governed by
    identical, independent Poisson processes with fixed-rate.

    The Brunel network may be in several different states of spiking activity,
    largely dependent on the values of the synaptic weight parameters. In the
    context of a biological neural network, synaptic weight parameters refer to
    parameters that determines the influence the firing of one neuron has on
    another. The spiking activity can be in a state of synchronous regular (SR),
    asynchronous irregular (AI) or synchronous irregular (SI), with either fast
    or slow oscillations, activity.

    Default parameters are chosen for the AI state.

    Parameters
    ----------
    order : int, optional

    eta : {int, float}, optional
        External rate relative to threshold rate. Default is 2.
    g : {int, float}, optional
        Ratio inhibitory weight/excitatory weight. Default is 5.
    delay : {int, float}, optional
        Synaptic delay in ms. Default is 1.5.
    J : {int, float}, optional
        Amplitude of excitatory postsynaptic current. Default is 0.1

    References
    ----------
    Brunel N, Dynamics of Sparsely Connected Networks of Excitatory and
    Inhibitory Spiking Neurons, Journal of Computational Neuroscience 8,
    183-208 (2000).
    """

    def __init__(
            self,
            order=100,
            epsilon=0.1,
            eta=2.0,
            g=5.0,
            J=0.1,
            C_m=1.,
            V_rest=0.,
            V_th=20.,
            V_reset=10.,
            tau_m=20.,
            tau_rp=2.,
            D=1.5
    ):

        # NETWORK PARAMETERS
        self._NE = 4 * int(order)              # no. of excitatory neurons
        self._NI = 1 * int(order)              # no. of inhibitory neurons
        self._N_neurons = self._NE + self._NI  # total no. of neurons
        self._epsilon = epsilon                # connection probability

        # no. of excitatory synapses per neuron
        self._CE = int(self._epsilon * self._NE)
        # no. of inhibitory synapses per neuron
        self._CI = int(self._epsilon * self._NI)
        # total no. of synapses per neuron
        self._C = self._CE + self._CI

        # NEURON PARAMETERS
        self._eta = eta         # background rate
        self._g = g             # relative strength of inhibitory synapses
        self._J = J             # absolute excitatory strength
        self._V_th = V_th       # firing threshold
        self._tau_m = tau_m     # membrane time constant
        self._D = D             # synaptic delay

        self._neuron_params = {'C_m': C_m,
                               'tau_m': self._tau_m,
                               't_ref': tau_rp,
                               'E_L': V_rest,
                               'V_th': self._V_th,
                               'V_reset': V_reset}

        # Flags
        self._is_simulated = False
        self._is_calibrated = False
        self._is_built = False
        self._is_connected = False

    def _calibrate(self):
        """Compute dependent variables"""

        # Excitatory PSP amplitude
        self._J_ex = self._J

        # Inhibitory PSP amplitude
        self._J_in = - self._g * self._J

        # Threshold rate; the external rate needed for a neuron to reach
        # threshold in absence of feedback
        self._nu_th = self._V_th / (self._J * self._CE * self._tau_m)

        # External firing rate; firing rate of a neuron in the external
        # population
        self._nu_ext = self._eta * self._nu_th

        # Population rate of the whole external population; the product of the
        # Poisson generator rate and the in-degree C_E. The factor 1000.0
        # in the product changes the units from spikes per ms to spikes per
        # second, i.e. the rate is converted to Hz.
        self._p_rate = 1000.0 * self._nu_ext * self._CE

    def _build_network(self):
        """Create and connect network elements.

        NEST recommends that all elements in the network, i.e., neurons,
        stimulating devices and recording devices, should be created before
        creating any connections between them.
        """

        # Set parameters for neurons
        nest.SetDefaults("iaf_psc_delta", self._neuron_params)

        # Create local excitatory neuron population
        self._nodes_ex = nest.Create("iaf_psc_delta", self._NE)
        # Create local inhibitory neuron population
        self._nodes_in = nest.Create("iaf_psc_delta", self._NI)

        # Distribute membrane potentials to random values between zero and
        # threshold
        nest.SetStatus(self._nodes_ex,
                       "V_m",
                       np.random.rand(len(self._nodes_ex)) * self._V_th
                       )
        nest.SetStatus(self._nodes_in,
                       "V_m",
                       np.random.rand(len(self._nodes_in)) * self._V_th
                       )

        self._Vm_ini_ex = np.array(nest.GetStatus(self._nodes_ex, 'V_m'))
        self._Vm_ini_in = np.array(nest.GetStatus(self._nodes_in, 'V_m'))

        # Create external population. The 'poisson_generator' device produces
        # a spike train governed by a Poisson process at a given rate. If a
        # Poisson generator is connected to N targets, it generates N i.i.d.
        # spike trains. Thus, we only need one generator to model an entire
        # population of randomly firing neurons.
        noise = nest.Create("poisson_generator", 1, {"rate": self._p_rate})

        # Create spike recorders to observe how the neurons in the recurrent
        # network respond to the random spikes from the external population.
        # We create one recorder for each neuron population (excitatory and
        # inhibitory).
        '''
        self._spikes = nest.Create("spike_detector", 2,
                                   [{"label": 'brunel-py-ex'},
                                    {"label": 'brunel-py-in'}])
        '''
        self._spikes = nest.Create("spike_recorder", 2,
                                   [{"label": 'brunel-py-ex'},
                                    {"label": 'brunel-py-in'}])
        self._espikes = self._spikes[:1]
        self._ispikes = self._spikes[1:]

        # Configure synapse using `CopyModel`, which expects the model name
        # of a pre-defined synapse, the name of the customary synapse and
        # an optional parameter dictionary
        nest.CopyModel("static_synapse",
                       "excitatory",
                       {"weight": self._J_ex,
                        "delay": self._D}
                       )

        nest.CopyModel("static_synapse",
                       "inhibitory",
                       {"weight": self._J_in,
                        "delay": self._D}
                       )

        # Connecting network nodes:
        # The function `Connect` expects four arguments: a list of source nodes,
        # a list of target nodes, a connection rule, and a synapse
        # specification (syn_spec).

        # Connect 'external population' Poisson generator to the local
        # excitatory and inhibitory neurons using the excitatory synapse.
        # Since the Poisson generator is connected to all neurons in the local
        # populations, the default rule, 'all_to_all', of `Connect` is used.
        # The synaptic properties are inserted via syn_spec.
        nest.Connect(noise,
                     self._nodes_ex,
                     'all_to_all',
                     syn_spec='excitatory'
                     )

        nest.Connect(noise,
                     self._nodes_in,
                     'all_to_all',
                     syn_spec='excitatory'
                     )

        # Connect subset of the nodes of the excitatory and inhibitory
        # populations to the associated spike recorder using excitatory
        # synapses.
        nest.Connect(self._nodes_ex[:self._N_rec],
                     self._espikes,
                     'all_to_all',
                     syn_spec='excitatory'
                     )

        nest.Connect(self._nodes_in[:self._N_rec],
                     self._ispikes,
                     'all_to_all',
                     syn_spec='excitatory'
                     )

        # Connect the excitatory/inhibitory population to all neurons using the
        # pre-defined excitatory/inhibitory synapse. Beforehand, the connection
        # parameters are defined in a dictionary. Here, we use the connection
        # rule 'fixed_indegree', which requires the definition of the indegree.
        # Since the synapse specification is reduced to assigning the
        # pre-defined excitatory synapse it suffices to insert a string.
        conn_params_ex = {'rule': 'fixed_indegree', 'indegree': self._CE}
        nest.Connect(self._nodes_ex,
                     self._nodes_ex + self._nodes_in,
                     conn_params_ex,
                     "excitatory"
                     )

        conn_params_in = {'rule': 'fixed_indegree', 'indegree': self._CI}
        nest.Connect(self._nodes_in,
                     self._nodes_ex + self._nodes_in,
                     conn_params_in,
                     "inhibitory"
                     )

    def simulate(self, T=1000, dt=0.1, N_rec=100, threads=1, print_time=False):
        """Simulate the model.

        Parameters
        ----------
        T : {int, float}, optional
            Simulation time in ms
        dt : float, optional
            Time resolution in ms
        N_rec : int, optional
            Number of neurons to record
        threads : int
            Number of threads
        print_time : bool
            Whether to print network time or not
        """

        self._T = T
        self._N_rec = N_rec

        # Start a new NEST session
        nest.ResetKernel()
        nest.set_verbosity(level="M_FATAL")

        nest.SetKernelStatus({"resolution": dt,
                              "print_time": print_time,
                              # "local_num_threads": threads,
                              "total_num_virtual_procs": 1
                              }
                             )

        # calibrate/compute network parameters
        self._calibrate()
        # build network
        self._build_network()
        # simulate network
        nest.Simulate(self._T)

        # Read out recordings
        self._events_ex = nest.GetStatus(self._espikes, "n_events")[0]
        self._events_in = nest.GetStatus(self._ispikes, "n_events")[0]

        ex_events = nest.GetStatus(self._espikes, 'events')[0]
        in_events = nest.GetStatus(self._ispikes, 'events')[0]
        ex_spikes = np.stack((ex_events['senders'], ex_events['times'])).T
        in_spikes = np.stack((in_events['senders'], in_events['times'])).T

    @ property
    def t_stop(self):
        return self._T

    @ property
    def n_neurons_exc(self):
        return self._NE

    @ property
    def n_neurons_inh(self):
        return self._NI

    @ property
    def n_neurons(self):
        return self._N_neurons

    @ property
    def n_synapses_exc(self):
        return nest.GetDefaults("excitatory")["num_connections"]

    @ property
    def n_synapses_inh(self):
        return nest.GetDefaults("inhibitory")["num_connections"]

    @ property
    def n_synapses(self):
        return self.n_synapses_exc + self.n_synapses_inh

    def print_network(self):
        print("-" * 40)
        print("Brunel network")
        print("-" * 40)
        print(f"Number of neurons            : {self.n_neurons}")
        print(f"        Excitatory           : {self.n_neurons_exc}")
        print(f"        Inhibitory           : {self.n_neurons_inh}")
        print(f"Number of synapses           : {self.n_synapses}")
        print(f"        Excitatory           : {self.n_synapses_exc}")
        print(f"        Inhibitory           : {self.n_synapses_inh}")
        print("-" * 40)

    def _neo_spiketrains(self, nodes, events, neuron):
        try:
            neo_spiketrains = []
            for sender in nodes[:self._N_rec]:
                try:
                    st = events['times'][events['senders'] == sender]
                    id = events['senders'][events['senders'] == sender][0]
                    neo_spiketrain = SpikeTrain(st,
                                                t_stop=self.t_stop,
                                                units=pq.ms,
                                                n_type=neuron,
                                                unitID=id
                                                )

                    neo_spiketrains.append(neo_spiketrain)
                except IndexError:
                    neo_spiketrain = SpikeTrain([],
                                                t_stop=self.t_stop,
                                                units=pq.ms
                                                )
        except AttributeError:
            msg = ("Missing call to simulate. No solution exists.")
            raise NetworkNotSimulated(msg)

        return neo_spiketrains

    @ property
    def spiketrains_exc(self):
        events = nest.GetStatus(self._espikes, 'events')[0]
        return self._neo_spiketrains(self._nodes_ex, events, "exc")

    @ property
    def spiketrains_inh(self):
        events = nest.GetStatus(self._ispikes, 'events')[0]
        return self._neo_spiketrains(self._nodes_in, events, "inh")

    def _check_neuron_type(self, n_type):
        """Check whether neuron type is provided as 'exc' or 'inh'."""
        if not isinstance(n_type, str):
            msg = ("'neuron' must be passed as str, either 'exc'"
                   " (excitatory) or 'inh' (inhibitory).")
            raise TypeError(msg)
        if n_type not in ['exc', 'inh']:
            msg = ("'neuron' must be set as either 'exc' (excitatory) or"
                   " 'in' (inhibitory).")
            raise ValueError(msg)

    # Get and set model parameters
    @ property
    def eta(self):
        return self._eta

    @ eta.setter
    def eta(self, eta):
        self._eta = eta

    @ property
    def g(self):
        return self._g

    @ g.setter
    def g(self, g):
        self._g = g

    @ property
    def J(self):
        return self._J

    @ J.setter
    def J(self, J):
        self._J = J

    @ property
    def D(self):
        return self._D

    @ D.setter
    def D(self, D):
        self._D = D
