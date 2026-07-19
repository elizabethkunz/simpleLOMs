"""
models/base.py
--------------
Abstract base class for all LC fitting strategies.

Usage pattern
-------------
    model = FosterFit(cpw_params=CPWParams())  # or OptimizedFit, AnalyticalFit
    model.fit(freq, d=7e-3, Cc1=..., Cc2=...)  # runs the fitting algorithm
    net   = model.get_network(freq, ...)        # build rf.Network from results
    params = model.get_params()                 # {"L": ..., "C": ...}
    assert model.is_fitted                      # True after fit()
"""
from __future__ import annotations
from abc import ABC, abstractmethod

import skrf as rf


class BaseFit(ABC):
    """
    Abstract base class for LC lumped-element fitting strategies.

    All subclasses must implement fit(), get_network(), and get_params().
    The is_fitted property is managed automatically by the base class
    once self.L and self.C are set by a subclass.
    """

    #: Effective inductance in Henries — set by fit()
    L: float | None = None

    #: Effective capacitance in Farads — set by fit()
    C: float | None = None

    @property
    def is_fitted(self) -> bool:
        """True if fit() has been called and produced valid L and C values."""
        return self.L is not None and self.C is not None

    def _require_fitted(self) -> None:
        """Raise a clear error if get_network() is called before fit()."""
        if not self.is_fitted:
            raise RuntimeError(
                "{} has not been fitted yet. Call fit() before get_network().".format(
                    type(self).__name__
                )
            )

    @abstractmethod
    def fit(self, freq: rf.Frequency, **kwargs) -> None:
        """
        Run the fitting algorithm and store results in self.L and self.C.

        After this method returns, self.is_fitted must be True.

        Parameters
        ----------
        freq : rf.Frequency
            Frequency sweep to use for network construction during fitting.
        **kwargs
            Subclass-specific parameters (e.g. d, Cc1, Cc2, data_ntw).
        """

    def _maybe_plot(
        self,
        net: rf.Network,
        *,
        show: bool = False,
        reference: rf.Network | None = None,
        m: int = 0,
        n: int = 0,
        lom_label: str | None = None,
        data_label: str = "CPW",
        save_path: str | None = None,
    ) -> None:
        """
        Optionally plot the built network.

        If ``show`` is True and a reference network is available, draw an
        overlay via :func:`simpleLOMs.plotting.plot_lom_vs_data`.  Otherwise
        fall back to :func:`simpleLOMs.plotting.plot_re_im` so that at least
        one figure is produced.
        """
        if not show:
            return
        from simpleLOMs import plotting

        label = lom_label or type(self).__name__
        if reference is not None:
            plotting.plot_lom_vs_data(
                net, reference,
                m=m, n=n,
                lom_label=label,
                data_label=data_label,
                show=True,
                save_path=save_path,
            )
        else:
            plotting.plot_re_im(
                net, m=m, n=n,
                title=label,
                show=True,
                save_path=save_path,
            )

    @abstractmethod
    def get_network(self, freq: rf.Frequency, **kwargs) -> rf.Network:
        """
        Build and return an rf.Network using the fitted L and C values.

        Must call self._require_fitted() at the start.

        Parameters
        ----------
        freq : rf.Frequency
            May differ from the freq used in fit() (e.g. a finer grid).
        show : bool, optional
            If True, automatically plot the built network (default False).
            Defaults to False so batch callers (analyze_system, sweeps)
            do not spam figures.
        reference : rf.Network, optional
            Reference network for overlay plots when ``show=True``.
            OptimizedFit can fall back to the ``data_ntw`` stored during fit().
        m, n : int, optional
            S-parameter indices for the auto-plot (0-based).
        lom_label, data_label : str, optional
            Legend labels for the auto-plot.
        save_path : str, optional
            If provided with ``show=True``, save the figure to this path.
        **kwargs
            Subclass-specific parameters (e.g. Cc1, Cc2, Ctog1, Ctog2).

        Returns
        -------
        rf.Network
        """

    @abstractmethod
    def get_params(self) -> dict:
        """
        Return the fitted parameters as a plain dictionary.

        At minimum returns {"L": self.L, "C": self.C}.
        Subclasses may include additional diagnostics.

        Returns
        -------
        dict
        """