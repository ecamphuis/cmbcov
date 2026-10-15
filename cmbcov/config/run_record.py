"""
The record of what a run actually used: ``params_resolved.yml``.

The parameter file a run saves in its output directory (``params.yml``) holds
what was *asked for* and is what the next run compares itself with. This
module writes a second file beside it, ``params_resolved.yml``, with the
*effective* value of every parameter -- the defaults the file did not set
included -- plus two sections the file cannot hold:

``derived``
    Values the run computes rather than reads: the mask resolution, the
    multipole bins, the internal band limit of ACC, the kernel directory, the
    ``map_filter`` probe plan, the thread policy.
``provenance``
    The cmbcov version and git revision, the versions of the numerical
    libraries, the thread environment, and the start time and, once the run
    ends, the end time, wall time and status.

The key names of the parameters are those of the parameter file, so the two
files can be compared with ``diff``. ``params_resolved.yml`` is written once
the configuration is resolved (before any heavy computation, so a run that
crashes still has it) and rewritten as the run learns its derived values and
when it ends. It is never read back by the package, and ``params.yml`` is
unchanged by it.

Nothing bulky goes in: no spectra or matrices, and a list longer than
:data:`MAX_ITEMS` is replaced by a note of its length.
"""

from __future__ import annotations

import dataclasses
import datetime
import enum
import importlib
import importlib.metadata
import logging
import os
import platform
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from ..utils.file_utils import get_git_revision_short_hash
from ..utils.threading_utils import _THREAD_ENV_VARS

if TYPE_CHECKING:
    from .validation import PipelineConfig

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_ITEMS",
    "PRECOMPUTE_RESOLVED_NAME",
    "RECORD_NAMES",
    "RESOLVED_NAME",
    "RunRecord",
    "provenance",
    "resolved_parameters",
    "yaml_safe",
]

#: File name of the run record, in the run's output (version) directory.
RESOLVED_NAME = "params_resolved.yml"

#: File name of the record of ``cmbcov-precompute``, in the kernel directory.
PRECOMPUTE_RESOLVED_NAME = "precompute_params_resolved.yml"

#: The names of the record files, which are not parameter files: a version
#: directory holding one is still one that holds a single parameter file
#: (``ParameterManager._find_saved_parameter_file``).
RECORD_NAMES = (RESOLVED_NAME, PRECOMPUTE_RESOLVED_NAME)

#: Longest list or array written out; a longer one is replaced by a note.
MAX_ITEMS = 512

#: The libraries whose versions are recorded.
_LIBRARIES = ("numpy", "scipy", "ducc0", "healpy")

_HEADER = (
    "# Effective parameters of this run: the parameter file's values and the\n"
    "# defaults it did not set (same keys as {params}), then 'derived' (values\n"
    "# the run computed) and 'provenance' (versions, environment, timing).\n"
    "# Written by cmbcov; nothing reads it back.\n"
)


# --------------------------------------------------------------------------- #
# YAML-safe values
# --------------------------------------------------------------------------- #
def yaml_safe(value: Any) -> Any:
    """
    ``value`` as plain Python that ``yaml.safe_dump`` writes and
    ``yaml.safe_load`` reads back: numpy scalars become Python scalars,
    tuples, sets and small arrays become lists, mappings get string keys,
    dataclasses and enum members their plain form, paths strings. A list or
    array of more than :data:`MAX_ITEMS` entries becomes a note of its length
    (``"<N values omitted>"``), and an object of another type its ``repr``.
    """
    if isinstance(value, np.generic):
        # first: np.float64 is a float subclass that yaml cannot represent
        return yaml_safe(value.item())
    if isinstance(value, enum.Enum):
        return yaml_safe(value.value)
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not np.isfinite(value):
            return str(float(value))
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, np.ndarray):
        if value.size > MAX_ITEMS:
            return f"<{value.size} values omitted>"
        return yaml_safe(value.tolist())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: yaml_safe(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(k): yaml_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset, range)):
        items = sorted(value) if isinstance(value, (set, frozenset)) else list(value)
        if len(items) > MAX_ITEMS:
            return f"<{len(items)} values omitted>"
        return [yaml_safe(v) for v in items]
    return repr(value)


# --------------------------------------------------------------------------- #
# The parameters
# --------------------------------------------------------------------------- #
def resolved_parameters(
    config: PipelineConfig, *, acc_precompute_defaults: bool = False
) -> dict[str, Any]:
    """
    Every parameter of ``config`` with its effective value, under the keys of
    the parameter file.

    That is the effective record ``config.params`` (the file's values with
    alias spellings normalised, the defaults of
    :attr:`~cmbcov.config.validation.ParameterValidator.OPTIONAL_PARAMS`
    applied, ``save_dir`` and ``git_hash`` added), with these completed from
    the configuration objects: ``stokes`` and ``observables`` as resolved,
    ``dmax`` and ``centralell`` (``null`` when the method has none),
    ``post_process_correction``, and the blocks ``map_filter`` and
    ``acc_precompute`` with every key of the block, the defaults included.

    Parameters
    ----------
    config : PipelineConfig
    acc_precompute_defaults : bool, default False
        Write the ``acc_precompute`` block with its defaults even when the
        file has none (for the record of ``cmbcov-precompute``, which uses
        them).

    Returns
    -------
    dict
        YAML-safe (:func:`yaml_safe`), in the order of the record.
    """
    resolved: dict[str, Any] = dict(config.params)
    resolved["stokes"] = list(config.stokes)
    resolved["observables"] = list(config.observables)
    for name in ("dmax", "centralell"):
        resolved[name] = getattr(config, name)
    resolved["post_process_correction"] = config.post_process_correction
    if config.map_filter is not None:
        resolved["map_filter"] = _block(config.map_filter, config.map_filter.KEYS)
    settings = config.acc_precompute
    if settings is None and acc_precompute_defaults:
        from .validation import AccPrecomputeConfig

        settings = AccPrecomputeConfig()
    if settings is not None:
        resolved["acc_precompute"] = _block(settings, settings.KEYS)
    return yaml_safe(resolved)


def _block(block: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    """A parameter block (a frozen dataclass) as ``{key: value}`` in file order."""
    return {key: getattr(block, key) for key in keys}


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def _version(name: str) -> str | None:
    """The version of an installed library, or ``None`` if it is not there."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        return str(importlib.import_module(name).__version__)
    except (ImportError, AttributeError):
        return None


def provenance() -> dict[str, Any]:
    """
    Where and with what the run happens: the cmbcov version and git
    revision, the Python and library versions, the thread environment (the
    variables of ``utils.threading_utils`` that are set, and the CPU count).
    No host name and no user paths.
    """
    return {
        "cmbcov_version": _version("cmbcov"),
        "git_revision": get_git_revision_short_hash(),
        "python": platform.python_version(),
        **{name: _version(name) for name in _LIBRARIES},
        "cpu_count": os.cpu_count(),
        "thread_environment": {
            name: os.environ[name] for name in _THREAD_ENV_VARS if name in os.environ
        },
    }


def _now() -> str:
    """The current local time, ISO 8601 with its UTC offset, to the second."""
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #
class RunRecord:
    """
    The content and the file of one run's ``params_resolved.yml``.

    Create it when the configuration is resolved, :meth:`write` it, add what
    the run computes with :meth:`add_derived` (each call rewrites the file),
    and :meth:`finish` it when the run ends. A failure to write the file is
    logged as a warning and never stops the run.

    Parameters
    ----------
    path : str
        Where the file goes.
    resolved : dict
        The parameters, :func:`resolved_parameters`.
    params_name : str, default "params.yml"
        Name of the parameter file it sits beside, for the header comment.
    clock : callable, optional
        A monotonic clock, for the wall time (injectable for tests).
    """

    def __init__(
        self,
        path: str,
        resolved: dict[str, Any],
        params_name: str = "params.yml",
        clock=time.monotonic,
    ) -> None:
        self.path = path
        self._params_name = params_name
        self._resolved = resolved
        self._derived: dict[str, Any] = {}
        self._provenance = provenance()
        self._provenance["start_time"] = _now()
        self._provenance["status"] = "running"
        self._clock = clock
        self._start = clock()

    @classmethod
    def for_config(
        cls,
        path: str,
        config: PipelineConfig,
        params_name: str = "params.yml",
        **kwargs: Any,
    ) -> RunRecord:
        """The record of ``config`` to be written at ``path``; ``kwargs`` go to
        :func:`resolved_parameters`."""
        return cls(path, resolved_parameters(config, **kwargs), params_name=params_name)

    # ------------------------------------------------------------------
    @property
    def content(self) -> dict[str, Any]:
        """The record as it is written: parameters, ``derived``, ``provenance``."""
        return {
            **self._resolved,
            "derived": self._derived,
            "provenance": self._provenance,
        }

    def add_derived(self, **values: Any) -> None:
        """Add (or replace) entries of the ``derived`` section and rewrite the file."""
        self._derived.update(yaml_safe(values))
        self.write()

    def finish(self, status: str, error: str | None = None) -> None:
        """
        Close the record: ``status`` (``"completed"``, ``"dry run"`` or
        ``"failed"``), the end time, the wall time in seconds and, for a
        failure, the name of the exception.
        """
        self._provenance["status"] = status
        self._provenance["end_time"] = _now()
        self._provenance["wall_time_s"] = round(self._clock() - self._start, 1)
        if error is not None:
            self._provenance["error"] = error
        self.write()

    def write(self) -> None:
        """Write the file (atomically: to a temporary file, then renamed)."""
        text = _HEADER.format(params=self._params_name) + yaml.safe_dump(
            self.content, default_flow_style=False, sort_keys=False
        )
        tmp = f"{self.path}.tmp"
        try:
            with open(tmp, "w") as handle:
                handle.write(text)
            os.replace(tmp, self.path)
        except OSError as exc:
            logger.warning("could not write %s: %s", self.path, exc)
