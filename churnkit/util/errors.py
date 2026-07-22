"""Exception types. Every failure the user can cause should be one of these."""


class ChurnKitError(Exception):
    """Base class for all expected, user-actionable failures."""


class ConfigError(ChurnKitError):
    """A config file is missing a required value or contains a contradiction."""


class DataError(ChurnKitError):
    """The data on disk does not match what the config promised."""


class LeakageError(ChurnKitError):
    """A BLOCK-level leakage finding under `leakage.on_block: fail`."""


class InsufficientDataError(ChurnKitError):
    """The panel is too small or too one-sided to model."""
