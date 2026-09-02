"""Safety semantics layered on the shared local C-space query."""

from curvenav.configuration_space import (
    CONFIGURATION_FIELD_CHANNELS,
    ConfigurationFieldQuery,
    query_configuration_field,
)
from curvenav.physical import EXTRA_CLEARANCE_M


SAFETY_CLEARANCE_M = EXTRA_CLEARANCE_M

__all__ = [
    "CONFIGURATION_FIELD_CHANNELS",
    "ConfigurationFieldQuery",
    "SAFETY_CLEARANCE_M",
    "query_configuration_field",
]
