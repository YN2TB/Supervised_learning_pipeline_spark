"""Schema and column policy for the 2015 USDOT on-time performance feed.

Two things live here because they are decisions, not mechanics:

1. The explicit read schema. Inferring it would cost a second full pass
   over a 592 MB CSV, and would guess wrong on the HHMM clock columns
   (``0005`` must not become the integer 5 before we parse it).

2. The leakage policy: which columns may be seen by a model that predicts
   arrival delay, and which must never be.
"""
from __future__ import annotations

from pyspark.sql.types import IntegerType, StringType, StructField, StructType

# HHMM clock fields are read as strings so leading zeros survive to the parser.
_CLOCK = ("SCHEDULED_DEPARTURE", "DEPARTURE_TIME", "WHEELS_OFF",
          "WHEELS_ON", "SCHEDULED_ARRIVAL", "ARRIVAL_TIME")

_STRING = ("AIRLINE", "TAIL_NUMBER", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
           "CANCELLATION_REASON") + _CLOCK

_COLUMNS = [
    "YEAR", "MONTH", "DAY", "DAY_OF_WEEK", "AIRLINE", "FLIGHT_NUMBER",
    "TAIL_NUMBER", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE", "DEPARTURE_TIME", "DEPARTURE_DELAY", "TAXI_OUT",
    "WHEELS_OFF", "SCHEDULED_TIME", "ELAPSED_TIME", "AIR_TIME", "DISTANCE",
    "WHEELS_ON", "TAXI_IN", "SCHEDULED_ARRIVAL", "ARRIVAL_TIME",
    "ARRIVAL_DELAY", "DIVERTED", "CANCELLED", "CANCELLATION_REASON",
    "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY",
]

FLIGHTS_SCHEMA = StructType([
    StructField(c, StringType() if c in _STRING else IntegerType(), True)
    for c in _COLUMNS
])

# ---------------------------------------------------------------------------
# Leakage policy
# ---------------------------------------------------------------------------
# The prediction point is fixed at WHEELS-OFF: the aircraft has pushed back
# and left the ground, so departure delay and taxi-out are legitimately known.
# Everything that is only observable *after* wheels-off is a leak.
#
# The five *_DELAY attribution columns are the worst offenders: by the USDOT
# definition they sum to ARRIVAL_DELAY for any flight delayed >= 15 minutes,
# so a model given them can reconstruct the target exactly (R^2 ~ 1.0).
LEAKY_COLUMNS = [
    "ARRIVAL_TIME",          # the arrival event itself
    "ELAPSED_TIME",          # gate-to-gate actual, implies arrival
    "AIR_TIME",              # component of ELAPSED_TIME
    "WHEELS_ON",             # landing time, after wheels-off
    "TAXI_IN",               # after landing
    "AIR_SYSTEM_DELAY",      # -- the five attribution columns --
    "SECURITY_DELAY",
    "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "WEATHER_DELAY",
    "CANCELLATION_REASON",   # only populated for cancelled flights (dropped anyway)
    "DIVERTED", "CANCELLED", # filter conditions, not features
    "YEAR",                  # constant 2015, zero variance
    "ARRIVAL_DELAY",         # the target itself; kept only as label_delay
]

# Known at wheels-off, therefore allowed as model inputs.
ALLOWED_RAW_FEATURES = [
    "MONTH", "DAY", "DAY_OF_WEEK", "AIRLINE", "FLIGHT_NUMBER", "TAIL_NUMBER",
    "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE",
    "DEPARTURE_TIME", "DEPARTURE_DELAY", "TAXI_OUT", "WHEELS_OFF",
    "SCHEDULED_TIME", "DISTANCE", "SCHEDULED_ARRIVAL",
]

SEVERE_DELAY_THRESHOLD = 30  # minutes; the brief's binary target
