"""Support code shared across pipeline stages.

Nothing here is a stage. These modules hold the pieces the numbered `l01_`-`l12_`
stages lean on — exception types, logging, chart rendering, and the parsers for
turning a messy export into typed columns — so that a stage file reads as the step
it performs rather than as plumbing.
"""
