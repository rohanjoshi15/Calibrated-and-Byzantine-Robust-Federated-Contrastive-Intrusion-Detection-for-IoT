"""Data locations, overridable with the DATA_PROCESSED and DATA_RAW environment variables."""
import os

PROCESSED = os.environ.get("DATA_PROCESSED", os.path.join("data", "processed"))
RAW = os.environ.get("DATA_RAW", os.path.join("data", "raw"))
