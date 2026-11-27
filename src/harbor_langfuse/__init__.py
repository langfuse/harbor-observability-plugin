"""Langfuse plugin for Harbor.

Each trial becomes a Langfuse trace (its agent trajectory as observations), each
job an experiment run per agent and model on a Langfuse dataset of the job's
tasks, and each verifier reward a score on the trial's trace.
"""

from .export import TrialExport, export_trial
from .plugin import LangfusePlugin

__all__ = ["LangfusePlugin", "TrialExport", "export_trial"]
__version__ = "0.1.0"
