"""Segmentation tool registry.

Each subpackage under ``tools/`` exposes a ``run_inference(cfg)`` function in
its ``infer`` module. To run a tool, use the Hydra entry point::

    python run.py tool=<name>

where ``<name>`` matches one of the keys in ``AVAILABLE_TOOLS`` below and a
matching ``configs/tool/<name>.yaml`` exists.

You can also import programmatically::

    from tools import get_tool
    run_inference = get_tool("spleen_seg")
    result = run_inference(cfg)
"""

from importlib import import_module
from typing import Callable

AVAILABLE_TOOLS = [
    "brain_tumor_seg",
    "cxr_lung_seg",
    "pancreas_tumor_seg",
    "spleen_seg",
    "nnunet_amos",
]


def get_tool(name: str) -> Callable:
    """Return the ``run_inference`` callable for the given tool name."""
    if name not in AVAILABLE_TOOLS:
        raise ValueError(
            f"Unknown tool '{name}'. Available: {AVAILABLE_TOOLS}"
        )
    return import_module(f"tools.{name}.infer").run_inference
