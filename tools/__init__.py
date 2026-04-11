"""Namespace package for self-contained segmentation tool plugins.

Each subpackage under ``tools/`` is an independent plugin with its own
``logic.py`` (plugin entrypoint), ``infer.py`` (inference code),
``tool.json`` (plugin manifest), and ``config.yaml`` (settings).

Tools are intended to be loaded and invoked individually from outside this
package (e.g. by a plugin host that reads each ``tool.json`` and imports
the corresponding ``logic:execute`` entrypoint). There is no central
registry here by design.
"""
