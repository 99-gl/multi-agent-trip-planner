# workflow/__init__.py
from .graph import build_graph

_workflow = None

def get_trip_planner_workflow():
    global _workflow
    if _workflow is None:
        _workflow = build_graph()
    return _workflow
