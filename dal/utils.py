"""
DAL — Shared Serialization Utilities

Pure utility module — no domain logic, no DAL-specific imports.
Importable by both dal/output_writer.py and governance/audit_logger.py.

See ADR-010 for design rationale.

No Streamlit imports. No BLL imports. No external API calls.
"""


def make_serializable(obj):
    """
    Recursively convert non-JSON-serializable types to serializable equivalents.

    Conversions:
      frozenset → sorted list  (deterministic order, JSON-serializable)
      dict      → dict         (recurse into values)
      list      → list         (recurse into items)
      all else  → unchanged    (str, int, float, bool, None are already serializable)

    Args:
        obj: Any Python object, typically the output of Pydantic's model_dump().

    Returns:
        A JSON-serializable equivalent of obj.

    Examples:
        make_serializable(frozenset({"Germany", "Austria"}))
            → ["Austria", "Germany"]  # sorted

        make_serializable({"countries": frozenset({"UK", "US"}), "score": 0.9})
            → {"countries": ["UK", "US"], "score": 0.9}
    """
    if isinstance(obj, frozenset):
        return sorted(obj)
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    return obj
