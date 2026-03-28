"""
DAL — InputSanitizer

Minimal-intervention sanitizer for raw company name strings.
Fixes encoding integrity and removes problematic Unicode characters
(control chars, null bytes, zero-width chars, BOM) without touching
valid content: German umlauts, ampersands, dots, and other punctuation
are preserved — those are the normalizer's responsibility.

Returns (cleaned_name, was_modified: bool) so the caller can selectively
log an `input_sanitized` audit event only when the input was actually changed.

See ADR-007 for design rationale.

No Streamlit imports. No BLL imports. No external API calls.
"""

import re


# ---------------------------------------------------------------------------
# Problematic Unicode character ranges to strip
#
# Kept:   \t (0x09), \n (0x0a), \r (0x0d) — handled by whitespace split
# Stripped:
#   0x00–0x08  : C0 control chars (NUL, SOH, STX, ETX, EOT, ENQ, ACK, BEL, BS)
#   0x0b       : vertical tab
#   0x0c       : form feed
#   0x0e–0x1f  : C0 control chars (SO through US)
#   0x7f       : DEL
#   0x200b     : zero-width space
#   0x200c     : zero-width non-joiner
#   0x200d     : zero-width joiner
#   0xfeff     : BOM / zero-width no-break space
#   0x00ad     : soft hyphen (invisible, causes string comparison issues)
# ---------------------------------------------------------------------------

_PROBLEMATIC = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f"
    r"\u200b\u200c\u200d\ufeff"
    r"\u00ad"
    r"]"
)


class InputSanitizer:
    """
    Minimal-intervention input sanitizer for company name strings.

    Fixes encoding artifacts and strips problematic Unicode characters.
    Does NOT strip punctuation, umlauts, or any content — that is the
    normalizer's responsibility.
    """

    def sanitize(self, name: str) -> tuple[str, bool]:
        """
        Sanitize a single company name string.

        Steps:
          1. Fix encoding via UTF-8 round-trip (errors='replace')
          2. Strip problematic Unicode characters
          3. Normalize whitespace (collapse tabs/newlines/multiple spaces)

        Args:
            name: Raw company name string (possibly with encoding artifacts).

        Returns:
            Tuple of (cleaned_name, was_modified).
            was_modified is True if the output differs from the input —
            caller should log an `input_sanitized` audit event in that case.
        """
        if not isinstance(name, str):
            name = str(name)

        original = name

        # Step 1: Fix encoding via UTF-8 round-trip
        name = name.encode("utf-8", errors="replace").decode("utf-8")

        # Step 2: Strip problematic Unicode characters
        name = _PROBLEMATIC.sub("", name)

        # Step 3: Normalize whitespace (tabs, newlines, multiple spaces → single space)
        name = " ".join(name.split())

        was_modified = name != original
        return name, was_modified

    def sanitize_batch(self, names: list[str]) -> list[tuple[str, bool]]:
        """
        Sanitize a list of company name strings.

        Args:
            names: List of raw company name strings.

        Returns:
            List of (cleaned_name, was_modified) tuples, same length and
            order as input.
        """
        return [self.sanitize(name) for name in names]
