"""
DAL — InputLoader

Loads Source A (CRM) and Source B (Core Banking) from CSV or JSON.
Raises typed exceptions on empty files or missing required fields — fail-fast
before any BLL processing begins.

See ADR-003 for design rationale.

No Streamlit imports. No BLL imports. No external API calls.
"""

import json
import pandas as pd


# --- Custom Exceptions ---

class EmptyInputError(Exception):
    """Raised when a source file contains zero records."""
    pass


class InputValidationError(Exception):
    """Raised when required fields are missing from the source file."""
    pass


# --- InputLoader ---

class InputLoader:
    """
    Loads company records from CSV or JSON files.

    Required fields in every source file: 'source_id', 'source_name'.
    Raises EmptyInputError on 0 entries.
    Raises InputValidationError on missing required fields.
    """

    REQUIRED_FIELDS = ("source_id", "source_name")

    def load_from_csv(self, path: str) -> list[dict]:
        """
        Load records from a CSV file.

        Args:
            path: Path to the CSV file.

        Returns:
            List of dicts with at least 'source_id' and 'source_name' keys.

        Raises:
            FileNotFoundError: If the file does not exist.
            EmptyInputError: If the file contains zero records.
            InputValidationError: If required fields are missing.
        """
        try:
            df = pd.read_csv(path, dtype=str)
        except FileNotFoundError:
            raise FileNotFoundError(f"Source file not found: {path}")

        if df.empty:
            raise EmptyInputError(f"Source file contains no records: {path}")

        missing = [f for f in self.REQUIRED_FIELDS if f not in df.columns]
        if missing:
            raise InputValidationError(
                f"Missing required fields {missing} in: {path}. "
                f"Found columns: {list(df.columns)}"
            )

        # Strip whitespace from string fields
        for col in self.REQUIRED_FIELDS:
            df[col] = df[col].astype(str).str.strip()

        # Drop rows where required fields are null/empty after stripping
        before = len(df)
        df = df[df["source_id"].str.len() > 0]
        df = df[df["source_name"].str.len() > 0]
        if df.empty:
            raise EmptyInputError(
                f"Source file has {before} row(s) but all have empty required fields: {path}"
            )

        return df.to_dict(orient="records")

    def load_from_json(self, path: str) -> list[dict]:
        """
        Load records from a JSON file (list of objects).

        Args:
            path: Path to the JSON file.

        Returns:
            List of dicts with at least 'source_id' and 'source_name' keys.

        Raises:
            FileNotFoundError: If the file does not exist.
            EmptyInputError: If the file contains zero records.
            InputValidationError: If required fields are missing.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Source file not found: {path}")

        if not isinstance(data, list):
            raise InputValidationError(
                f"JSON source file must contain a list of objects, got: {type(data).__name__} in: {path}"
            )

        if len(data) == 0:
            raise EmptyInputError(f"Source file contains no records: {path}")

        # Validate required fields on the first record as a representative check
        first = data[0]
        missing = [f for f in self.REQUIRED_FIELDS if f not in first]
        if missing:
            raise InputValidationError(
                f"Missing required fields {missing} in: {path}. "
                f"Found keys in first record: {list(first.keys())}"
            )

        # Filter out records with empty required fields
        valid = [
            r for r in data
            if str(r.get("source_id", "")).strip() and str(r.get("source_name", "")).strip()
        ]
        if not valid:
            raise EmptyInputError(
                f"Source file has {len(data)} record(s) but all have empty required fields: {path}"
            )

        return valid

    def load(self, path: str) -> list[dict]:
        """
        Auto-detect format from file extension and load accordingly.

        Args:
            path: Path to the source file (.csv or .json).

        Returns:
            List of dicts.

        Raises:
            ValueError: If the file extension is not .csv or .json.
        """
        lower = path.lower()
        if lower.endswith(".csv"):
            return self.load_from_csv(path)
        elif lower.endswith(".json"):
            return self.load_from_json(path)
        else:
            raise ValueError(
                f"Unsupported file format. Expected .csv or .json, got: {path}"
            )
