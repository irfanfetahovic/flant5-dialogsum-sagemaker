"""
Unit tests for dataset utilities.
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
from src.dataset_utils import save_jsonl, load_jsonl


class TestDatasetUtils:
    """Test suite for dataset utilities."""

    def test_save_and_load_jsonl(self):
        """Test saving and loading JSONL files."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.jsonl"

            # Create test data
            test_data = [
                {"id": 1, "text": "Hello world", "label": "greeting"},
                {"id": 2, "text": "Goodbye", "label": "farewell"},
                {"id": 3, "text": "How are you?", "label": "question"},
            ]

            # Save to JSONL
            with open(filepath, "w") as f:
                for item in test_data:
                    f.write(json.dumps(item) + "\n")

            # Load from JSONL
            loaded_data = load_jsonl(str(filepath))

            # Verify
            assert len(loaded_data) == 3
            assert loaded_data[0]["id"] == 1
            assert loaded_data[1]["text"] == "Goodbye"
            assert loaded_data[2]["label"] == "question"

    def test_load_empty_jsonl(self):
        """Test loading empty JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "empty.jsonl"

            # Create empty file
            filepath.touch()

            # Load
            loaded_data = load_jsonl(str(filepath))

            # Verify
            assert loaded_data == []

    def test_jsonl_with_special_characters(self):
        """Test JSONL with special characters and unicode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "unicode.jsonl"

            # Create test data with special characters
            test_data = [
                {"text": "Hello 世界", "lang": "mixed"},
                {"text": "Здравствуй мир", "lang": "russian"},
                {"text": "🚀 Emoji test 🎉", "lang": "emoji"},
            ]

            # Save
            with open(filepath, "w", encoding="utf-8") as f:
                for item in test_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            # Load
            loaded_data = load_jsonl(str(filepath))

            # Verify
            assert len(loaded_data) == 3
            assert loaded_data[0]["text"] == "Hello 世界"
            assert "🚀" in loaded_data[2]["text"]

    def test_jsonl_large_file(self):
        """Test handling large JSONL files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "large.jsonl"

            # Create large dataset (1000 records)
            num_records = 1000
            with open(filepath, "w") as f:
                for i in range(num_records):
                    record = {"id": i, "value": f"record_{i}"}
                    f.write(json.dumps(record) + "\n")

            # Load
            loaded_data = load_jsonl(str(filepath))

            # Verify
            assert len(loaded_data) == num_records
            assert loaded_data[0]["id"] == 0
            assert loaded_data[-1]["id"] == num_records - 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
