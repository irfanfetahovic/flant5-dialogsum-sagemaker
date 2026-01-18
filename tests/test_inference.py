"""
Unit tests for inference functions.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
from src.inference import summarize_dialogue


class TestInferenceFunctions:
    """Test suite for inference functions."""

    @pytest.fixture
    def mock_model_and_tokenizer(self):
        """Create mock model and tokenizer."""
        model = MagicMock()
        tokenizer = MagicMock()

        # Mock tokenizer behavior
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.return_value = "Generated summary text"
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        # Mock model behavior
        model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        return model, tokenizer

    def test_summarize_dialogue_valid_input(self, mock_model_and_tokenizer):
        """Test summarization with valid input."""
        model, tokenizer = mock_model_and_tokenizer
        dialogue = "Alice: Hello. Bob: Hi, how are you? Alice: I'm doing great!"

        result = summarize_dialogue(model, tokenizer, dialogue)

        assert isinstance(result, str)
        assert len(result) > 0
        assert result == "Generated summary text"

    def test_summarize_dialogue_empty_input(self, mock_model_and_tokenizer):
        """Test summarization with empty dialogue."""
        model, tokenizer = mock_model_and_tokenizer

        # Empty dialogue should still work but return empty summary
        result = summarize_dialogue(model, tokenizer, "")

        assert isinstance(result, str)

    def test_summarize_dialogue_long_input(self, mock_model_and_tokenizer):
        """Test summarization with very long input."""
        model, tokenizer = mock_model_and_tokenizer

        # Create a very long dialogue
        long_dialogue = " ".join(
            ["Speaker A: Hello. " * 100, "Speaker B: Response. " * 100]
        )

        result = summarize_dialogue(model, tokenizer, long_dialogue)

        assert isinstance(result, str)
        model.generate.assert_called_once()

    def test_summarize_dialogue_with_special_characters(self, mock_model_and_tokenizer):
        """Test summarization with special characters."""
        model, tokenizer = mock_model_and_tokenizer

        dialogue = "Alice: How are you? 😊\nBob: Great! 🎉\nAlice: That's awesome! 💯"

        result = summarize_dialogue(model, tokenizer, dialogue)

        assert isinstance(result, str)

    def test_summarize_dialogue_custom_max_tokens(self, mock_model_and_tokenizer):
        """Test summarization with custom max_new_tokens."""
        model, tokenizer = mock_model_and_tokenizer
        dialogue = "Alice: Hello Bob. Bob: Hello Alice."

        summarize_dialogue(model, tokenizer, dialogue, max_new_tokens=50)

        # Verify generate was called with max_new_tokens
        model.generate.assert_called_once()
        call_kwargs = model.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 50

    def test_summarize_dialogue_with_beam_search(self, mock_model_and_tokenizer):
        """Test summarization with beam search."""
        model, tokenizer = mock_model_and_tokenizer
        dialogue = "Alice: How's the project? Bob: Going great!"

        summarize_dialogue(model, tokenizer, dialogue, num_beams=4)

        # Verify generate was called with num_beams
        model.generate.assert_called_once()
        call_kwargs = model.generate.call_args[1]
        assert call_kwargs["num_beams"] == 4

    def test_summarize_dialogue_prompt_format(self, mock_model_and_tokenizer):
        """Test that prompt is formatted correctly."""
        model, tokenizer = mock_model_and_tokenizer
        dialogue = "A: Hello. B: Hi."

        summarize_dialogue(model, tokenizer, dialogue)

        # Verify tokenizer was called with correctly formatted prompt
        call_args = tokenizer.call_args
        prompt = call_args[0][0] if call_args[0] else None

        assert prompt is not None
        assert "Summarize" in prompt or "summarize" in prompt.lower()
        assert "A: Hello. B: Hi." in prompt


class TestInferenceIntegration:
    """Integration tests for inference."""

    @pytest.mark.skip(reason="Requires model download - enable for full tests")
    def test_real_model_inference(self):
        """Test with real model (skipped by default)."""
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        model_id = "google/flan-t5-small"  # Use small model for testing
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        dialogue = "Alice: How are you? Bob: I'm doing well!"
        result = summarize_dialogue(model, tokenizer, dialogue)

        assert isinstance(result, str)
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
