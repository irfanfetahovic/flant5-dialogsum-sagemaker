# Contributing to FLAN-T5 Dialog Summarization

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Report issues professionally
- Help others learn and grow

## Getting Started

### Fork & Clone

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/flan-t5-base-dialogsum-sagemaker.git
cd flan-t5-base-dialogsum-sagemaker

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/flan-t5-base-dialogsum-sagemaker.git
```

### Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features
- `bugfix/` - Bug fixes
- `docs/` - Documentation updates
- `test/` - Test additions

### 2. Make Changes

- Keep commits focused and atomic
- Write descriptive commit messages
- Add tests for new functionality

### 3. Code Quality

#### Format Code

```bash
black src/ scripts/ tests/
```

#### Lint Code

```bash
flake8 src/ scripts/ tests/
```

#### Type Checking

```bash
mypy src/
```

#### Run Tests

```bash
pytest tests/ -v
```

### 4. Documentation

- Update README.md if functionality changes
- Add docstrings to all functions (following existing style)
- Include examples for new public APIs

### 5. Commit & Push

```bash
git add .
git commit -m "feat: add feature X

- Detailed explanation of changes
- Why this change was needed
"

git push origin feature/your-feature-name
```

### 6. Create Pull Request

- Go to GitHub and create a Pull Request
- Describe what your PR does
- Reference any related issues (#123)
- Ensure CI/CD checks pass

## Types of Contributions

### 🐛 Bug Reports

Found a bug? Please create an issue with:

- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment (Python version, OS, dependencies)
- Error logs/traceback

### Feature Requests

Have an idea? Submit an issue with:

- Clear description of the feature
- Use case and motivation
- Proposed implementation (optional)
- Example usage

### 📚 Documentation

Help improve docs:

- Fix typos or unclear explanations
- Add examples
- Improve installation instructions
- Add troubleshooting guides

### 🧪 Tests

Improve test coverage:

- Add unit tests for new functions
- Add integration tests
- Test edge cases
- Test error conditions

## Code Style Guide

### Python

- Follow PEP 8
- Use type hints for functions
- Write descriptive variable names
- Max line length: 100 characters

### Example

```python
def summarize_dialogue(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    dialogue: str,
    max_new_tokens: int = 200
) -> str:
    """
    Summarize a dialogue using the fine-tuned model.
    
    Args:
        model: Transformer model for inference
        tokenizer: Model tokenizer
        dialogue: Input dialogue text
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Summary text
        
    Raises:
        ValueError: If dialogue is empty
    """
    if not dialogue.strip():
        raise ValueError("Dialogue cannot be empty")
    
    # Implementation...
    return summary
```

## Testing Guidelines

- Write tests for all new functions
- Test both happy path and error cases
- Use descriptive test names
- Target 80%+ code coverage

```python
def test_summarize_dialogue_valid_input():
    """Test summarization with valid input."""
    model, tokenizer = load_test_models()
    dialogue = "Alice: Hello. Bob: Hi, how are you?"
    
    result = summarize_dialogue(model, tokenizer, dialogue)
    
    assert isinstance(result, str)
    assert len(result) > 0

def test_summarize_dialogue_empty_input():
    """Test summarization with empty input."""
    model, tokenizer = load_test_models()
    
    with pytest.raises(ValueError):
        summarize_dialogue(model, tokenizer, "")
```

## PR Review Process

- Maintainers will review PRs within 3-5 business days
- Feedback will be constructive and timely
- Multiple iterations may be needed
- All CI/CD checks must pass

## Release Process

Releases follow semantic versioning:

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

## Questions?

- Open a GitHub Discussion
- Email: [your-email]
- Check existing issues first

Thank you for contributing! 🙌
