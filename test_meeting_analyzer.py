import pytest
from unittest.mock import Mock, patch
import json
import os
from meeting_analyzer import MeetingAnalyzer

@pytest.fixture
def price_file():
    """Create a temporary price file for testing."""
    test_prices = [
        {
            "API Provider": "Google",
            "Model": "Gemini 1.5 Pro",
            "Input": 0.075,
            "Output": 0.30
        }
    ]
    filename = "test_prices.json"
    with open(filename, "w") as f:
        json.dump(test_prices, f)
    
    yield filename
    
    # Cleanup
    if os.path.exists(filename):
        os.remove(filename)

@pytest.fixture
def transcript_file():
    """Create a temporary transcript file for testing."""
    content = "This is a test transcript."
    filename = "test_transcript.md"
    with open(filename, "w") as f:
        f.write(content)
    
    yield filename
    
    # Cleanup
    if os.path.exists(filename):
        os.remove(filename)

@pytest.fixture
def mock_gemini_response():
    """Create a mock Gemini API response."""
    response = Mock()
    response.text = json.dumps({
        "summary": "Test summary",
        "actions": ["Action 1"],
        "positives": ["Positive 1"],
        "improvements": ["Improvement 1"]
    })
    response.usage_metadata = Mock()
    response.usage_metadata.prompt_token_count = 100
    response.usage_metadata.candidates_token_count = 50
    return response

def test_load_prices(price_file):
    """Test loading prices from JSON file."""
    analyzer = MeetingAnalyzer("test_api_key", price_file)
    assert analyzer.prices["input"] == 0.075
    assert analyzer.prices["output"] == 0.30

@patch('google.generativeai.GenerativeModel')
def test_analyze_meeting_token_counting(mock_model, price_file, transcript_file, mock_gemini_response):
    """Test token counting and cost calculation."""
    # Set up mock
    mock_model.return_value.generate_content.return_value = mock_gemini_response
    
    # Run analyzer
    analyzer = MeetingAnalyzer("test_api_key", price_file)
    results, costs = analyzer.analyze_meeting(transcript_file)
    
    # Check results structure
    assert "summary" in results
    assert "actions" in results
    assert "positives" in results
    assert "improvements" in results
    
    # Check cost calculations
    expected_input_cost = (100 / 1000) * 0.075  # 100 tokens at $0.075 per 1000
    expected_output_cost = (50 / 1000) * 0.30   # 50 tokens at $0.30 per 1000
    expected_total_cost = expected_input_cost + expected_output_cost
    
    assert costs["input_cost"] == pytest.approx(expected_input_cost)
    assert costs["output_cost"] == pytest.approx(expected_output_cost)
    assert costs["total_cost"] == pytest.approx(expected_total_cost)

def test_price_file_not_found():
    """Test error handling for missing price file."""
    with pytest.raises(FileNotFoundError):
        MeetingAnalyzer("test_api_key", "nonexistent_file.json")

def test_gemini_price_not_found():
    """Test error handling for missing Gemini pricing."""
    # Create price file without Gemini pricing
    wrong_prices = [{"Model": "Wrong Model", "Input": 1.0, "Output": 1.0}]
    filename = "wrong_prices.json"
    with open(filename, "w") as f:
        json.dump(wrong_prices, f)
    
    try:
        with pytest.raises(ValueError, match="Gemini 1.5 Pro pricing not found in price file"):
            MeetingAnalyzer("test_api_key", filename)
    finally:
        if os.path.exists(filename):
            os.remove(filename)

def test_invalid_json_response(price_file, transcript_file):
    """Test handling of invalid JSON in API response."""
    with patch('google.generativeai.GenerativeModel') as mock_model:
        # Create response with invalid JSON
        mock_response = Mock()
        mock_response.text = "Invalid JSON"
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        
        mock_model.return_value.generate_content.return_value = mock_response
        
        analyzer = MeetingAnalyzer("test_api_key", price_file)
        results, costs = analyzer.analyze_meeting(transcript_file)
        
        # Check that results are empty but structured correctly
        assert results["summary"] == ""
        assert results["actions"] == []
        assert results["positives"] == []
        assert results["improvements"] == []
        
        # Costs should still be calculated
        assert costs["total_cost"] > 0
