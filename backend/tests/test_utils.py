""
Test utilities for the Crop Health Prediction System.
"""
import os
import json
import pytest
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import status
from fastapi.testclient import TestClient

# Test data paths
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_IMAGES_DIR = TEST_DATA_DIR / "images"

# Ensure test data directories exist
os.makedirs(TEST_IMAGES_DIR, exist_ok=True)

def create_test_image(filename: str = "test.jpg", size: tuple = (100, 100), color: tuple = (0, 128, 0)) -> Path:
    """
    Create a test image file.
    
    Args:
        filename: Name of the image file
        size: Image dimensions (width, height)
        color: RGB color tuple
        
    Returns:
        Path to the created image file
    """
    from PIL import Image, ImageDraw
    
    # Create a simple image
    img = Image.new('RGB', size, color=color)
    draw = ImageDraw.Draw(img)
    
    # Add some content to make the image non-uniform
    for i in range(0, size[0], 20):
        draw.line([(i, 0), (i, size[1])], fill=(255, 0, 0), width=1)
    
    # Save the image
    filepath = TEST_IMAGES_DIR / filename
    img.save(filepath)
    
    return filepath

def load_test_data(filename: str) -> Dict[str, Any]:
    """
    Load test data from a JSON file.
    
    Args:
        filename: Name of the JSON file in the test_data directory
        
    Returns:
        Dictionary containing the test data
    """
    filepath = TEST_DATA_DIR / filename
    with open(filepath, 'r') as f:
        return json.load(f)

def assert_response(
    response,
    expected_status: int = status.HTTP_200_OK,
    expected_keys: Optional[list] = None,
    expected_type: type = dict
) -> Dict[str, Any]:
    """
    Assert that a response has the expected status code and structure.
    
    Args:
        response: TestClient response object
        expected_status: Expected HTTP status code
        expected_keys: List of keys expected in the response JSON
        expected_type: Expected type of the response JSON
        
    Returns:
        The parsed JSON response
    """
    assert response.status_code == expected_status, \
        f"Expected status {expected_status}, got {response.status_code}. Response: {response.text}"
    
    # For 204 No Content responses, return empty dict
    if expected_status == status.HTTP_204_NO_CONTENT:
        return {}
    
    # Parse JSON response
    try:
        data = response.json()
    except ValueError as e:
        pytest.fail(f"Response is not valid JSON: {response.text}")
    
    # Check response type
    assert isinstance(data, expected_type), \
        f"Expected response to be {expected_type}, got {type(data)}: {data}"
    
    # Check for expected keys
    if expected_keys and isinstance(data, dict):
        for key in expected_keys:
            assert key in data, f"Expected key '{key}' not found in response: {data}"
    
    return data

def assert_pagination(response_data: Dict[str, Any], expected_page: int = 1, expected_per_page: int = 10):
    """
    Assert that a paginated response has the expected structure.
    
    Args:
        response_data: Parsed JSON response
        expected_page: Expected current page number
        expected_per_page: Expected number of items per page
    """
    assert "items" in response_data, "Paginated response missing 'items' key"
    assert "total" in response_data, "Paginated response missing 'total' key"
    assert "page" in response_data, "Paginated response missing 'page' key"
    assert "per_page" in response_data, "Paginated response missing 'per_page' key"
    
    assert isinstance(response_data["items"], list), "'items' should be a list"
    assert isinstance(response_data["total"], int), "'total' should be an integer"
    assert response_data["page"] == expected_page, f"Expected page {expected_page}, got {response_data['page']}"
    assert response_data["per_page"] == expected_per_page, f"Expected {expected_per_page} items per page, got {response_data['per_page']}"

def assert_error_response(
    response,
    expected_status: int,
    expected_detail: Optional[str] = None,
    expected_message: Optional[str] = None
):
    """
    Assert that an error response has the expected status and error details.
    
    Args:
        response: TestClient response object
        expected_status: Expected HTTP status code
        expected_detail: Expected error detail
        expected_message: Expected error message
    """
    data = assert_response(response, expected_status=expected_status)
    
    if expected_detail is not None:
        assert "detail" in data, "Error response missing 'detail' key"
        
        if isinstance(expected_detail, str):
            assert data["detail"] == expected_detail, \
                f"Expected error detail '{expected_detail}', got '{data['detail']}'"
        elif isinstance(expected_detail, list):
            assert isinstance(data["detail"], list), "Expected 'detail' to be a list"
            assert any(detail["msg"] == expected_detail for detail in data["detail"]), \
                f"Expected error detail '{expected_detail}' not found in {data['detail']}"
    
    if expected_message is not None:
        assert "message" in data, "Error response missing 'message' key"
        assert data["message"] == expected_message, \
            f"Expected error message '{expected_message}', got '{data['message']}'"

def assert_validation_error(response, field: str, msg: str):
    """
    Assert that a validation error occurred for a specific field.
    
    Args:
        response: TestClient response object
        field: Name of the field with validation error
        msg: Expected error message
    """
    data = assert_response(response, status.HTTP_422_UNPROCESSABLE_ENTITY)
    
    assert "detail" in data, "Validation error response missing 'detail' key"
    assert isinstance(data["detail"], list), "Expected 'detail' to be a list"
    
    # Check if any error matches the expected field and message
    field_errors = [
        error for error in data["detail"] 
        if error.get("loc") and error["loc"][-1] == field
    ]
    
    assert field_errors, f"No validation error found for field '{field}'"
    
    if msg is not None:
        assert any(msg in error.get("msg", "") for error in field_errors), \
            f"Expected error message containing '{msg}' not found in {field_errors}"
