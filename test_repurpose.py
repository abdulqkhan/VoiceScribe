#!/usr/bin/env python3
"""
Test script for the repurpose endpoint
"""

import requests
import os
import sys
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:5000"  # Update this to your server URL
API_KEY = os.getenv("API_KEY", "your_api_key_here")  # Update with your actual API key

def test_repurpose_endpoint(file_path, email, repurpose_message):
    """
    Test the /repurpose endpoint
    """
    endpoint = f"{BASE_URL}/repurpose"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return False
    
    # Prepare the request
    headers = {
        "X-API-Key": API_KEY
    }
    
    data = {
        "email": email,
        "repurpose_message": repurpose_message
    }
    
    files = {
        "file": open(file_path, "rb")
    }
    
    print(f"Testing repurpose endpoint...")
    print(f"  File: {file_path}")
    print(f"  Email: {email}")
    print(f"  Repurpose message: {repurpose_message}")
    print(f"  Endpoint: {endpoint}")
    
    try:
        # Send the request
        response = requests.post(endpoint, headers=headers, data=data, files=files)
        
        # Close the file
        files["file"].close()
        
        # Check response
        print(f"\nResponse Status: {response.status_code}")
        print(f"Response Body: {response.json()}")
        
        if response.status_code == 202:
            job_data = response.json()
            job_id = job_data.get("job_id")
            print(f"\nSuccess! Job ID: {job_id}")
            print(f"You can check the job status at: {BASE_URL}/job_status/{job_id}")
            return True
        else:
            print(f"\nError: {response.json()}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"\nRequest failed: {e}")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return False

def test_job_status(job_id):
    """
    Check the status of a job
    """
    endpoint = f"{BASE_URL}/job_status/{job_id}"
    
    print(f"\nChecking job status...")
    print(f"  Job ID: {job_id}")
    print(f"  Endpoint: {endpoint}")
    
    try:
        response = requests.get(endpoint)
        
        print(f"\nResponse Status: {response.status_code}")
        print(f"Response Body: {response.json()}")
        
        return response.status_code == 200
        
    except requests.exceptions.RequestException as e:
        print(f"\nRequest failed: {e}")
        return False

def test_regular_upload(file_path, email):
    """
    Test the regular /upload_and_process endpoint for comparison
    """
    endpoint = f"{BASE_URL}/upload_and_process"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return False
    
    # Prepare the request
    headers = {
        "X-API-Key": API_KEY
    }
    
    data = {
        "email": email
    }
    
    files = {
        "file": open(file_path, "rb")
    }
    
    print(f"\nTesting regular upload_and_process endpoint...")
    print(f"  File: {file_path}")
    print(f"  Email: {email}")
    print(f"  Endpoint: {endpoint}")
    
    try:
        # Send the request
        response = requests.post(endpoint, headers=headers, data=data, files=files)
        
        # Close the file
        files["file"].close()
        
        # Check response
        print(f"\nResponse Status: {response.status_code}")
        print(f"Response Body: {response.json()}")
        
        if response.status_code == 202:
            job_data = response.json()
            job_id = job_data.get("job_id")
            print(f"\nSuccess! Job ID: {job_id}")
            return True
        else:
            print(f"\nError: {response.json()}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"\nRequest failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("REPURPOSE ENDPOINT TEST SCRIPT")
    print("=" * 60)
    
    # Test parameters - update these as needed
    test_file = "test_audio.mp3"  # Update with your test file
    test_email = "test@example.com"
    test_repurpose_message = "Convert this content to be relevant for software developers learning Python"
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    if len(sys.argv) > 2:
        test_email = sys.argv[2]
    if len(sys.argv) > 3:
        test_repurpose_message = " ".join(sys.argv[3:])
    
    print("\nTest Configuration:")
    print(f"  Base URL: {BASE_URL}")
    print(f"  API Key: {'*' * (len(API_KEY) - 4) + API_KEY[-4:] if len(API_KEY) > 4 else '****'}")
    
    # Run tests
    print("\n" + "=" * 60)
    print("TEST 1: Repurpose Endpoint")
    print("=" * 60)
    
    if test_repurpose_endpoint(test_file, test_email, test_repurpose_message):
        print("\n✓ Repurpose endpoint test passed!")
    else:
        print("\n✗ Repurpose endpoint test failed!")
    
    print("\n" + "=" * 60)
    print("TEST 2: Regular Upload Endpoint (for comparison)")
    print("=" * 60)
    
    if test_regular_upload(test_file, test_email):
        print("\n✓ Regular upload endpoint test passed!")
    else:
        print("\n✗ Regular upload endpoint test failed!")
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
    print("\nNote: Check your webhook endpoint (n8n) to verify the payload includes:")
    print("  - is_repurpose: true/false")
    print("  - repurpose_message: (for repurpose jobs)")
    print("  - email: (for both job types)")