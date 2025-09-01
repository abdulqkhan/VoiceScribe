#!/bin/bash

# Test script for the repurpose endpoint using curl
# Update these variables according to your setup

BASE_URL="http://localhost:5000"
API_KEY="your_api_key_here"
TEST_FILE="test_audio.mp3"
EMAIL="test@example.com"
REPURPOSE_MESSAGE="Convert this content to be relevant for software developers"

echo "=========================================="
echo "TESTING REPURPOSE ENDPOINT"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Base URL: $BASE_URL"
echo "  API Key: ${API_KEY:0:4}****"
echo "  Test File: $TEST_FILE"
echo "  Email: $EMAIL"
echo "  Repurpose Message: $REPURPOSE_MESSAGE"
echo ""

# Test 1: Repurpose endpoint
echo "Test 1: Repurpose Endpoint"
echo "------------------------------------------"
echo "Sending request to /repurpose..."
echo ""

curl -X POST "$BASE_URL/repurpose" \
  -H "X-API-Key: $API_KEY" \
  -F "file=@$TEST_FILE" \
  -F "email=$EMAIL" \
  -F "repurpose_message=$REPURPOSE_MESSAGE" \
  -w "\n\nHTTP Status: %{http_code}\n"

echo ""
echo ""

# Test 2: Regular upload_and_process endpoint for comparison
echo "Test 2: Regular Upload Endpoint (for comparison)"
echo "------------------------------------------"
echo "Sending request to /upload_and_process..."
echo ""

curl -X POST "$BASE_URL/upload_and_process" \
  -H "X-API-Key: $API_KEY" \
  -F "file=@$TEST_FILE" \
  -F "email=$EMAIL" \
  -w "\n\nHTTP Status: %{http_code}\n"

echo ""
echo "=========================================="
echo "TESTING COMPLETE"
echo "=========================================="
echo ""
echo "Check your webhook endpoint (n8n) to verify:"
echo "  - Repurpose job has: is_repurpose=true, repurpose_message, email"
echo "  - Regular job has: is_repurpose=false, email"