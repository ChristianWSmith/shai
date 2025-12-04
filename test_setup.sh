#!/bin/bash

# --- Test Setup for shai Agent (Text Task) ---

# 1. Create a dedicated test directory
TEST_DIR="shai_text_test"
echo "Creating test directory: $TEST_DIR"
# Remove existing test directory if it exists to ensure a clean slate
rm -rf "$TEST_DIR" 
mkdir -p "$TEST_DIR/logs"
cd "$TEST_DIR"

# 2. Create dummy LOG files with content
echo "Creating dummy LOG files with content..."

# Log file 1: 2 errors
cat <<EOF > logs/server_20251203.log
INFO: Startup successful
WARN: Disk space low
ERROR: File not found in path /etc/config
INFO: User logged in
ERROR: Timeout exceeded for database query
EOF

# Log file 2: 1 error
cat <<EOF > logs/api_service.log
INFO: API request processed
ERROR: Authentication failed for token X
INFO: Response sent
EOF

# 3. Create a non-target file
touch logs/notes.md

# 4. Compile the shai agent (assuming shai.go is in the parent directory)
echo "Compiling shai.go..."
go build ../shai.go

echo "Setup complete. You are now inside the '$TEST_DIR' directory."
echo "---"
echo "Next step: Run the shai agent interactively using the following command:"
echo ""
echo "./shai \"Find the total number of times the word 'ERROR' appears across all '.log' files in the 'logs' subdirectory and save the final count to a new file named 'error_count.txt' in the current directory.\""
echo ""
echo "You will need to approve each command the agent suggests with 'y'."