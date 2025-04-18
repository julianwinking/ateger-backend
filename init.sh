#!/bin/bash
# Initialize backend directories and dependencies

# Make sure the reports directory exists
mkdir -p reports

# Install the reportlab package if not already installed
pip install reportlab==4.0.5

echo "Backend initialization complete!"