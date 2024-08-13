#!/bin/bash

# Start a new detached screen session and run the Prometheus script
screen -dmS prometheus-session bash -c "prometheus --config.file=prometheus.yml"

# Optional: Display a message
echo "Prometheus has been started in a detached screen session named 'prometheus-session."