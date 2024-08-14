#!/bin/bash
 
# Start a new detached screen session and run the Prometheus script
screen -dmS grafana-session bash -c "/usr/sbin/grafana-server --config=/tmp/ray/session_latest/metrics/grafana/grafana.ini --homepath=/usr/share/grafana web"

# Optional: Display a message
echo "grafana-server has been started in a detached screen session named 'grafana-session."