#!/bin/bash
# Wait for Neo4j to be ready

set -e

host="$1"
shift
cmd="$@"

until curl -s http://$host:7474 > /dev/null; do
  >&2 echo "Neo4j is unavailable - sleeping"
  sleep 2
done

>&2 echo "Neo4j is up - executing command"
exec $cmd
