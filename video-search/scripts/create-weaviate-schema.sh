#!/usr/bin/env bash
curl -s -X POST "http://localhost:8080/v1/schema" -H "Content-Type: application/json" --data-binary @ops/weaviate.schema.json
