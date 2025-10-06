# Feature Store Lab

An end-to-end lab that produces streaming video engagement features with Kafka, Flink, and Redis. It demonstrates how to build both offline parquet feature files and low-latency online features from the same event stream.

## Architecture
```
[Event Generator] -> Kafka topic: user_events
                          |
                     Flink job
            +-----------+--------------+
            |                          |
  (A) 5m tumbling -> offline files     |   (B) Rolling "last 5m" features
       (Table/SQL)                     |       (keyed state + TTL)
                                      Redis  (online feature store)
```

## Prerequisites
- Apache Flink 1.18.x installation, with `FLINK_HOME` pointing to the root of the distribution (for example `/opt/flink`). The `flink` CLI then lives at `$FLINK_HOME/bin/flink`.
- Copy the aligned Parquet 1.13.1 jars into `$FLINK_HOME/lib/` so the runtime can create parquet files:
  ```
  parquet-column-1.13.1.jar
  parquet-common-1.13.1.jar
  parquet-encoding-1.13.1.jar
  parquet-hadoop-bundle-1.13.1.jar
  parquet-format-1.13.1.jar
  parquet-format-structures-1.13.1.jar
  ```
- Java 11+ and [sbt 1.11+](https://www.scala-sbt.org/) for building the job.
- Docker & Docker Compose (v2 syntax) to run Kafka, ZooKeeper, and Redis.
- Python 3 with `pip install confluent-kafka` for the event generator.

## Start the streaming infrastructure
- Launch Kafka, ZooKeeper, and Redis:
  ```bash
  docker compose up -d
  ```
- Create the Kafka topic (idempotent):
  ```bash
  ./create-kafka-topic.sh
  ```
- When you are done experimenting, stop the services with `docker compose down` and optionally delete the topic with `./delete-kafka-topic.sh`.

## Generate synthetic user events
- With the Docker services running, start emitting events into Kafka:
  ```bash
  python event_generator.py
  ```
  This writes JSON payloads to the `user_events` topic at `localhost:9092` until you press `Ctrl+C`.

## Build the Flink job jar
- Assemble a runnable fat jar that bundles the Kafka connector and dependencies:
  ```bash
  sbt assembly
  ```
  The resulting artifact is under `target/scala-2.12/` (for example `feature-store-lab-assembly-0.1.0-SNAPSHOT.jar`).

## Run the Flink job
- Submit the job with the `flink` CLI (either reference it directly or ensure `$FLINK_HOME/bin` is on your `PATH`):
  ```bash
  flink run target/scala-2.12/feature-store-lab-assembly-0.1.0-SNAPSHOT.jar
  ```
- Or send it to an existing cluster (replace host/port as needed):
  ```bash
  flink run -m localhost:8081 target/scala-2.12/feature-store-lab-assembly-0.1.0-SNAPSHOT.jar
  ```
- Override connection settings via environment variables if your endpoints differ:
  - `KAFKA_BOOTSTRAP_SERVERS` (default `localhost:9092`)
  - `REDIS_HOST` (default `localhost`)

## Outputs
- **Offline parquet features** are written partitioned by day/hour to `file:///tmp/offline/5m/` and expect the parquet runtime libraries to be present in the Flink installation.
- **Online features** are continuously updated in Redis under keys per user, holding rolling 5-minute aggregates.

## Useful references
- Kafka quickstart: https://kafka.apache.org/quickstart
- Redis CLI: `redis-cli -h localhost` for inspecting online feature keys.
