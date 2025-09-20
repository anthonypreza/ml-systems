# Feature Store Lab

An end-to-end lab demonstrating the usage of Kafka, Flink, and Redis for an online/offline feature store pipeline.

# Architecture

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
