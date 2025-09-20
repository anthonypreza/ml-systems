"""Simulate user events and send them to a Kafka topic."""

import json
import random
import time
import uuid
from confluent_kafka import Producer


KAFKA_TOPIC_NAME = "user_events"
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"


def now_ms():
    return int(time.time() * 1000)


if __name__ == "__main__":
    print("Starting event generator...")
    print(f"Kafka broker: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"Target topic: {KAFKA_TOPIC_NAME}")

    p = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})
    users = [f"user_{i}" for i in range(1, 2001)]
    content = [f"content_{i}" for i in range(1, 501)]
    event_types = ["view_start", "view_heartbeat", "click", "like", "share"]

    print(f"Initialized {len(users)} users and {len(content)} content items")
    print("Event generation started. Press Ctrl+C to stop.")
    print("Generating events with 2ms intervals...")

    try:
        while True:
            # Simulate bursty traffic by randomly picking hot content
            hot = random.random() < 0.05
            cid = random.choice(content[:20]) if hot else random.choice(content)
            uid = random.choice(users)

            event = {
                "event_id": str(uuid.uuid4()),
                "user_id": uid,
                "content_id": cid,
                "event_type": random.choices(
                    event_types, weights=[0.3, 0.3, 0.3, 0.09, 0.01]
                )[0],
                "watch_secs": random.choice([0, 0, 0, 5, 10, 15, 30]),
                "ts": now_ms(),
            }

            p.produce(
                KAFKA_TOPIC_NAME, key=uid.encode(), value=json.dumps(event).encode()
            )
            p.poll(0)
            time.sleep(0.002)
    except KeyboardInterrupt:
        pass
    finally:
        p.flush()
        print(f"\nFlushed remaining messages to topic '{KAFKA_TOPIC_NAME}'")
        print("Exiting.")
