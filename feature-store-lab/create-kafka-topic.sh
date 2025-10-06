set -ex

docker exec -it $(docker ps -qf "name=kafka") \
kafka-topics --create --topic user_events --bootstrap-server localhost:9092 --partitions 6 --replication-factor 1
