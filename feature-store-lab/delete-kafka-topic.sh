set -ex

docker exec -it $(docker ps -qf "name=kafka") kafka-topics --bootstrap-server localhost:9092 --delete --topic user_events
