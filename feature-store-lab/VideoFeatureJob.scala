import org.apache.flink.api.common.eventtime.{
  SerializableTimestampAssigner,
  WatermarkStrategy
}
import org.apache.flink.api.common.state.{
  ListState,
  ListStateDescriptor,
  ValueState,
  ValueStateDescriptor
}
import org.apache.flink.api.common.time.Time
import org.apache.flink.api.scala._
import org.apache.flink.configuration.Configuration
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator
import org.apache.flink.streaming.api.scala.StreamExecutionEnvironment
import org.apache.flink.streaming.api.functions.ProcessFunction
import org.apache.flink.streaming.api.scala.DataStream
import org.apache.flink.streaming.api.scala.function.ProcessWindowFunction
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows
import org.apache.flink.streaming.api.windowing.windows.TimeWindow
import org.apache.flink.util.Collector
import org.apache.flink.table.api._
import org.apache.flink.table.api.bridge.scala.StreamTableEnvironment
import org.apache.flink.connector.kafka.source.KafkaSource
import org.apache.flink.api.common.serialization.SimpleStringSchema
import org.apache.flink.connector.kafka.source.reader.deserializer.KafkaRecordDeserializationSchema
import org.apache.flink.connector.kafka.sink.KafkaRecordSerializationSchema
import org.apache.flink.formats.json.JsonRowDataSerializationSchema
import redis.clients.jedis.JedisPooled
import scala.collection.JavaConverters._
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.databind.ObjectMapper
import org.apache.flink.streaming.api.functions.sink.RichSinkFunction

case class UserEvent(
    event_id: String,
    user_id: String,
    content_id: String,
    event_type: String,
    watch_secs: Int,
    ts: Long
)

case class OfflineBucket(
    user_id: String,
    window_start: Long,
    window_end: Long,
    clicks_5min: Long,
    likes_5min: Long,
    shares_5min: Long,
    watch_secs_5min: Long
)

case class OnlineFeature(
    user_id: String,
    asOf: Long,
    clicks_last_5min: Long,
    likes_last_5min: Long,
    shares_last_5min: Long,
    watch_secs_last_5min: Long
)

object VideoFeatureJob {
  val mapper = new ObjectMapper()

  def main(args: Array[String]): Unit = {
    println("=== VideoFeatureJob Starting ===")
    println(s"Flink job started at: ${java.time.Instant.now()}")

    val env = StreamExecutionEnvironment.getExecutionEnvironment
    env.getConfig.setAutoWatermarkInterval(1000)
    env.enableCheckpointing(60000)
    env.getCheckpointConfig.setMinPauseBetweenCheckpoints(30000)
    env.getCheckpointConfig.setCheckpointTimeout(120000)
    env.getCheckpointConfig.setTolerableCheckpointFailureNumber(3)
    println(
      s"StreamExecutionEnvironment created with parallelism: ${env.getParallelism}"
    )

    // Table/SQL setup
    val tableEnv = StreamTableEnvironment.create(env)
    tableEnv.getConfig.set("table.exec.source.idle-timeout", "10 s")

    val bootstrap =
      sys.env.getOrElse("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    println(s"Kafka bootstrap servers: $bootstrap")
    println(s"Redis host: ${sys.env.getOrElse("REDIS_HOST", "localhost")}")

    // Source (Kafka JSON to table)
    println("Creating Kafka source table 'user_events'...")
    tableEnv.executeSql(
      s"""
        |CREATE TABLE user_events (
        |  event_id STRING,
        |  user_id STRING,
        |  content_id STRING,
        |  event_type STRING,
        |  watch_secs INT,
        |  ts BIGINT,
        |  `event_time` AS TO_TIMESTAMP_LTZ(ts, 3),
        |  WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
        |) WITH (
        |  'connector' = 'kafka',
        |  'topic' = 'user_events',
        |  'properties.bootstrap.servers' = '$bootstrap',
        |  'scan.startup.mode' = 'earliest-offset',
        |  'format' = 'json',
        |  'json.ignore-parse-errors' = 'true'
        |)
        |""".stripMargin
    )
    println("✓ Kafka source table created")

    // Offline 5-min tumbling window aggregation
    println("Creating filesystem sink table 'offline_sink'...")
    tableEnv.executeSql(
      s"""
        |CREATE TABLE offline_sink (
        |  user_id STRING,
        |  window_start TIMESTAMP_LTZ(3),
        |  window_end   TIMESTAMP_LTZ(3),
        |  clicks_5min BIGINT,
        |  likes_5min BIGINT,
        |  shares_5min BIGINT,
        |  watch_secs_5min BIGINT,
        |  dt STRING,
        |  hr STRING
        |)
        |PARTITIONED BY (dt, hr) 
        |WITH (
        |  'connector' = 'filesystem',
        |  'path' = 'file:///tmp/offline/5m/',
        |  'format' = 'parquet'
        |)
        |""".stripMargin
    )
    println("✓ Filesystem sink table created")

    println("Starting offline aggregation job (5-minute tumbling windows)...")
    tableEnv.executeSql(
      """
        |INSERT INTO offline_sink
        |SELECT
        |  user_id,
        |  TUMBLE_START(event_time, INTERVAL '5' MINUTE) AS window_start,
        |  TUMBLE_END  (event_time, INTERVAL '5' MINUTE) AS window_end,
        |  SUM(CASE WHEN event_type = 'click' THEN 1 ELSE 0 END) AS clicks_5min,
        |  SUM(CASE WHEN event_type = 'like' THEN 1 ELSE 0 END) AS likes_5min,
        |  SUM(CASE WHEN event_type = 'share' THEN 1 ELSE 0 END) AS shares_5min,
        |  SUM(COALESCE(watch_secs, 0)) AS watch_secs_5min,
        |  DATE_FORMAT(TUMBLE_START(event_time, INTERVAL '5' MINUTE), 'yyyy-MM-dd') AS dt,
        |  DATE_FORMAT(TUMBLE_START(event_time, INTERVAL '5' MINUTE), 'HH')         AS hr
        |FROM user_events
        |GROUP BY
        |  user_id,
        |  TUMBLE(event_time, INTERVAL '5' MINUTE)
        |""".stripMargin
    )
    println("✓ Offline aggregation job submitted")

    // Online features: rolling last-5min aggregation to Redis
    println("Setting up online feature processing stream...")
    val events: DataStream[UserEvent] =
      tableEnv
        .toDataStream(tableEnv.from("user_events"))
        .map { row =>
          UserEvent(
            row.getField("event_id").asInstanceOf[String],
            row.getField("user_id").asInstanceOf[String],
            row.getField("content_id").asInstanceOf[String],
            row.getField("event_type").asInstanceOf[String],
            Option(row.getField("watch_secs").asInstanceOf[Int]).getOrElse(0),
            row.getField("ts").asInstanceOf[Long]
          )
        }
        .assignTimestampsAndWatermarks(
          WatermarkStrategy
            .forBoundedOutOfOrderness[UserEvent](
              java.time.Duration.ofSeconds(30)
            )
            .withTimestampAssigner(
              new SerializableTimestampAssigner[UserEvent] {
                override def extractTimestamp(
                    e: UserEvent,
                    recordTimestamp: Long
                ): Long = e.ts
              }
            )
        )

    // Key by user and maintain a rolling deque of events within the last 5 minutes
    println("Creating rolling 5-minute feature aggregation per user...")
    val rolling: DataStream[OnlineFeature] =
      events
        .keyBy(_.user_id)
        .process(new Last5mRollingFeatures)

    println("Adding Redis sink for online features...")
    rolling.addSink(
      new RedisFeatureSink(sys.env.getOrElse("REDIS_HOST", "localhost"), 6379)
    )

    println("=== All components configured, starting job execution ===")
    env.execute("Video Feature Job: offline tumbling + online rolling")
  }

  // Rolling last 5min features per user
  class Last5mRollingFeatures
      extends ProcessFunction[UserEvent, OnlineFeature] {
    private var clicksState: ListState[Long] = _
    private var likesState: ListState[Long] = _
    private var sharesState: ListState[Long] = _
    private var watchState: ListState[(Long, Int)] = _
    private var eventCounter = 0L

    override def open(parameters: Configuration): Unit = {
      println(
        s"[Task ${getRuntimeContext.getTaskNameWithSubtasks}] Initializing rolling feature processor..."
      )
      val clicksDesc = new ListStateDescriptor[Long]("click_ts", classOf[Long])
      val likesDesc = new ListStateDescriptor[Long]("like_ts", classOf[Long])
      val sharesDesc = new ListStateDescriptor[Long]("share_ts", classOf[Long])
      val watchDesc = new ListStateDescriptor[(Long, Int)](
        "watch_ts_secs",
        org.apache.flink.api.scala
          .createTypeInformation[(Long, Int)]
          .createSerializer(getRuntimeContext.getExecutionConfig)
      )

      clicksState = getRuntimeContext.getListState(clicksDesc)
      likesState = getRuntimeContext.getListState(likesDesc)
      sharesState = getRuntimeContext.getListState(sharesDesc)
      watchState = getRuntimeContext.getListState(watchDesc)
      println(
        s"✓ Rolling feature processor initialized for task ${getRuntimeContext.getTaskNameWithSubtasks}"
      )
    }

    override def processElement(
        e: UserEvent,
        ctx: ProcessFunction[UserEvent, OnlineFeature]#Context,
        out: Collector[OnlineFeature]
    ): Unit = {
      eventCounter += 1

      if (eventCounter % 1000 == 0) {
        println(
          s"[${getRuntimeContext.getTaskNameWithSubtasks}] Processed $eventCounter events"
        )
      }

      val now = e.ts
      val cutoff = now - 5 * 60 * 1000L

      // Update clicks state
      if (e.event_type == "click") {
        val lst = clicksState.get().iterator().asScala.toBuffer
        lst += e.ts

        // Remove old clicks
        val trimmed = lst.filter(_ >= cutoff)
        clicksState.update(trimmed.asJava)
      }

      // Update likes state
      if (e.event_type == "like") {
        val lst = likesState.get().iterator().asScala.toBuffer
        lst += e.ts

        // Remove old likes
        val trimmed = lst.filter(_ >= cutoff)
        likesState.update(trimmed.asJava)
      }

      // Update shares state
      if (e.event_type == "share") {
        val lst = sharesState.get().iterator().asScala.toBuffer
        lst += e.ts
        // Remove old shares
        val trimmed = lst.filter(_ >= cutoff)
        sharesState.update(trimmed.asJava)
      }

      // Update watch_secs state
      if (e.watch_secs > 0) {
        val lst = watchState.get().iterator().asScala.toBuffer
        lst += ((e.ts, e.watch_secs))

        // Remove old watch_secs
        val trimmed = lst.filter(_._1 >= cutoff)
        watchState.update(trimmed.asJava)
      }
      // Compute current features
      val clicks = clicksState.get().iterator().asScala.count(_ >= cutoff)
      val likes = likesState.get().iterator().asScala.count(_ >= cutoff)
      val shares = sharesState.get().iterator().asScala.count(_ >= cutoff)
      val watch_secs =
        watchState.get().iterator().asScala.filter(_._1 >= cutoff).map(_._2).sum

      out.collect(
        OnlineFeature(e.user_id, now, clicks, likes, shares, watch_secs)
      )
    }
  }

  // Redis sink for online features
  class RedisFeatureSink(host: String, port: Int)
      extends RichSinkFunction[
        OnlineFeature
      ] {
    @transient private var jedis: JedisPooled = _
    private var featureCounter = 0L

    override def open(parameters: Configuration): Unit = {
      println(
        s"[${getRuntimeContext.getTaskNameWithSubtasks}] Opening Redis connection to $host:$port..."
      )
      try {
        jedis = new JedisPooled(host, port)
        println(
          s"✓ Redis connection established for task ${getRuntimeContext.getTaskNameWithSubtasks}"
        )
      } catch {
        case e: Exception =>
          println(s"✗ Failed to connect to Redis: ${e.getMessage}")
          throw e
      }
    }

    override def invoke(
        v: OnlineFeature,
        ctx: org.apache.flink.streaming.api.functions.sink.SinkFunction.Context
    ): Unit = {
      featureCounter += 1

      if (featureCounter % 500 == 0) {
        println(
          s"[${getRuntimeContext.getTaskNameWithSubtasks}] Wrote $featureCounter features to Redis"
        )
      }

      val key = s"feat:user:${v.user_id}"
      val payload =
        s"""{"asOf":${v.asOf},"clicks_last_5min":${v.clicks_last_5min},"shares_last_5min":${v.shares_last_5min},"likes_last_5min":${v.likes_last_5min},"watch_secs_last_5min":${v.watch_secs_last_5min}}"""

      // Single packed value, set short TTL to trim cold users
      jedis.setex(key, 15 * 60, payload)
    }

    override def close(): Unit = {
      println(
        s"[${getRuntimeContext.getTaskNameWithSubtasks}] Closing Redis connection... (wrote $featureCounter features total)"
      )
      if (jedis != null) {
        jedis.close()
      }
    }
  }
}
