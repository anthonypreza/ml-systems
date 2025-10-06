// ---- imports for sbt-assembly and helpers
import sbt._
import sbt.Keys._
import sbtassembly.AssemblyPlugin
import sbtassembly.AssemblyPlugin.autoImport._
import sbtassembly.MergeStrategy
import sbt.io.Path

// ---- versions
ThisBuild / scalaVersion := "2.12.17"
lazy val flinkVersion = "1.18.1"

// ---- project definition + enable assembly plugin
lazy val root = (project in file("."))
  .enablePlugins(AssemblyPlugin)
  .settings(
    name := "feature-store-lab",
    // Flink core/runtime are 'provided' (Flink supplies them at runtime)
    libraryDependencies ++= Seq(
      "org.apache.flink" %% "flink-scala" % flinkVersion % "provided",
      "org.apache.flink" %% "flink-streaming-scala" % flinkVersion % "provided",
      "org.apache.flink" %% "flink-table-api-scala-bridge" % flinkVersion % "provided",

      // include connectors/format libs in the fat jar
      "org.apache.flink" % "flink-connector-kafka" % "3.2.0-1.18",
      "org.apache.flink" % "flink-json" % flinkVersion,
      "org.apache.flink" % "flink-table-runtime" % flinkVersion,
      "redis.clients" % "jedis" % "5.1.0"
    ),

    // optional: set the main class so 'flink run' doesn’t need -c
    Compile / mainClass := Some("VideoFeatureJob"),

    // assembly merge rules to avoid META-INF conflicts
    assembly / assemblyMergeStrategy := {
      case "module-info.class" => MergeStrategy.discard
      case PathList(
            "META-INF",
            "services",
            "org.apache.flink.table.factories.Factory"
          ) =>
        MergeStrategy.concat
      case PathList("META-INF", xs @ _*) => MergeStrategy.discard
      case _                             => MergeStrategy.first
    }
  )
