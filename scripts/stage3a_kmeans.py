from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, round as spark_round
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

spark = (
    SparkSession.builder
    .appName("RDD2022_Stage3A_KMeans_Validation")
    .getOrCreate()
)

FEATURE_PATH = (
    "hdfs://localhost:9000/"
    "rdd2022/india/features/india_features.parquet"
)

# Separate output path: preserve the old results.
OUTPUT_BASE = (
    "hdfs://localhost:9000/"
    "rdd2022/india/results/kmeans_validation"
)

FEATURE_COLS = [
    "count_D00",
    "count_D10",
    "count_D20",
    "count_D40",
    "area_ratio_D00",
    "area_ratio_D10",
    "area_ratio_D20",
    "area_ratio_D40",
    "damage_class_count",
]

try:
    print("\n1. Reading feature dataset")
    df = spark.read.parquet(FEATURE_PATH)

    required = FEATURE_COLS + [
        "filename", "total_instances", "severity_score"
    ]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {df.columns}"
        )

    print("Original rows:", df.count())

    # Keep only images with at least one annotated damage instance.
    df = df.filter(col("total_instances") > 0)

    # Exclude rows with missing feature values.
    df = df.dropna(subset=FEATURE_COLS + [
        "filename", "severity_score"
    ])

    print("Damaged images used:", df.count())
    print("Features:", FEATURE_COLS)

    if df.count() < 13:
        raise ValueError("Not enough rows to evaluate K=2 through K=12.")

    print("\n2. Assemble and standardise nine features")
    assembler = VectorAssembler(
        inputCols=FEATURE_COLS,
        outputCol="raw_features",
        handleInvalid="error"
    )
    assembled = assembler.transform(df)

    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withMean=True,
        withStd=True
    )
    scaler_model = scaler.fit(assembled)
    prepared = scaler_model.transform(assembled).cache()
    prepared.count()

    print("\n3. Evaluate K=2 through K=12")
    evaluator = ClusteringEvaluator(
        featuresCol="features",
        predictionCol="prediction",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean"
    )

    metrics = []
    models = {}

    for k in range(2, 16):
        model = KMeans(
            k=k,
            seed=42,
            featuresCol="features",
            predictionCol="prediction",
            maxIter=50
        ).fit(prepared)

        clustered = model.transform(prepared).cache()
        clustered.count()

        silhouette = evaluator.evaluate(clustered)
        wcss = model.summary.trainingCost

        metrics.append((k, float(silhouette), float(wcss)))
        models[k] = model

        print(
            f"K={k:2d} | Silhouette={silhouette:.6f} "
            f"| WCSS={wcss:.6f}"
        )
        clustered.unpersist()

    metrics_df = spark.createDataFrame(
        metrics, ["k", "silhouette", "wcss"]
    )
    metrics_df.orderBy("k").coalesce(1).write.mode(
        "overwrite"
    ).option("header", True).csv(OUTPUT_BASE + "/metrics")

    # Select the highest measured silhouette among tested K values.
    best_k, best_silhouette, best_wcss = max(
        metrics, key=lambda item: item[1]
    )

    print(
        f"\nSelected K={best_k}; "
        f"silhouette={best_silhouette:.6f}; "
        f"WCSS={best_wcss:.6f}"
    )

    print("\n4. Fit selected model and save clustered rows")
    final_model = models[best_k]
    clustered = final_model.transform(prepared).cache()
    clustered.count()

    clustered.select(
        "filename", "total_instances", "severity_score",
        *FEATURE_COLS, "prediction"
    ).write.mode("overwrite").parquet(
        OUTPUT_BASE + "/clustered_images"
    )

    print("\n5. Generate correctly labelled cluster summaries")
    cluster_stats = clustered.groupBy("prediction").agg(
        count("*").alias("image_count"),
        spark_round(avg("severity_score"), 4).alias(
            "avg_severity_score"
        ),
        spark_round(avg("damage_class_count"), 4).alias(
            "avg_damage_class_count"
        ),
        spark_round(avg("count_D00"), 4).alias("avg_count_D00"),
        spark_round(avg("count_D10"), 4).alias("avg_count_D10"),
        spark_round(avg("count_D20"), 4).alias("avg_count_D20"),
        spark_round(avg("count_D40"), 4).alias("avg_count_D40"),
    ).orderBy("prediction")

    cluster_stats.show(50, truncate=False)
    cluster_stats.coalesce(1).write.mode("overwrite").option(
        "header", True
    ).csv(OUTPUT_BASE + "/cluster_summary")

    print("\n6. Print two high-severity damaged-image candidates per cluster")
    for cluster_id in range(best_k):
        reps = (
            clustered
            .filter(
                (col("prediction") == cluster_id)
                & (col("total_instances") > 0)
            )
            .orderBy(col("severity_score").desc())
            .select(
                "filename", "prediction", "total_instances",
                "severity_score", "count_D00", "count_D10",
                "count_D20", "count_D40"
            )
            .limit(2)
        )

        print(f"\nCluster {cluster_id}:")
        reps.show(truncate=False)

    print("\nNew validation outputs saved under:")
    print(OUTPUT_BASE)
    print("Existing K-Means results have not been overwritten.")

finally:
    spark.stop()
