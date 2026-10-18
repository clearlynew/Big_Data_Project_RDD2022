from pyspark.sql import SparkSession
from pyspark.sql.functions import col, array, when, lit
from pyspark.ml.fpm import FPGrowth

spark = (
    SparkSession.builder
    .appName("RDD2022_FPGrowth_AllDamagedImages")
    .getOrCreate()
)

FEATURE_PATH = (
    "hdfs://localhost:9000/rdd2022/india/"
    "features/india_features.parquet"
)

OUTPUT_BASE = (
    "hdfs://localhost:9000/rdd2022/india/"
    "results/fpgrowth_all_damaged"
)

try:
    print("=" * 70)
    print("RDD2022 FPGrowth — All Damage-Bearing Images")
    print("=" * 70)

    df = spark.read.parquet(FEATURE_PATH)

    # Include every image with at least one annotated damage instance.
    damaged = df.filter(col("total_instances") > 0)

    print("All feature rows:", df.count())
    print("Damage-bearing images:", damaged.count())

    # One transaction per image; include each damage class present.
    transactions = damaged.select(
        array(
            when(col("count_D00") > 0, lit("D00")),
            when(col("count_D10") > 0, lit("D10")),
            when(col("count_D20") > 0, lit("D20")),
            when(col("count_D40") > 0, lit("D40"))
        ).alias("items")
    )

    # Remove null entries from each transaction.
    from pyspark.sql.functions import expr
    transactions = transactions.select(
        expr("filter(items, x -> x is not null)").alias("items")
    )

    print("\nTransaction count:", transactions.count())
    transactions.show(10, truncate=False)

    # Preserve the previous experiment's thresholds.
    min_support = 0.05
    min_confidence = 0.2

    fp = FPGrowth(
        itemsCol="items",
        minSupport=min_support,
        minConfidence=min_confidence
    )

    model = fp.fit(transactions)

    itemsets = model.freqItemsets
    rules = model.associationRules

    print("\nFrequent itemsets:")
    itemsets.orderBy(col("freq").desc()).show(100, truncate=False)

    print("\nAssociation rules:")
    rules.orderBy(col("confidence").desc()).show(100, truncate=False)

    # New output location: existing results are not overwritten.
    itemsets_path = OUTPUT_BASE + "/frequent_itemsets"
    rules_path = OUTPUT_BASE + "/association_rules"

    itemsets.write.mode("overwrite").parquet(itemsets_path)
    rules.write.mode("overwrite").parquet(rules_path)

    # Convert array columns to strings before CSV export.
    from pyspark.sql.functions import concat_ws

    rules_csv = rules.select(
        concat_ws(", ", col("antecedent")).alias("antecedent"),
        concat_ws(", ", col("consequent")).alias("consequent"),
        "support",
        "confidence",
        "lift"
    )

    rules_csv.coalesce(1).write.mode("overwrite").option(
        "header", True
    ).csv(OUTPUT_BASE + "/association_rules_csv")

    print("\nSaved frequent itemsets to:", itemsets_path)
    print("Saved association rules to:", rules_path)
    print("Saved rule CSV directory to:", OUTPUT_BASE + "/association_rules_csv")

    print("\nRule count:", rules.count())
    print("Frequent itemset count:", itemsets.count())

finally:
    spark.stop()
