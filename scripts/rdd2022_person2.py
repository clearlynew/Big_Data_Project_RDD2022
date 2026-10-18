from pyspark.sql import SparkSession


# ============================================================
# RDD2022 INDIA - PERSON 2
# MLlib Analysis
# ============================================================

spark = (
    SparkSession.builder
    .appName("RDD2022_India_Person2_MLlib")
    .getOrCreate()
)


# ============================================================
# Person 1 Parquet output
# ============================================================

PARQUET_PATH = (
    "hdfs://localhost:9000/"
    "rdd2022/india/features/india_features.parquet"
)


# ============================================================
# Read Parquet
# ============================================================

print("=" * 70)
print("PERSON 2 - READING PERSON 1 OUTPUT")
print("=" * 70)

df = spark.read.parquet(PARQUET_PATH)


# ============================================================
# Basic validation
# ============================================================

print("\nNumber of rows:")
print(df.count())


print("\nNumber of columns:")
print(len(df.columns))


print("\nColumn names:")
print(df.columns)


print("\nSchema:")
df.printSchema()


print("\nFirst 5 rows:")
df.show(5, truncate=False)


# ============================================================
# Verify damage-bearing images
# ============================================================

print("\nDamage-bearing images:")

df.filter(
    df["total_instances"] > 0
).count()


print(
    df.filter(
        df["total_instances"] > 0
    ).count()
)


# ============================================================
# Verify multi-class images
# ============================================================

multi_class_count = (
    df.filter(
        df["damage_class_count"] >= 2
    ).count()
)

print(
    "\nMulti-class images:",
    multi_class_count
)


# ============================================================
# Stop Spark
# ============================================================

spark.stop()
