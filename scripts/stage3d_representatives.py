from pathlib import Path
import csv
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = (
    SparkSession.builder
    .appName("RDD2022_KMeans_RepresentativeImages")
    .getOrCreate()
)

# Resolve the project directory from this script, not from a machine-specific path.
BASE = Path(__file__).resolve().parents[1]

# Set RDD2022_DATASET_ROOT if the dataset is stored somewhere else.
# Expected layout: <dataset root>/India/train/images
DATASET_ROOT = Path(os.environ.get(
    "RDD2022_DATASET_ROOT",
    str(BASE.parent / "RDD2022_India")
))
IMAGE_DIR = DATASET_ROOT / "India" / "train" / "images"
OUTPUT = BASE / "results/final_analysis"
PARQUET = (
    "hdfs://localhost:9000/rdd2022/india/"
    "results/kmeans_validation/clustered_images"
)

try:
    df = spark.read.parquet(PARQUET)

    print("Clustered image rows:", df.count())
    print("Cluster IDs:", sorted(
        r["prediction"]
        for r in df.select("prediction").distinct().collect()
    ))

    rows = []

    cluster_ids = [
        r["prediction"]
        for r in df.select("prediction").distinct().collect()
    ]

    for cluster_id in sorted(cluster_ids):
        reps = (
            df.filter(
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
            .collect()
        )

        for r in reps:
            filename = r["filename"]
            image_path = IMAGE_DIR / filename

            rows.append({
                "cluster_id": r["prediction"],
                "filename": filename,
                "total_instances": r["total_instances"],
                "severity_score": r["severity_score"],
                "count_D00": r["count_D00"],
                "count_D10": r["count_D10"],
                "count_D20": r["count_D20"],
                "count_D40": r["count_D40"],
                "image_exists": image_path.is_file(),
                "image_path": str(image_path)
            })

    csv_path = OUTPUT / "representative_images_k15.csv"
    OUTPUT.mkdir(parents=True, exist_ok=True)

    columns = [
        "cluster_id", "filename", "total_instances",
        "severity_score", "count_D00", "count_D10",
        "count_D20", "count_D40", "image_exists",
        "image_path"
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print("\nRepresentative-image CSV:", csv_path)
    print("Candidates selected:", len(rows))
    print("Existing images:", sum(r["image_exists"] for r in rows))
    print("Missing images:", sum(not r["image_exists"] for r in rows))

    for r in rows:
        print(
            f'Cluster {r["cluster_id"]}: {r["filename"]} | '
            f'severity={r["severity_score"]:.4f} | '
            f'exists={r["image_exists"]}'
        )

finally:
    spark.stop()
