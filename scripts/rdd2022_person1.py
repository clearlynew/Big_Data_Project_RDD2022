from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
import xml.etree.ElementTree as ET


# ============================================================
# RDD2022 INDIA - PERSON 1
# Distributed XML ETL using PySpark + HDFS
# ============================================================


# ============================================================
# 1. CREATE SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName("RDD2022_India_Person1_ETL")
    .getOrCreate()
)

sc = spark.sparkContext

print("=" * 70)
print("RDD2022 INDIA - PERSON 1 ETL")
print("=" * 70)
print("Spark version:", spark.version)


# ============================================================
# 2. HDFS PATHS
# ============================================================

XML_PATH = (
    "hdfs://localhost:9000/"
    "rdd2022/india/annotations/*.xml"
)

OUTPUT_PATH = (
    "hdfs://localhost:9000/"
    "rdd2022/india/features/india_features.parquet"
)


# ============================================================
# 3. XML PARSER
# ============================================================

def parse_xml(record):

    file_path, xml_content = record

    try:

        # ----------------------------------------------------
        # Parse XML
        # ----------------------------------------------------

        root = ET.fromstring(xml_content)


        # ----------------------------------------------------
        # Filename
        # ----------------------------------------------------

        filename_element = root.find("filename")

        if filename_element is not None:
            filename = filename_element.text
        else:
            filename = (
                file_path
                .split("/")[-1]
                .replace(".xml", ".jpg")
            )


        # ----------------------------------------------------
        # Image dimensions
        # ----------------------------------------------------

        width_element = root.find("./size/width")
        height_element = root.find("./size/height")

        if width_element is not None:
            width = int(width_element.text)
        else:
            width = 720

        if height_element is not None:
            height = int(height_element.text)
        else:
            height = 720

        image_area = width * height


        # ----------------------------------------------------
        # Initialize damage classes
        # ----------------------------------------------------

        damage_classes = [
            "D00",
            "D10",
            "D20",
            "D40"
        ]

        counts = {
            "D00": 0,
            "D10": 0,
            "D20": 0,
            "D40": 0
        }

        areas = {
            "D00": 0.0,
            "D10": 0.0,
            "D20": 0.0,
            "D40": 0.0
        }


        # ----------------------------------------------------
        # Parse every object
        # ----------------------------------------------------

        for obj in root.findall("object"):

            name_element = obj.find("name")

            if name_element is None:
                continue

            damage_type = name_element.text.strip()

            # Ignore classes outside D00/D10/D20/D40
            if damage_type not in damage_classes:
                continue


            # ------------------------------------------------
            # Get bounding box
            # ------------------------------------------------

            xmin_element = obj.find("./bndbox/xmin")
            ymin_element = obj.find("./bndbox/ymin")
            xmax_element = obj.find("./bndbox/xmax")
            ymax_element = obj.find("./bndbox/ymax")

            if (
                xmin_element is None
                or ymin_element is None
                or xmax_element is None
                or ymax_element is None
            ):
                continue


            xmin = float(xmin_element.text)
            ymin = float(ymin_element.text)
            xmax = float(xmax_element.text)
            ymax = float(ymax_element.text)


            # ------------------------------------------------
            # Calculate bounding-box area
            # ------------------------------------------------

            box_width = max(0.0, xmax - xmin)
            box_height = max(0.0, ymax - ymin)

            box_area = box_width * box_height


            # ------------------------------------------------
            # Update class statistics
            # ------------------------------------------------

            counts[damage_type] += 1
            areas[damage_type] += box_area


        # ----------------------------------------------------
        # Total number of damage instances
        # ----------------------------------------------------

        total_instances = sum(counts.values())


        # ----------------------------------------------------
        # Total damage area
        # ----------------------------------------------------

        total_damage_area = sum(areas.values())


        # ----------------------------------------------------
        # Presence flags
        # ----------------------------------------------------

        has_D00 = 1 if counts["D00"] > 0 else 0
        has_D10 = 1 if counts["D10"] > 0 else 0
        has_D20 = 1 if counts["D20"] > 0 else 0
        has_D40 = 1 if counts["D40"] > 0 else 0


        # ----------------------------------------------------
        # Area ratios
        #
        # Actual image area is used instead of assuming
        # 600 x 600.
        # ----------------------------------------------------

        area_ratio_D00 = areas["D00"] / image_area
        area_ratio_D10 = areas["D10"] / image_area
        area_ratio_D20 = areas["D20"] / image_area
        area_ratio_D40 = areas["D40"] / image_area


        # ----------------------------------------------------
        # Severity score
        #
        # D40 = weight 4
        # D20 = weight 3
        # D00 = weight 2
        # D10 = weight 1
        # ----------------------------------------------------

        weighted_damage = (
            counts["D40"] * 4
            + counts["D20"] * 3
            + counts["D00"] * 2
            + counts["D10"] * 1
        )

        severity_score = (
            weighted_damage
            * (total_damage_area / image_area)
        )


        # ----------------------------------------------------
        # Return one row per image
        # ----------------------------------------------------

        return {

            "filename": filename,

            "width": width,

            "height": height,

            "count_D00": counts["D00"],

            "count_D10": counts["D10"],

            "count_D20": counts["D20"],

            "count_D40": counts["D40"],

            "area_D00": areas["D00"],

            "area_D10": areas["D10"],

            "area_D20": areas["D20"],

            "area_D40": areas["D40"],

            "has_D00": has_D00,

            "has_D10": has_D10,

            "has_D20": has_D20,

            "has_D40": has_D40,

            "total_instances": total_instances,

            "area_ratio_D00": area_ratio_D00,

            "area_ratio_D10": area_ratio_D10,

            "area_ratio_D20": area_ratio_D20,

            "area_ratio_D40": area_ratio_D40,

            "total_damage_area": total_damage_area,

            "severity_score": severity_score
        }


    # --------------------------------------------------------
    # Handle malformed XML safely
    # --------------------------------------------------------

    except Exception as e:

        print(
            f"ERROR parsing {file_path}: {str(e)}"
        )

        return None


# ============================================================
# 4. READ XML FILES FROM HDFS
# ============================================================

print("\n" + "=" * 70)
print("READING XML FILES FROM HDFS")
print("=" * 70)

xml_rdd = sc.wholeTextFiles(XML_PATH)

xml_count = xml_rdd.count()

print(
    "XML files found:",
    xml_count
)


# ============================================================
# 5. PARSE XML FILES IN PARALLEL
# ============================================================

print("\n" + "=" * 70)
print("PARSING XML FILES USING SPARK")
print("=" * 70)

parsed_rdd = (
    xml_rdd
    .map(parse_xml)
    .filter(lambda x: x is not None)
)


# ============================================================
# 6. CREATE SPARK DATAFRAME
# ============================================================

df = spark.createDataFrame(parsed_rdd)


# ============================================================
# 7. ADD CO-OCCURRENCE FEATURES
# ============================================================

df = (

    df

    # D00 + D20
    .withColumn(
        "co_D00_D20",
        when(
            (col("has_D00") == 1)
            & (col("has_D20") == 1),
            1
        ).otherwise(0)
    )

    # D00 + D40
    .withColumn(
        "co_D00_D40",
        when(
            (col("has_D00") == 1)
            & (col("has_D40") == 1),
            1
        ).otherwise(0)
    )

    # D20 + D40
    .withColumn(
        "co_D20_D40",
        when(
            (col("has_D20") == 1)
            & (col("has_D40") == 1),
            1
        ).otherwise(0)
    )

    # D00 + D20 + D40
    .withColumn(
        "co_D00_D20_D40",
        when(
            (col("has_D00") == 1)
            & (col("has_D20") == 1)
            & (col("has_D40") == 1),
            1
        ).otherwise(0)
    )

    # Number of distinct damage classes
    .withColumn(
        "damage_class_count",
        col("has_D00")
        + col("has_D10")
        + col("has_D20")
        + col("has_D40")
    )
)


# ============================================================
# 8. CACHE DATAFRAME
# ============================================================

df.cache()


# ============================================================
# 9. BASIC VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)


# Total rows

total_rows = df.count()

print(
    "Total parsed rows:",
    total_rows
)


# ============================================================
# 10. PRINT SCHEMA
# ============================================================

print("\nDataFrame schema:")

df.printSchema()


# ============================================================
# 11. SHOW FIRST 10 ROWS
# ============================================================

print("\nFirst 10 rows:")

(
    df
    .orderBy("filename")
    .show(
        10,
        truncate=False
    )
)


# ============================================================
# 12. COUNT DAMAGE-BEARING IMAGES
# ============================================================

damage_bearing_count = (

    df

    .filter(
        col("total_instances") > 0
    )

    .count()
)

print(
    "\nDamage-bearing images:",
    damage_bearing_count
)


# ============================================================
# 13. TOTAL DAMAGE INSTANCES
# ============================================================

total_damage_instances = (

    df

    .selectExpr(
        "sum(total_instances) as total"
    )

    .collect()[0]["total"]
)

print(
    "Total damage instances:",
    total_damage_instances
)


# ============================================================
# 14. DAMAGE CLASS TOTALS
# ============================================================

print(
    "\nDamage class instance totals:"
)

(
    df

    .selectExpr(

        "sum(count_D00) as D00",

        "sum(count_D10) as D10",

        "sum(count_D20) as D20",

        "sum(count_D40) as D40"
    )

    .show()
)


# ============================================================
# 15. DAMAGE CLASS IMAGE PRESENCE
# ============================================================

print(
    "\nDamage class image presence:"
)

(
    df

    .selectExpr(

        "sum(has_D00) as images_with_D00",

        "sum(has_D10) as images_with_D10",

        "sum(has_D20) as images_with_D20",

        "sum(has_D40) as images_with_D40"
    )

    .show()
)


# ============================================================
# 16. CO-OCCURRENCE TOTALS
# ============================================================

print(
    "\nCo-occurrence totals:"
)

(
    df

    .selectExpr(

        "sum(co_D00_D20) as co_D00_D20",

        "sum(co_D00_D40) as co_D00_D40",

        "sum(co_D20_D40) as co_D20_D40",

        "sum(co_D00_D20_D40) as co_D00_D20_D40"
    )

    .show()
)


# ============================================================
# 17. DAMAGE CLASS COUNT DISTRIBUTION
# ============================================================

print(
    "\nDamage class count distribution:"
)

(
    df

    .groupBy(
        "damage_class_count"
    )

    .count()

    .orderBy(
        "damage_class_count"
    )

    .show()
)


# ============================================================
# 18. SAVE FINAL DATAFRAME TO HDFS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "WRITING FINAL FEATURES TO HDFS"
)

print(
    "=" * 70
)

(
    df

    .write

    .mode("overwrite")

    .parquet(
        OUTPUT_PATH
    )
)


# ============================================================
# 19. VERIFY PARQUET OUTPUT
# ============================================================

print(
    "\nParquet output created:"
)

print(
    OUTPUT_PATH
)


# ============================================================
# 20. READ PARQUET AGAIN TO VERIFY
# ============================================================

print(
    "\nVerifying saved Parquet..."
)

saved_df = spark.read.parquet(
    OUTPUT_PATH
)

print(
    "Rows in saved Parquet:",
    saved_df.count()
)

print(
    "Columns in saved Parquet:"
)

print(
    saved_df.columns
)


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "PERSON 1 ETL COMPLETED"
)

print(
    "=" * 70
)

print(
    "Input XML files:",
    xml_count
)

print(
    "Output DataFrame rows:",
    total_rows
)

print(
    "Damage-bearing images:",
    damage_bearing_count
)

print(
    "Total damage instances:",
    total_damage_instances
)

print(
    "Parquet path:",
    OUTPUT_PATH
)

print(
    "=" * 70
)


# ============================================================
# 22. STOP SPARK
# ============================================================

spark.stop()
