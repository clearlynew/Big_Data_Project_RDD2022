# RDD2022 India Road Damage Analysis — Reproducible Big Data Project

## 1. Project overview

This project analyses road-damage annotations from the **India portion of the RDD2022 dataset** using Apache Hadoop HDFS and Apache Spark. XML annotations are converted into image-level features, then analysed with KMeans clustering and FP-Growth association-rule mining. Local Python scripts create plots and annotated grids of representative images.

This is an **annotation-based exploratory big-data analysis**, not a trained road-damage detection model. KMeans groups images using engineered numerical features. The bounding boxes shown in the representative-image figures come from the original XML ground-truth annotations; they are not predictions made by KMeans.

### Objectives

1. Extract structured, image-level features from XML annotation files.
2. Store and process the feature dataset using HDFS and Spark.
3. Evaluate KMeans for values of *K* from 2 through 15.
4. Visualise clustering metrics, cluster sizes, and cluster feature profiles.
5. Inspect representative images and their original annotation boxes.
6. Use FP-Growth to explore frequent damage-class combinations and association rules.
7. Preserve selected final outputs and document how to rerun the analysis.

## 2. Technologies and environment

| Component | Version / environment | Purpose |
|---|---|---|
| Operating system | Ubuntu under WSL on Windows | Development environment |
| Hadoop | 3.3.6 | HDFS storage |
| Apache Spark / PySpark | 3.5.9 | Distributed ETL, KMeans, and FP-Growth |
| Java | OpenJDK 17 | JVM runtime |
| Python | 3.10.12 | PySpark scripts and local analysis |
| pandas | see `requirements.txt` | CSV handling |
| matplotlib | see `requirements.txt` | Plots and grids |
| Pillow | see `requirements.txt` | Image loading and annotation |
| Dataset annotations | XML | Damage classes and bounding boxes |
| Feature and intermediate outputs | Parquet on HDFS | Structured analysis data |

### Check the environment

```bash
java -version
python3 --version
hadoop version
spark-submit --version
jps
```

When HDFS is running, `jps` shows the NameNode and DataNode processes (a SecondaryNameNode may also appear).

### Python virtual environment

From the repository root:

```bash
cd ~/projects/RDD2022_BigData_Project

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Java

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
java -version
```

## 3. Dataset setup

The project uses the **India portion of RDD2022**: the training images and their XML annotations. The raw dataset is not included in this repository and is available from the official RDD2022 distribution.

Directory layout used during development:

```text
RDD2022_India/
└── India/
    └── train/
        ├── images/
        └── annotations/
            └── xmls/
```

### Download and extract

The full release archive `RDD2022_released_through_CRDDC2022.zip` (about 12.35 GB) was downloaded from Figshare and stored in `~/datasets`. The nested India archive was then extracted from it:

```bash
mkdir -p ~/datasets
cd ~/datasets

# Confirm the India archive is inside the full release
unzip -Z1 RDD2022_released_through_CRDDC2022.zip | grep 'RDD2022/India.zip'

# Extract only the nested India archive
unzip -o RDD2022_released_through_CRDDC2022.zip \
  'RDD2022/India.zip' -d ~/datasets
```

Only the training XML annotations are needed for the HDFS feature-extraction stage:

```bash
mkdir -p ~/datasets/India_annotations
unzip -o ~/datasets/RDD2022/India.zip \
  'India/train/annotations/xmls/*.xml' \
  -d ~/datasets/India_annotations

find ~/datasets/India_annotations/India/train/annotations/xmls \
  -maxdepth 1 -type f -name '*.xml' | wc -l
```

The representative-image stages also need the JPEG images locally, so extract `India/train/images/` from the same archive if you want to regenerate the image grids.

### Local paths

`scripts/stage3d_representatives.py` and `scripts/stage3e_cluster_image_grid.py` contain absolute paths from the original WSL machine. Update the path constants near the top of each file (project output directory, image directory, XML directory) to match your setup.

The feature-extraction script reads XML files from HDFS, not from the local XML directory (see Section 6).

### Feature-data counts

| Measure | Count |
|---|---:|
| Feature rows / images represented | 7,706 |
| Images with at least one annotated damage instance | 3,223 |
| Rows with no annotated damage instance | 4,483 |
| Annotated damage instances | 6,831 |
| Images containing more than one damage class | 1,049 |

## 4. Repository structure

```text
RDD2022_BigData_Project/
├── README_RDD2022_Reproducible.md
├── .gitignore
├── requirements.txt
├── scripts/
│   ├── rdd2022_person1.py
│   ├── rdd2022_person2.py
│   ├── stage3a_kmeans.py
│   ├── stage3b_fpgrowth.py
│   ├── stage3c_visualize.py
│   ├── stage3d_representatives.py
│   └── stage3e_cluster_image_grid.py
└── results/
    └── final_analysis/
        ├── kmeans_metrics_final_k2_k15.csv
        ├── cluster_summary_final_k15.csv
        ├── silhouette_scores_k2_k15.png
        ├── kmeans_wcss_elbow_k2_k15.png
        ├── cluster_sizes_k15.png
        ├── cluster_feature_heatmap_k15.png
        ├── representative_images_k15.csv
        ├── representative_bbox_audit_k15.csv
        └── k15_representative_boxes_part*.png
```

### Script responsibilities

| Script | Responsibility |
|---|---|
| `scripts/rdd2022_person1.py` | Reads XML annotations from HDFS, extracts image-level features, and writes a Parquet dataset to HDFS. |
| `scripts/rdd2022_person2.py` | Validates the feature data and displays the output schema and sample rows. |
| `scripts/stage3a_kmeans.py` | Standardises selected features, evaluates K=2–15, and writes metrics, cluster assignments, and summaries to HDFS. |
| `scripts/stage3b_fpgrowth.py` | Builds damage-class transactions from damage-bearing images and writes frequent itemsets and association rules to HDFS. |
| `scripts/stage3c_visualize.py` | Generates local KMeans plots from CSV files in `results/final_analysis/`. |
| `scripts/stage3d_representatives.py` | Selects two high-`severity_score` representative candidates per cluster and checks that the local images exist. |
| `scripts/stage3e_cluster_image_grid.py` | Creates representative-image grids using the original XML ground-truth bounding boxes. |

## 5. HDFS layout

### Feature data

```text
/rdd2022/india/annotations/                 # XML input files
/rdd2022/india/features/india_features.parquet
```

Feature URI used in the scripts:

```text
hdfs://localhost:9000/rdd2022/india/features/india_features.parquet
```

### KMeans outputs

```text
/rdd2022/india/results/kmeans_validation/
├── metrics/
├── clustered_images/
└── cluster_summary/
```

### FP-Growth outputs

```text
/rdd2022/india/results/fpgrowth_all_damaged/
├── frequent_itemsets/
├── association_rules/
└── association_rules_csv/
```

These directories are generated by running the scripts and do not exist in a fresh clone.

## 6. Reproducing the workflow

Run all commands from the project root:

```bash
cd /path/to/RDD2022_BigData_Project
```

Before starting, set the local dataset paths in the representative-image scripts (Section 3). If your NameNode address or port differs from `localhost:9000`, update it in the scripts as well.

### Step 1 — Start and verify HDFS

```bash
jps
hdfs dfs -ls /
```

If the NameNode and DataNode are not running:

```bash
start-dfs.sh
jps
hdfs dfs -ls /
```

### Step 2 — Create the HDFS directories

```bash
hdfs dfs -mkdir -p /rdd2022/india/annotations
hdfs dfs -mkdir -p /rdd2022/india/features
hdfs dfs -mkdir -p /rdd2022/india/results
```

### Step 3 — Upload XML annotations to HDFS

`scripts/rdd2022_person1.py` reads `/rdd2022/india/annotations/*.xml`, so the XML files must sit directly in that directory.

```bash
hdfs dfs -put \
  ~/datasets/India_annotations/India/train/annotations/xmls/*.xml \
  /rdd2022/india/annotations/

hdfs dfs -count /rdd2022/india/annotations
hdfs dfs -ls /rdd2022/india/annotations | head
```

`hdfs dfs -put` fails on files that already exist in the destination. The HDFS input location must match `XML_PATH` in `scripts/rdd2022_person1.py`.

### Step 4 — Extract image-level features

```bash
spark-submit scripts/rdd2022_person1.py
```

Output: `/rdd2022/india/features/india_features.parquet`

```bash
hdfs dfs -ls /rdd2022/india/features
hdfs dfs -du -h /rdd2022/india/features/india_features.parquet
```

### Step 5 — Validate the feature data

```bash
spark-submit scripts/rdd2022_person2.py
```

This prints basic row, column, and schema information for the feature Parquet.

### Step 6 — Run KMeans

```bash
spark-submit scripts/stage3a_kmeans.py
```

The script reads the feature Parquet, keeps images with at least one annotated damage instance, drops rows with missing selected features, standardises the feature vector, evaluates K=2–15, and writes its outputs to `/rdd2022/india/results/kmeans_validation/`.

The nine clustering features are:

```text
count_D00
count_D10
count_D20
count_D40
area_ratio_D00
area_ratio_D10
area_ratio_D20
area_ratio_D40
damage_class_count
```

`severity_score` is not part of the KMeans feature vector; it is used later to rank candidate images within each cluster.

```bash
hdfs dfs -ls /rdd2022/india/results/kmeans_validation
```

### Step 7 — Export KMeans outputs for local visualisation

`stage3c_visualize.py` reads its inputs from `results/final_analysis/`:

```text
results/final_analysis/kmeans_metrics_k2_k15.csv
results/final_analysis/cluster_summary_k2_k15.csv
```

Export the HDFS CSV outputs to those files:

```bash
mkdir -p results/final_analysis

hdfs dfs -getmerge \
  /rdd2022/india/results/kmeans_validation/metrics \
  results/final_analysis/kmeans_metrics_k2_k15.csv

hdfs dfs -getmerge \
  /rdd2022/india/results/kmeans_validation/cluster_summary \
  results/final_analysis/cluster_summary_k2_k15.csv
```

`getmerge` works for CSV part files only. Parquet outputs must be read with Spark or another Parquet-aware reader.

### Step 8 — Generate KMeans visualisations

```bash
python3 scripts/stage3c_visualize.py
```

Outputs in `results/final_analysis/`:

- `silhouette_scores_k2_k15.png`
- `kmeans_wcss_elbow_k2_k15.png`
- `cluster_sizes_k15.png`
- `cluster_feature_heatmap_k15.png`

### Step 9 — Run FP-Growth

```bash
spark-submit scripts/stage3b_fpgrowth.py
```

The script creates one transaction per image with `total_instances > 0`, containing the damage classes present in that image (D00, D10, D20, D40). The transaction population is 3,223 images.

Configured thresholds:

- Minimum support: `0.05`
- Minimum confidence: `0.20`

Outputs:

```text
/rdd2022/india/results/fpgrowth_all_damaged/frequent_itemsets
/rdd2022/india/results/fpgrowth_all_damaged/association_rules
/rdd2022/india/results/fpgrowth_all_damaged/association_rules_csv
```

```bash
hdfs dfs -ls /rdd2022/india/results/fpgrowth_all_damaged/association_rules_csv
hdfs dfs -cat /rdd2022/india/results/fpgrowth_all_damaged/association_rules_csv/part-*.csv
```

Spark's CSV writer cannot handle array columns, so `antecedent` and `consequent` are converted to strings for the CSV export; the Parquet output keeps them as arrays. The script writes in overwrite mode.

### Step 10 — Select representative images

```bash
python3 scripts/stage3d_representatives.py
```

Selects up to two candidates per cluster by descending `severity_score` and checks that each image file exists locally (30 candidates checked, all 30 found).

Output: `results/final_analysis/representative_images_k15.csv`

### Step 11 — Generate annotated representative-image grids

```bash
python3 scripts/stage3e_cluster_image_grid.py
```

Outputs:

```text
results/final_analysis/k15_representative_boxes_part1.png   # clusters 0–4
results/final_analysis/k15_representative_boxes_part2.png   # clusters 5–9
results/final_analysis/k15_representative_boxes_part3.png   # clusters 10–14
```

The script also writes an annotation audit CSV and individual annotated images. All boxes are original XML ground-truth annotations, not detector predictions.

### Step 12 — Check generated artifacts

```bash
ls -lh results/final_analysis/
```

## 7. KMeans methodology and results

### Method

1. Read image-level features from the HDFS Parquet dataset.
2. Filter to images with `total_instances > 0`.
3. Select the nine features listed in Step 6.
4. Drop rows with missing selected features or required fields.
5. Assemble and standardise the feature vector with Spark ML's `VectorAssembler` and `StandardScaler` (`withMean=True`, `withStd=True`).
6. Evaluate K=2–15.
7. Record silhouette score and WCSS (`trainingCost`) for each K.
8. Use the K=15 assignments for the cluster summary and representative-image analysis.

### Results

| Metric | Result |
|---|---:|
| K used for visual analysis | 15 |
| Silhouette score at K=15 | 0.5336528 |
| WCSS at K=15 | 6366.7276 |
| Evaluated range | K=2–15 |

K=15 had the highest silhouette score among the tested values, which makes it the best-scoring candidate within K=2–15 and not necessarily the optimal number of clusters.

### Interpretation and limitations

- Silhouette score measures separation and cohesion under the chosen features and distance measure.
- WCSS decreases as K increases, so it is read alongside the elbow plot.
- Cluster sizes are uneven, and small clusters may reflect rare feature patterns or outliers rather than distinct real-world severity categories.
- The cluster-feature heatmap shows standardised profiles, relative to the scaling used for clustering.
- KMeans is unsupervised; cluster IDs are arbitrary labels, not ordered severity classes.
- `severity_score` is a project-defined score derived from annotation features and has not been independently validated as a road-severity measure.

## 8. Representative-image analysis

The visual analysis uses K=15 and selects two high-`severity_score` candidates per cluster (30 images, all found locally). The grids group clusters 0–4, 5–9, and 10–14. Because candidates are chosen by `severity_score`, they are not necessarily the images closest to a cluster centroid.

Bounding boxes and damage labels come from the original XML annotations. They are ground truth, not KMeans or detector predictions.

Artifacts:

```text
results/final_analysis/representative_images_k15.csv
results/final_analysis/representative_bbox_audit_k15.csv
results/final_analysis/k15_representative_boxes_part1.png
results/final_analysis/k15_representative_boxes_part2.png
results/final_analysis/k15_representative_boxes_part3.png
```

The CSV and audit file record local image paths, so they need to be regenerated after changing the dataset location.

## 9. FP-Growth methodology and results

### Transaction construction

Each damage-bearing image is one transaction containing the damage classes present in it (D00, D10, D20, D40). Images without annotated damage are excluded.

### Configuration and outputs

| Setting / measure | Value |
|---|---:|
| Transaction population | 3,223 damage-bearing images |
| Minimum support | 0.05 |
| Minimum confidence | 0.20 |
| Frequent itemsets saved | 7 |
| Association rules saved | 9 |

A minimum confidence of 0.50 produced no rules, so it was lowered to 0.20.

### Frequent itemsets

| Itemset | Frequency |
|---|---:|
| D20 | 1,758 |
| D40 | 1,530 |
| D00 | 1,109 |
| D40, D20 | 549 |
| D00, D20 | 445 |
| D00, D40 | 363 |
| D00, D40, D20 | 168 |

### Association rules

All nine rules have lift below 1. For example, `{D00, D40} → {D20}` has support ≈ 0.0521, confidence ≈ 0.4628, and lift ≈ 0.8485. A lift below 1 means the antecedent and consequent co-occur less often than expected under independence, so these are not positive associations. Rules describe co-occurrence, not causation.

## 10. Output inventory

### Local outputs (`results/final_analysis/`)

| Output | Description |
|---|---|
| `kmeans_metrics_final_k2_k15.csv` | KMeans evaluation metrics |
| `cluster_summary_final_k15.csv` | K=15 cluster summary |
| `silhouette_scores_k2_k15.png` | Silhouette comparison for K=2–15 |
| `kmeans_wcss_elbow_k2_k15.png` | WCSS curve for K=2–15 |
| `cluster_sizes_k15.png` | Image counts by cluster |
| `cluster_feature_heatmap_k15.png` | Standardised feature profiles by cluster |
| `representative_images_k15.csv` | Selected representative candidates |
| `representative_bbox_audit_k15.csv` | Audit of annotation-box processing |
| `k15_representative_boxes_part1.png` | Annotated representatives, clusters 0–4 |
| `k15_representative_boxes_part2.png` | Annotated representatives, clusters 5–9 |
| `k15_representative_boxes_part3.png` | Annotated representatives, clusters 10–14 |

### Generated HDFS outputs

| HDFS path | Contents |
|---|---|
| `/rdd2022/india/features/india_features.parquet` | Image-level features extracted from XML |
| `/rdd2022/india/results/kmeans_validation/metrics` | KMeans evaluation metrics |
| `/rdd2022/india/results/kmeans_validation/clustered_images` | KMeans cluster assignments |
| `/rdd2022/india/results/kmeans_validation/cluster_summary` | Cluster summaries |
| `/rdd2022/india/results/fpgrowth_all_damaged/frequent_itemsets` | FP-Growth frequent itemsets |
| `/rdd2022/india/results/fpgrowth_all_damaged/association_rules` | FP-Growth association rules |
| `/rdd2022/india/results/fpgrowth_all_damaged/association_rules_csv` | CSV-readable association rules |

## 11. Troubleshooting

| Problem | Check |
|---|---|
| HDFS unavailable | `jps`, `hdfs dfs -ls /`, then `start-dfs.sh` |
| XML input missing | `hdfs dfs -ls /rdd2022/india/annotations`; confirm the path matches `XML_PATH` in `rdd2022_person1.py` |
| Feature Parquet missing | `hdfs dfs -ls /rdd2022/india/features`; rerun Step 4 |
| Spark script fails | `python3 -m py_compile scripts/<script>.py` for syntax; then check HDFS paths and schema |
| CSV export fails on array columns | Spark's CSV writer does not support arrays; the FP-Growth script stringifies them, and Parquet keeps them as arrays |
| Visualisation script can't find CSVs | `ls -lh results/final_analysis/`; export the HDFS outputs as in Step 7 |
| Images or XMLs not found | Check `IMAGE_DIR` and `XML_DIR` in the representative-image scripts and regenerate the representative CSV |

## 12. Reproducibility notes and limitations

- **Dataset availability:** Raw images and XML annotations are not in the repository.
- **Local paths:** The representative-image scripts contain paths from the original WSL machine.
- **HDFS state:** Input and output directories must be created or generated as described above.
- **Generated artifacts:** Intermediate Parquet directories and local runtime files are excluded from Git.
- **Output overwrite:** Some scripts overwrite their output directories, so keep a copy of any earlier results you need.
- **Determinism:** KMeans uses a fixed random seed, but library versions, input rows, and preprocessing changes can alter results.
- **Feature design:** Clustering results depend on the chosen counts, area ratios, class-count features, and preprocessing.
- **Severity score:** `severity_score` is project-defined and not independently validated.
- **Clustering:** Clusters are exploratory; IDs are not ordered labels.
- **FP-Growth:** Results depend on the support and confidence thresholds, and rules do not imply causation.
- **Bounding boxes:** Grids show original XML ground-truth annotations, not detector predictions.
- **Coverage:** 3,223 is the number of damage-bearing images in the cluster summary, not the total number of XML files in the release.

## 13. Conclusion

This project converts road-damage XML annotations into image-level features, processes them with Spark, evaluates KMeans clustering, explores damage-class co-occurrence with FP-Growth, and produces visual inspection artifacts.

K=15 achieved the highest silhouette score among K=2–15, although cluster sizes were uneven and the result remains exploratory. FP-Growth produced seven frequent itemsets and nine rules at the chosen thresholds, all with lift below 1. The representative-image grids provide visual context using original XML annotations rather than predictions from a trained detector.
