# RDD2022 India Road Damage Analysis 

## 1. Project overview

This project analyses road-damage annotations from the **India portion of the RDD2022 dataset** using Apache Hadoop HDFS and Apache Spark. XML annotations are converted into image-level features, then analysed with KMeans clustering and FP-Growth association-rule mining. Local Python scripts create plots and annotated grids of representative images.

This is an **annotation-based exploratory big-data analysis**, not a trained road-damage detection model. KMeans groups images using engineered numerical features. The bounding boxes shown in representative-image figures come from the original XML ground-truth annotations; they are not predictions made by KMeans.

### Objectives

1. Extract structured, image-level features from XML annotation files.
2. Store and process the feature dataset using HDFS and Spark.
3. Evaluate KMeans for values of *K* from 2 through 15.
4. Visualise clustering metrics, cluster sizes, and cluster feature profiles.
5. Inspect representative images and their original annotation boxes.
6. Use FP-Growth to explore frequent damage-class combinations and association rules.
7. Preserve selected final outputs and document how to rerun the analysis.

## 2. Technologies and environment

The following versions were recorded for the development environment. Confirm them on the machine used for reproduction; these values are not a guarantee that every installation has the same versions.

| Component | Reported version / environment | Purpose |
|---|---|---|
| Operating system | Ubuntu under WSL on Windows | Development environment |
| Hadoop | 3.3.6 | HDFS storage |
| Apache Spark / PySpark | 3.5.9 | Distributed ETL, KMeans, and FP-Growth |
| Java | OpenJDK 17 (WSL setup used for the recent run) | JVM runtime |
| Python | 3.10.12 | PySpark scripts and local analysis |
| pandas | Used by local analysis scripts | CSV handling |
| matplotlib | Used by local analysis scripts | Plots and grids |
| Pillow | Used by representative-image script | Image loading and annotation |
| Dataset annotations | XML | Damage classes and bounding boxes |
| Feature and intermediate outputs | Parquet on HDFS | Structured analysis data |

### Check the environment

Run these commands in the Linux/WSL terminal:

```bash
java -version
python3 --version
hadoop version
spark-submit --version
jps
```

When HDFS is running, `jps` should normally show the NameNode and DataNode processes; a SecondaryNameNode may also appear depending on the configuration.

### Python virtual environment and dependencies

The recent WSL setup used Python 3.10.12 and a project-local virtual environment named `.venv`. From the repository root:

```bash
cd ~/projects/RDD2022_BigData_Project

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python -m compileall -q scripts
```

The recorded checks completed successfully. `pip check` verifies installed package requirements, while `compileall` checks Python syntax; neither one proves that every Spark job will run successfully.

If `requirements.txt` does not include a package imported by a local plotting/image script, add that dependency to the requirements file rather than relying on an undeclared local installation. The project also needs separately installed and configured Hadoop and Spark; creating `.venv` does not install them.

For the recent WSL environment, Java was available at `/usr/lib/jvm/java-17-openjdk-amd64`. Use the path that exists on your machine:

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
java -version
spark-submit --version
```

The recent environment used Hadoop 3.3.6, Spark 3.5.9, Java 17, and Python 3.10.12. These are the versions to use as the reproduction baseline for this run, rather than the older versions previously listed in this README.

## 3. Dataset setup

The project uses the **India portion of RDD2022**, specifically the training images and their XML annotations. The raw dataset is not included in this repository. Obtain it from the official RDD2022 distribution or the dataset source provided by your course or supervisor, and follow the applicable dataset licence and access conditions.

The original development environment used directories equivalent to:

```text
RDD2022_India/
└── India/
    └── train/
        ├── images/
        └── annotations/
            └── xmls/
```

These are example directory names, not paths that will automatically exist on another computer. Keep the image files and their corresponding XML files together in the expected structure.

### Download and extract the released dataset

The archive used in the recent WSL run was the full RDD2022 release archive:

```text
RDD2022_released_through_CRDDC2022.zip
```

It was downloaded from the RDD2022 release hosted on Figshare. The full archive is large (approximately 12.35 GB before extraction). Keep it outside the Git repository; do not commit the archive or the extracted image dataset.

The downloaded archive was stored under `~/datasets`. After downloading it, the India country archive was extracted from the outer archive:

```bash
mkdir -p ~/datasets
cd ~/datasets

# Check that the India archive is present inside the full release.
unzip -Z1 RDD2022_released_through_CRDDC2022.zip | grep 'RDD2022/India.zip'

# Extract only the nested India archive.
unzip -o RDD2022_released_through_CRDDC2022.zip \\
  'RDD2022/India.zip' -d ~/datasets
```

The nested archive contains training images and XML annotations, including paths such as `India/train/annotations/xmls/India_000000.xml`. In the recent run, only the training XML annotations were extracted for the HDFS feature-extraction stage:

```bash
unzip -Z1 ~/datasets/RDD2022/India.zip | grep -m 10 -E '\\.xml$'

mkdir -p ~/datasets/India_annotations
unzip -o ~/datasets/RDD2022/India.zip \\
  'India/train/annotations/xmls/*.xml' \\
  -d ~/datasets/India_annotations
```

Count local XML files before uploading them:

```bash
find ~/datasets/India_annotations/India/train/annotations/xmls \\
  -maxdepth 1 -type f -name '*.xml' | wc -l
```

The exact source and extraction paths may differ if you choose another download location. This extraction step does **not** extract the corresponding JPEG images. The representative-image stages need those images locally, so extract the image directory as well if you intend to regenerate representative-image grids.

### Configure local paths before running

The representative-image scripts in the current project version contain absolute paths from the original WSL machine. Before running them on another computer, update the path constants near the top of:

- `scripts/stage3d_representatives.py`
- `scripts/stage3e_cluster_image_grid.py`

Set the project output directory, image directory, and XML directory to the correct local locations. The CSV generated by the representative-image script also records local image paths; regenerate it after changing machines.

The feature-extraction script reads XML files from HDFS rather than directly from the local XML directory. The HDFS setup is described in Section 6.

### Reported feature-data counts

The previous project run reported these counts:

| Measure | Reported count |
|---|---:|
| Feature rows / images represented | 7,706 |
| Images with at least one annotated damage instance | 3,223 |
| Rows with no annotated damage instance | 4,483 |
| Annotated damage instances | 6,831 |
| Images containing more than one damage class | 1,049 |

The first three values describe rows in the generated feature dataset. The damage-instance and multi-class values were reported by earlier validation. Recheck the counts after rebuilding the dataset; they are reported results, not guarantees for a new run.

## 4. Repository structure

The intended Git repository contains source code, documentation, and selected final analysis artifacts:

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

This is the intended cleaned repository structure. Keep only files that actually exist and that you intend to commit. The raw images and XML annotations, local Derby metastore, runtime logs, Python bytecode, Hadoop checksum files, and regenerable intermediate Parquet directories should not be committed.

### Script responsibilities

| Script | Responsibility |
|---|---|
| `scripts/rdd2022_person1.py` | Reads XML annotations from HDFS, extracts image-level features, and writes a Parquet dataset to HDFS. |
| `scripts/rdd2022_person2.py` | Performs the earlier feature-data validation and displays the output schema and sample rows. |
| `scripts/stage3a_kmeans.py` | Standardises selected features, evaluates K=2–15, and writes metrics, cluster assignments, and summaries to HDFS. |
| `scripts/stage3b_fpgrowth.py` | Builds damage-class transactions from damage-bearing images and writes frequent itemsets and association rules to HDFS. |
| `scripts/stage3c_visualize.py` | Generates local KMeans plots from CSV files in `results/final_analysis/`. |
| `scripts/stage3d_representatives.py` | Selects two high-`severity_score` representative candidates per cluster and checks whether local images exist. |
| `scripts/stage3e_cluster_image_grid.py` | Creates representative-image grids using the original XML ground-truth bounding boxes. |

## 5. HDFS layout

The scripts use the following HDFS paths on the local Hadoop cluster.

### Feature data

```text
/rdd2022/india/annotations/                 # XML input files
/rdd2022/india/features/india_features.parquet
```

The corresponding feature URI in the scripts is:

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

These Parquet and CSV output directories are generated by running the scripts. They are not guaranteed to exist in a fresh clone. Older experiment directories may exist in a development HDFS installation, but they are not required as committed repository files.

## 6. Reproducing the workflow

Run the commands from the cloned project root. Replace `/path/to/RDD2022_BigData_Project` with the actual location of your clone:

```bash
cd /path/to/RDD2022_BigData_Project
```

Before running the workflow, configure the local dataset paths in the representative-image scripts as described in Section 3. Also inspect the input and output paths in the scripts if your HDFS NameNode address or port differs from `localhost:9000`.

### Step 1 — Start and verify HDFS

Check whether Hadoop is already running:

```bash
jps
hdfs dfs -ls /
```

If the NameNode and DataNode are not running, start HDFS using the configured installation:

```bash
start-dfs.sh
jps
hdfs dfs -ls /
```

Do not start duplicate services if HDFS is already running. If startup fails, resolve the local Hadoop configuration problem before continuing.

### Step 2 — Create the HDFS directories

```bash
hdfs dfs -mkdir -p /rdd2022/india/annotations
hdfs dfs -mkdir -p /rdd2022/india/features
hdfs dfs -mkdir -p /rdd2022/india/results
```

### Step 3 — Upload XML annotations to HDFS

The current `scripts/rdd2022_person1.py` expects XML files directly under `/rdd2022/india/annotations/` and reads them using the pattern `annotations/*.xml`.

Upload the XML files from your local annotation directory. Replace the local placeholder with the actual path on your machine:

```bash
hdfs dfs -put \\
  ~/datasets/India_annotations/India/train/annotations/xmls/*.xml \\
  /rdd2022/india/annotations/
```

For this project, the feature extraction script reads the XML files directly from `/rdd2022/india/annotations/*.xml`; the XML files should be directly inside that HDFS directory, not nested several directories deep.

Count the HDFS files and inspect the destination after upload:

```bash
hdfs dfs -count /rdd2022/india/annotations
hdfs dfs -ls /rdd2022/india/annotations | head
```

During the recent upload, an intermediate count showed 502 files while the upload was still in progress. That was only a snapshot, not the final dataset count. Always verify the count after the upload has finished.

If the HDFS destination already contains files with the same names, `hdfs dfs -put` may fail rather than overwrite them. Inspect the destination first and deliberately remove or replace old input files only if appropriate.

Verify that the XML files are present:

```bash
hdfs dfs -ls /rdd2022/india/annotations
```

The HDFS input location must match the `XML_PATH` configured in `scripts/rdd2022_person1.py`.

### Step 4 — Extract image-level features

Check Python syntax, then run the feature extraction stage:

```bash
python3 -m py_compile scripts/rdd2022_person1.py
spark-submit scripts/rdd2022_person1.py
```

The script reads XML files from HDFS and writes the feature dataset to:

```text
/rdd2022/india/features/india_features.parquet
```

Verify that the output exists:

```bash
hdfs dfs -ls /rdd2022/india/features
hdfs dfs -du -h /rdd2022/india/features/india_features.parquet
```

If the stage fails, verify that HDFS is running, the XML files are present at the configured input path, the files are valid, and the Spark/Hadoop configuration can access HDFS.

### Step 5 — Run the earlier feature validation

```bash
python3 -m py_compile scripts/rdd2022_person2.py
spark-submit scripts/rdd2022_person2.py
```

This stage reads the feature Parquet and prints basic row, column, and schema information. It is an earlier validation stage; the revised KMeans and FP-Growth analyses are implemented in the dedicated `stage3a` and `stage3b` scripts.

### Step 6 — Run KMeans validation

```bash
python3 -m py_compile scripts/stage3a_kmeans.py
spark-submit scripts/stage3a_kmeans.py
```

The script reads the feature Parquet, keeps images with at least one annotated damage instance, removes rows with missing selected features, standardises the feature vector, evaluates K from 2 through 15, and writes its outputs under:

```text
/rdd2022/india/results/kmeans_validation/
```

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

`severity_score` is not part of the KMeans feature vector. The representative-image stage uses it to rank candidate images within each cluster.

Inspect the output directories:

```bash
hdfs dfs -ls /rdd2022/india/results/kmeans_validation
hdfs dfs -ls /rdd2022/india/results/kmeans_validation/metrics
hdfs dfs -ls /rdd2022/india/results/kmeans_validation/clustered_images
hdfs dfs -ls /rdd2022/india/results/kmeans_validation/cluster_summary
```

### Step 7 — Export KMeans outputs for local visualisation

For the latest run, we exported the HDFS text/CSV outputs into a separate `results/current_run/` directory. This keeps the current-run artifacts separate from the existing `results/final_analysis/` files.

```bash
mkdir -p results/current_run

hdfs dfs -getmerge \\
  /rdd2022/india/results/kmeans_validation/metrics \\
  results/current_run/kmeans_metrics.csv

hdfs dfs -getmerge \\
  /rdd2022/india/results/kmeans_validation/cluster_summary \\
  results/current_run/cluster_summary.csv

hdfs dfs -getmerge \\
  /rdd2022/india/results/fpgrowth_all_damaged/association_rules_csv \\
  results/current_run/association_rules.csv
```

`hdfs dfs -getmerge` is appropriate for compatible text outputs such as CSV part files. Do not use it to combine Parquet directories into CSV; Parquet must be read with Spark or another Parquet-aware reader.

The existing `scripts/stage3c_visualize.py` historically reads CSV files from `results/final_analysis/`, not the `results/current_run/` paths above. The new current-run plots were therefore generated directly from the current-run CSVs, without overwriting older final-analysis plots.

Expected input files include:

```text
results/final_analysis/kmeans_metrics_k2_k15.csv
results/final_analysis/cluster_summary_k2_k15.csv
```

Create the directory if necessary:

```bash
mkdir -p results/final_analysis
```

Use Spark or an appropriate export procedure to convert the HDFS outputs into CSV files. Do not rename Parquet data to `.csv`: Parquet is a binary columnar format. The local final-analysis files should correspond to the same run as the HDFS outputs being documented.

### Step 8 — Generate KMeans visualisations

Once the required CSV files are present:

```bash
python3 -m py_compile scripts/stage3c_visualize.py
python3 scripts/stage3c_visualize.py
```

The script uses pandas and matplotlib and writes plots under `results/final_analysis/`, including:

- `silhouette_scores_k2_k15.png`
- `kmeans_wcss_elbow_k2_k15.png`
- `cluster_sizes_k15.png`
- `cluster_feature_heatmap_k15.png`

### Current-run plots (separate from the original final-analysis plots)

After exporting the current-run metrics and cluster summary, these commands generate three plots in `results/current_run/`:

```bash
python - <<'PY'
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

out = Path("results/current_run")
out.mkdir(parents=True, exist_ok=True)
metrics = pd.read_csv(out / "kmeans_metrics.csv")
summary = pd.read_csv(out / "cluster_summary.csv")

plt.figure(figsize=(8, 5))
plt.plot(metrics["k"], metrics["silhouette"], marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Silhouette score")
plt.title("K-Means Model Selection: Silhouette Score")
plt.xticks(metrics["k"])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out / "silhouette_vs_k.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(metrics["k"], metrics["wcss"], marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("WCSS")
plt.title("K-Means Elbow Analysis")
plt.xticks(metrics["k"])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out / "wcss_vs_k.png", dpi=200)
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(summary["prediction"].astype(str), summary["image_count"])
plt.xlabel("Cluster ID")
plt.ylabel("Number of images")
plt.title("Image Distribution Across K-Means Clusters")
plt.tight_layout()
plt.savefig(out / "cluster_sizes.png", dpi=200)
plt.close()

print("Metrics rows:", len(metrics))
print("Best silhouette row:")
print(metrics.loc[metrics["silhouette"].idxmax()].to_string())
print("Cluster summary rows:", len(summary))
print("Images represented:", summary["image_count"].sum())
PY
```

Open the output folder from WSL using Windows Explorer:

```bash
explorer.exe "$(wslpath -w "$PWD/results/current_run")"
```

The plot files are `silhouette_vs_k.png`, `wcss_vs_k.png`, and `cluster_sizes.png`.

### Step 9 — Run FP-Growth

```bash
python3 -m py_compile scripts/stage3b_fpgrowth.py
spark-submit scripts/stage3b_fpgrowth.py
```

The script creates one transaction per image with `total_instances > 0`. Transactions include the damage classes present in each image: D00, D10, D20, and D40. The reported transaction population is 3,223 images.

The configured thresholds are:

- Minimum support: `0.05`
- Minimum confidence: `0.20`

The script writes frequent itemsets, association rules, and a CSV-readable rule export under:

```text
/rdd2022/india/results/fpgrowth_all_damaged/frequent_itemsets
/rdd2022/india/results/fpgrowth_all_damaged/association_rules
/rdd2022/india/results/fpgrowth_all_damaged/association_rules_csv
```

Inspect the CSV export with:

```bash
hdfs dfs -ls /rdd2022/india/results/fpgrowth_all_damaged/association_rules_csv
hdfs dfs -cat /rdd2022/india/results/fpgrowth_all_damaged/association_rules_csv/part-*.csv
```

Spark's CSV data source cannot directly write array-valued columns. The script converts `antecedent` and `consequent` to strings for its CSV export while retaining the array columns in Parquet. The script uses overwrite mode for these output directories, so preserve any previous run that you need before rerunning it.

### Step 10 — Select representative images

First ensure that the KMeans cluster assignments exist in HDFS and that `IMAGE_DIR` and the output directory are correctly configured in `scripts/stage3d_representatives.py`.

```bash
python3 -m py_compile scripts/stage3d_representatives.py
python3 scripts/stage3d_representatives.py
```

The script selects up to two candidates per cluster by descending `severity_score` and checks whether the corresponding local image files exist. The reported run checked 30 candidates and found all 30 image files.

Expected output:

```text
results/final_analysis/representative_images_k15.csv
```

### Step 11 — Generate annotated representative-image grids

Configure the image and XML directories in `scripts/stage3e_cluster_image_grid.py`, and make sure the representative-image CSV exists.

```bash
python3 -m py_compile scripts/stage3e_cluster_image_grid.py
python3 scripts/stage3e_cluster_image_grid.py
```

Expected grid files:

```text
results/final_analysis/k15_representative_boxes_part1.png
results/final_analysis/k15_representative_boxes_part2.png
results/final_analysis/k15_representative_boxes_part3.png
```

The three grids cover clusters 0–4, 5–9, and 10–14, respectively. The script also writes an annotation audit CSV and individual annotated image files. These figures use original XML ground-truth bounding boxes, not predictions from a trained detector.

### Step 12 — Check generated local artifacts

```bash
ls -lh results/final_analysis/
```

Confirm that the required CSV and PNG files exist before referencing them in a report. A successful `py_compile` checks Python syntax only; it does not guarantee that the Spark job will run or that its input schema and paths are correct.

## 7. KMeans methodology and reported results

### Method

1. Read image-level features from the HDFS Parquet dataset.
2. Filter to images with `total_instances > 0`.
3. Select the nine features listed in Section 6.
4. Drop rows with missing selected features or required fields.
5. Assemble and standardise the feature vector using Spark ML's `VectorAssembler` and `StandardScaler` (`withMean=True`, `withStd=True`).
6. Evaluate K from 2 through 15.
7. Record silhouette score and WCSS (`trainingCost`) for each K.
8. Use the K=15 assignments for the current cluster summary and representative-image analysis.

### Best score within the tested range

| Metric | Reported result |
|---|---:|
| K used for the current visual analysis | 15 |
| Silhouette score at K=15 | 0.5336528 |
| WCSS at K=15 | 6366.7276 |
| Evaluated range | K=2–15 |

K=15 had the highest silhouette score among the tested values in the reported run. This makes it the best-scoring candidate **within the tested range**, not proof that 15 clusters is universally optimal. Cluster-size balance and interpretability should also be considered.

### Interpretation and limitations

- The silhouette score evaluates separation and cohesion under the chosen features and distance measure.
- WCSS generally decreases as K increases, so it should be considered with the elbow plot and other evidence rather than in isolation.
- Cluster sizes are uneven, including small clusters. These may reflect uncommon feature patterns or outliers and should not automatically be described as distinct real-world road-severity categories.
- The cluster-feature heatmap shows standardised feature profiles. Values are relative to the scaling used for clustering.
- KMeans is unsupervised. Cluster IDs are arbitrary labels, not ordered severity classes.
- `severity_score` is a project-defined score derived from annotation features. Unless independently validated, it should not be described as a validated engineering or operational road-severity measure.

## 8. Representative-image analysis

The current visual analysis uses K=15, selecting two high-`severity_score` candidates within each cluster. The reported run verified 30 image files, with no missing files.

Relevant artifacts include:

```text
results/final_analysis/representative_images_k15.csv
results/final_analysis/representative_bbox_audit_k15.csv
results/final_analysis/k15_representative_boxes_part1.png
results/final_analysis/k15_representative_boxes_part2.png
results/final_analysis/k15_representative_boxes_part3.png
```

The grids group clusters 0–4, 5–9, and 10–14. These examples help inspect images assigned to each cluster. Since candidates are selected by `severity_score`, they are not necessarily the points closest to a cluster centroid.

The bounding boxes and damage labels are read from the original XML annotations. They are ground-truth annotations, not KMeans predictions or object-detection predictions. The representative-image CSV and audit may contain local file paths; regenerate them after changing the dataset location.

## 9. FP-Growth methodology and reported results

### Transaction construction

Each damage-bearing image becomes one transaction. The transaction contains the damage-class labels present in that image, drawn from D00, D10, D20, and D40. Images without annotated damage instances are excluded.

### Reported configuration and outputs

| Setting / measure | Reported value |
|---|---:|
| Transaction population | 3,223 damage-bearing images |
| Minimum support | 0.05 |
| Minimum confidence | 0.20 |
| Frequent itemsets saved | 7 |
| Association rules saved | 9 |

The initial confidence threshold of 0.50 produced no rules in the revised all-damaged-image run, so it was reduced to 0.20 to inspect more candidate rules. These are analysis settings, not claims of statistical significance.

### Frequent itemsets

| Itemset | Reported frequency |
|---|---:|
| D20 | 1,758 |
| D40 | 1,530 |
| D00 | 1,109 |
| D40, D20 | 549 |
| D00, D20 | 445 |
| D00, D40 | 363 |
| D00, D40, D20 | 168 |

### Association-rule interpretation

The reported nine generated rules all had lift below 1. For example, `{D00, D40} → {D20}` had support of approximately 0.0521, confidence of approximately 0.4628, and lift of approximately 0.8485.

A lift below 1 indicates that the observed co-occurrence is lower than the independence baseline for the antecedent and consequent in this dataset. These rules should therefore not be presented as positive associations stronger than independence. Association rules describe co-occurrence and do not establish causation.

These values are the previously reported results. Re-run the stage and inspect the generated outputs before claiming that a fresh execution reproduces them exactly.

## 10. Output inventory

### Local outputs

Commit selected final artifacts that are useful for reviewing the analysis without rerunning Spark:

| Output | Description |
|---|---|
| `kmeans_metrics_final_k2_k15.csv` | Consolidated KMeans evaluation metrics |
| `cluster_summary_final_k15.csv` | Consolidated K=15 cluster summary |
| `silhouette_scores_k2_k15.png` | Silhouette comparison for K=2–15 |
| `kmeans_wcss_elbow_k2_k15.png` | WCSS curve for K=2–15 |
| `cluster_sizes_k15.png` | Image counts by cluster |
| `cluster_feature_heatmap_k15.png` | Standardised feature profiles by cluster |
| `representative_images_k15.csv` | Selected representative candidates |
| `representative_bbox_audit_k15.csv` | Audit of annotation-box processing |
| `k15_representative_boxes_part1.png` | Annotated representative images for clusters 0–4 |
| `k15_representative_boxes_part2.png` | Annotated representative images for clusters 5–9 |
| `k15_representative_boxes_part3.png` | Annotated representative images for clusters 10–14 |

Keep this inventory aligned with the files actually committed. Older K=8, K=12, or duplicate plots may be retained only if they are discussed in the report or needed to document an experiment comparison.

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

These are generated outputs, not files supplied automatically by Git. Recreate them by running the corresponding stages.

## 11. Troubleshooting

### HDFS is unavailable

```bash
jps
hdfs dfs -ls /
```

If required Hadoop services are not running, start HDFS using the configured installation:

```bash
start-dfs.sh
```

If startup fails, inspect the Hadoop configuration and logs.

### XML annotation input is missing

```bash
hdfs dfs -ls /rdd2022/india/annotations
```

Upload the XML files as described in Step 3. Confirm that the input path matches `XML_PATH` in `scripts/rdd2022_person1.py`.

### Feature Parquet is missing

```bash
hdfs dfs -ls /rdd2022/india/features
```

Run feature extraction after checking that the XML input is available and that the script's HDFS paths are correct.

### Spark script fails

Check syntax:

```bash
python3 -m py_compile scripts/rdd2022_person1.py
python3 -m py_compile scripts/stage3a_kmeans.py
python3 -m py_compile scripts/stage3b_fpgrowth.py
```

Syntax checks do not validate HDFS connectivity, schemas, dependencies, or successful execution.

### CSV export fails on array columns

Spark's CSV data source does not directly support array-valued columns. The FP-Growth script converts `antecedent` and `consequent` to strings for its CSV export. Parquet can retain the array columns.

### Visualisation script cannot find CSV inputs

```bash
ls -lh results/final_analysis/
```

Export the corresponding HDFS metrics and summary outputs to CSV if required files are missing. Do not rename Parquet data to CSV.

### Representative images or XML annotations are missing

Verify that `IMAGE_DIR` and `XML_DIR` in the representative-image scripts point to the correct local dataset directories. Confirm that each image has a matching XML annotation filename and regenerate the representative-image CSV after changing the dataset location.

## Latest execution recorded in this README

The recent execution was run in WSL from `~/projects/RDD2022_BigData_Project` using the Python virtual environment `.venv`. The source dataset was the full RDD2022 release archive, from which the India archive and training XML annotations were extracted. XML files were uploaded to HDFS and the feature-extraction, feature-validation, KMeans, and FP-Growth stages were run.

Confirmed output details from that execution:

- The feature Parquet directory `/rdd2022/india/features/india_features.parquet` contained `_SUCCESS` and two Parquet part files.
- KMeans validation completed successfully for K=2 through K=15.
- `results/current_run/kmeans_metrics.csv` contained 14 data rows, one for each tested K.
- `results/current_run/cluster_summary.csv` contained 15 cluster rows.
- The cluster summary represented 3,223 damage-bearing images.
- The highest observed silhouette score was approximately `0.533653` at K=15; WCSS at K=15 was approximately `6366.727556`.
- FP-Growth output directories for frequent itemsets, association rules, and a CSV export were created in HDFS.
- The current-run plots were generated from the exported metrics and summary CSVs.

**Dataset coverage caution:** 3,223 is the number of damage-bearing images represented in the cluster summary, not the number of all XML files or all images in the original release. Verify the final HDFS annotation file count and feature-data row counts before claiming that the full India dataset was processed. An intermediate HDFS count during upload showed 502 files, but the upload was still in progress at that point; it must not be treated as the final count.

## 12. Reproducibility notes and limitations

- **Dataset availability:** Raw images and XML annotations are not included in the repository. Reproduction requires access to the same dataset version.
- **Local paths:** The representative-image scripts currently contain paths from the original WSL development environment. Update these before running on another machine.
- **HDFS state:** HDFS input and output directories must be created or generated as described in the workflow.
- **Generated artifacts:** Intermediate Parquet directories and local runtime files are excluded from Git. They must be regenerated when needed.
- **Output overwrite:** Some scripts use overwrite mode for their configured outputs. Preserve previous results before rerunning if they are needed for comparison.
- **Runtime determinism:** KMeans uses a fixed random seed, but library versions, input rows, and preprocessing changes can affect results.
- **Feature design:** Clustering results depend on the selected counts, area ratios, class-count features, and preprocessing.
- **Severity score:** `severity_score` is a project-defined annotation-derived score and is not independently validated.
- **Clustering:** Clusters are exploratory groups; cluster IDs are not ordered labels and do not establish causal or semantic categories by themselves.
- **FP-Growth:** Support and confidence settings affect the itemsets and rules returned. Rules describe association, not causation.
- **Bounding boxes:** Representative-image grids display original XML ground-truth annotations; they are not predictions from a trained detection model.
- **Local plots:** Plot generation depends on CSV exports being consistent with the HDFS run being documented.
- **Reported results:** Counts and metrics in this README are from the reported project run. Re-run the pipeline and compare generated files before claiming exact reproduction.

## 13. Conclusion

This project demonstrates a big-data workflow for converting road-damage XML annotations into image-level features, processing them with Spark, evaluating KMeans clustering, exploring damage-class co-occurrence with FP-Growth, and generating visual inspection artifacts.

In the reported run, K=15 achieved the highest silhouette score among K=2–15, although cluster sizes were uneven and the result remains exploratory. FP-Growth produced seven frequent itemsets and nine rules at the revised thresholds; all reported rule lifts were below 1. The representative-image grids provide visual context using original XML annotations rather than predictions from a trained detection model.
