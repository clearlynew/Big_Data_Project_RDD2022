from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path("results/final_analysis")
METRICS = BASE / "kmeans_metrics_k2_k15.csv"
SUMMARY = BASE / "cluster_summary_k2_k15.csv"

if not METRICS.exists():
    raise FileNotFoundError(f"Missing metrics file: {METRICS}")
if not SUMMARY.exists():
    raise FileNotFoundError(f"Missing cluster summary: {SUMMARY}")

metrics = pd.read_csv(METRICS).sort_values("k")
summary = pd.read_csv(SUMMARY).sort_values("prediction")

if not set(range(2, 16)).issubset(set(metrics["k"])):
    raise ValueError("Metrics must contain every K from 2 through 15.")
if len(summary) != 15:
    raise ValueError("Expected 15 cluster summaries for K=15.")

# 1. Silhouette scores
best = metrics.loc[metrics["silhouette"].idxmax()]

plt.figure(figsize=(9, 5))
plt.plot(metrics["k"], metrics["silhouette"], marker="o")
plt.scatter(
    [best["k"]], [best["silhouette"]],
    marker="*", s=180, label=f'Highest: K={int(best["k"])}'
)
plt.xticks(range(2, 16))
plt.xlabel("Number of clusters (K)")
plt.ylabel("Silhouette score")
plt.title("K-Means Silhouette Scores: K=2 to K=15")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(BASE / "silhouette_scores_k2_k15.png", dpi=300)
plt.close()

# 2. WCSS elbow curve
plt.figure(figsize=(9, 5))
plt.plot(metrics["k"], metrics["wcss"], marker="o")
plt.xticks(range(2, 16))
plt.xlabel("Number of clusters (K)")
plt.ylabel("Within-cluster sum of squares (WCSS)")
plt.title("K-Means WCSS Elbow Curve: K=2 to K=15")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(BASE / "kmeans_wcss_elbow_k2_k15.png", dpi=300)
plt.close()

# 3. Cluster sizes
plt.figure(figsize=(10, 5))
plt.bar(summary["prediction"].astype(str), summary["image_count"])
plt.xlabel("Cluster ID")
plt.ylabel("Number of damaged images")
plt.title("Cluster Sizes for K=15")
plt.tight_layout()
plt.savefig(BASE / "cluster_sizes_k15.png", dpi=300)
plt.close()

# 4. Cluster feature-profile heatmap
feature_cols = [
    "avg_count_D00", "avg_count_D10", "avg_count_D20",
    "avg_count_D40", "avg_damage_class_count",
    "avg_severity_score"
]
heat = summary.set_index("prediction")[feature_cols].astype(float)

# Standardise each feature across clusters for comparison.
std = heat.std(axis=0).replace(0, 1)
heat_z = heat.sub(heat.mean(axis=0), axis=1).div(std, axis=1)

fig, ax = plt.subplots(figsize=(11, 7))
im = ax.imshow(heat_z.values, aspect="auto", cmap="coolwarm")
ax.set_xticks(range(len(feature_cols)))
ax.set_xticklabels(feature_cols, rotation=35, ha="right")
ax.set_yticks(range(len(heat_z.index)))
ax.set_yticklabels([f"Cluster {i}" for i in heat_z.index])
ax.set_title("K=15 Cluster Feature Profiles (Standardised)")
fig.colorbar(im, ax=ax, label="Standardised feature value")
fig.tight_layout()
fig.savefig(BASE / "cluster_feature_heatmap_k15.png", dpi=300)
plt.close(fig)

# Save clear final copies of the source data.
metrics.to_csv(BASE / "kmeans_metrics_final_k2_k15.csv", index=False)
summary.to_csv(BASE / "cluster_summary_final_k15.csv", index=False)

print("\nVisualizations generated from K=2..15 metrics and K=15 summary.")
print("Highest silhouette K:", int(best["k"]))
print("Highest silhouette score:", round(float(best["silhouette"]), 6))
print("\nFiles saved:")
for name in [
    "silhouette_scores_k2_k15.png",
    "kmeans_wcss_elbow_k2_k15.png",
    "cluster_sizes_k15.png",
    "cluster_feature_heatmap_k15.png",
    "kmeans_metrics_final_k2_k15.csv",
    "cluster_summary_final_k15.csv",
]:
    p = BASE / name
    print(f"{p}: {p.stat().st_size:,} bytes")
