"""
k-means clustering of councils by gambling profile.

Groups councils on three measures -- how much is lost per adult, how many
machines are available, and how hard each machine is worked -- without ever
seeing the disadvantage variable. If the resulting clusters still turn out to
be unrelated to disadvantage, that is independent support for the regression.

    python src/clustering.py
"""

import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import EXCLUDE_FROM_FITS, PROCESSED, RANDOM_SEED

FEATURES = ["loss_per_adult", "egms_per_1000_adults", "loss_per_egm"]
K_RANGE = range(2, 9)
STABILITY_RESTARTS = 100


def choose_k(X):
    """Report the elbow and silhouette for each k, then pick by restart stability."""
    print("Choosing k")
    print(f"  {'k':>3} {'WCSS':>9} {'drop':>8} {'silhouette':>12}")
    previous = None
    for k in K_RANGE:
        model = KMeans(n_clusters=k, n_init=50, random_state=RANDOM_SEED).fit(X)
        drop = "" if previous is None else f"{previous - model.inertia_:8.1f}"
        print(f"  {k:>3} {model.inertia_:9.1f} {drop:>8} "
              f"{silhouette_score(X, model.labels_):12.3f}")
        previous = model.inertia_

    print(f"\n  best-silhouette k over {STABILITY_RESTARTS} random restarts:")
    winners = []
    for seed in range(STABILITY_RESTARTS):
        scores = {k: silhouette_score(
            X, KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X).labels_)
            for k in K_RANGE}
        winners.append(max(scores, key=scores.get))
    counts = pd.Series(winners).value_counts().sort_index()
    print(f"    {dict(counts)}")
    best_k = int(counts.idxmax())
    print(f"  -> k = {best_k} (optimal in {counts.max()} of {STABILITY_RESTARTS})")
    return best_k


def name_clusters(profile):
    """Label each cluster by where it sits on loss, availability and intensity."""
    ranked = profile.sort_values("loss_per_adult").index.tolist()
    names = {}
    for rank, cluster in enumerate(ranked):
        level = "Low" if rank == 0 else ("High" if rank == len(ranked) - 1 else "Mid")
        machines = ("many" if profile.loc[cluster, "egms_per_1000_adults"]
                    > profile.egms_per_1000_adults.mean() else "few")
        intensity = ("high" if profile.loc[cluster, "loss_per_egm"]
                     > profile.loss_per_egm.mean() else "low")
        names[cluster] = f"{level} loss / {machines} machines / {intensity} per-machine"
    return names


def main():
    data = pd.read_csv(PROCESSED / "pooled_2021_2025.csv")
    data = data[~data.lga_raw.isin(EXCLUDE_FROM_FITS)].dropna(subset=FEATURES)
    data = data.reset_index(drop=True)

    # Standardise: the three features are on wildly different scales, and
    # k-means measures straight-line distance.
    X = StandardScaler().fit_transform(data[FEATURES])
    print(f"councils: {len(data)}   features: {FEATURES}\n")

    k = choose_k(X)
    model = KMeans(n_clusters=k, n_init=100, random_state=RANDOM_SEED).fit(X)
    data["cluster"] = model.labels_
    print(f"\nfinal silhouette: {silhouette_score(X, model.labels_):.3f}")

    profile = data.groupby("cluster")[FEATURES + ["irsd_score", "median_income",
                                                  "adults"]].mean()
    profile["n"] = data.groupby("cluster").size()
    print("\nCluster profiles (means)")
    print(profile.round(1).to_string())

    names = name_clusters(profile)
    data["cluster_name"] = data.cluster.map(names)
    print()
    for cluster, name in sorted(names.items()):
        print(f"  cluster {cluster}: {name}")

    print("\nDo the clusters line up with disadvantage?")
    data["seifa_tercile"] = pd.qcut(data.irsd_score, 3,
                                    labels=["Most disadv", "Middle", "Least disadv"])
    table = pd.crosstab(data.cluster_name, data.seifa_tercile)
    print(table.to_string())
    chi2, p_value, dof, _ = chi2_contingency(table)
    print(f"\n  chi-square = {chi2:.2f}, dof = {dof}, p = {p_value:.3f}")
    print(f"  -> clusters are "
          f"{'associated with' if p_value < 0.05 else 'INDEPENDENT of'} disadvantage")

    print("\nCouncils in each cluster")
    for cluster in sorted(data.cluster.unique()):
        members = sorted(data[data.cluster == cluster].lga_raw.str.title())
        print(f"\n  [{names[cluster]}]  n = {len(members)}")
        print("    " + ", ".join(members))

    output = PROCESSED / "clustered_councils.csv"
    data.to_csv(output, index=False)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
