"""Per-feature diagnostics: distribution + correlation with silver_score."""
import pandas as pd

from src.utils.paths import PROCESSED


FEATURE_MATRIX_PATH = PROCESSED / "feature_matrix.parquet"


def main() -> None:
    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    print(f"Loaded feature matrix: {len(df):,} rows x {len(df.columns)} columns\n")

    meta_cols = {
        "candidate_id",
        "silver_score",
        "honeypot_score",
        "bm25_score",
        "dense_score",
    }
    feature_cols = [c for c in df.columns if c not in meta_cols]

    records = []
    for col in feature_cols:
        series = df[col]
        corr = series.corr(df["silver_score"])
        records.append(
            {
                "feature": col,
                "min": series.min(),
                "mean": series.mean(),
                "median": series.median(),
                "max": series.max(),
                "std": series.std(),
                "corr_with_silver": corr,
                "abs_corr": abs(corr) if pd.notna(corr) else 0.0,
            }
        )

    stats = pd.DataFrame(records)
    stats = stats.sort_values("abs_corr", ascending=False)

    print(
        f"{'feature':<45s} {'min':>6s} {'mean':>6s} {'median':>6s} {'max':>6s} {'std':>6s} {'corr':>7s}"
    )
    print("-" * 90)
    for _, row in stats.iterrows():
        print(
            f"{row['feature']:<45s} "
            f"{row['min']:6.3f} {row['mean']:6.3f} {row['median']:6.3f} "
            f"{row['max']:6.3f} {row['std']:6.3f} {row['corr_with_silver']:7.4f}"
        )

    print("\nTop-5 features by |corr| with silver_score:")
    for name in stats["feature"].head(5):
        corr = stats.loc[stats["feature"] == name, "corr_with_silver"].values[0]
        print(f"  {name:<45s} {corr:7.4f}")

    print("\nBottom-5 features by |corr| with silver_score (candidates for removal):")
    for name in stats["feature"].tail(5):
        corr = stats.loc[stats["feature"] == name, "corr_with_silver"].values[0]
        print(f"  {name:<45s} {corr:7.4f}")


if __name__ == "__main__":
    main()
