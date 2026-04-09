import pandas as pd
from experiment import RollingTestAnalyzer

raw_path = r"artifacts\LGBMRegressor_feat\20260408_154116\rolling_test_raw.csv"
history_df = pd.read_csv(r"data\power_daily_raw.csv")

rolling_df = pd.read_csv(raw_path)
first_target = pd.to_datetime(rolling_df["target_date"]).min()

history_actuals = history_df.loc[
    pd.to_datetime(history_df["date"]) < first_target,
    "OT",
].to_numpy(dtype=float)

analyzer = RollingTestAnalyzer(
    rolling_raw=raw_path,
    history_actuals=history_actuals,
    seasonality=7,
)

print(analyzer.overall_metrics())
print(analyzer.loss_matrix("horizon"))
print(analyzer.loss_matrix("window"))
print(analyzer.loss_summary("window"))
