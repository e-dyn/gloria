### --- Module Imports --- ###
# Standard Library
from pathlib import Path

# Third Party
import pandas as pd

# Gloria
# Inhouse Packages
from gloria import CalendricData, Gloria, RunConfig, cast_series_to_kind

### --- Global Constants Definitions --- ###
CONFIG_FILE = "run_config"
COMPARE_TO_PROPHET = False
# Note: predicting after deserialization currently only works when there is no
# external regressor
INCLUDE_SERIALIZATION_STEP = False

SEASONALITIES = {
    "weekly": {"period": "7d", "fourier_order": 1, "prior_scale": 0.1},
    "monthly": {
        "period": f"{365.25/12}d",
        "fourier_order": 1,
        "prior_scale": 0.1,
    },
    "quarterly": {
        "period": f"{365.25/4}d",
        "fourier_order": 3,
        "prior_scale": 0.1,
    },
    "yearly": {"period": "365.25d", "fourier_order": 10, "prior_scale": 0.1},
}

### --- Class and Function Definitions --- ###


### --- Main Script --- ###
if __name__ == "__main__":
    basepath = Path(__file__).parent

    config = RunConfig.load_json(basepath / f"run_configs/{CONFIG_FILE}.json")

    timestamp_name = config.data_config.timestamp_name
    metric_name = config.metric_config.metric_name
    df = pd.read_csv(basepath / config.data_config.data_source)
    df[timestamp_name] = pd.to_datetime(df[timestamp_name])
    df[metric_name] = cast_series_to_kind(
        df[metric_name], config.metric_config.dtype_kind
    )

    gloria_pars = {
        **{k: v for k, v in config.data_config if k != "data_source"},
        **{
            k: v
            for k, v in config.metric_config
            if k not in ["dtype_kind", "augmentation_config"]
        },
        **{
            k: v
            for k, v in config.gloria_config
            if k not in ["optimize_mode", "sample"]
        },
    }
    fit_pars = {
        k: v
        for k, v in config.gloria_config
        if k in ["optimize_mode", "sample"]
    }

    model = Gloria(**gloria_pars)

    protocol = CalendricData(
        country="DE",
        yearly_seasonality=False,
        quarterly_seasonality=True,
        monthly_seasonality="auto",
        weekly_seasonality=True,
        holiday_event={"event_type": "Gaussian", "sigma": "3d"},
    )

    model.add_protocol(protocol)
    model.add_external_regressor("ano_deviation", 3.0)
    # Standard Library
    import time

    t0 = time.time()
    model.fit(
        df,
        **fit_pars,
        augmentation_config=config.metric_config.augmentation_config,
    )

    data = model.make_future_dataframe(periods=100)
    data = (
        pd.concat([data, df["ano_deviation"]], axis=1)
        .reset_index(drop=True)
        .fillna(0)
    )

    result = model.predict(data)

    # plot(model, result, include_legend=True, show_changepoints=False)
    model.plot(result, include_legend=True, show_changepoints=False)
    model.plot_components(result, weekly_start=1)
