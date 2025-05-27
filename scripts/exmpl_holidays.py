### --- Module Imports --- ###
# Standard Library
from pathlib import Path

# Third Party
import matplotlib.pyplot as plt
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
    # 'quarterly': {
    #     'period': f'{365.25/4}d',
    #     'fourier_order': 3,
    #     'prior_scale': 0.1
    # },
    # 'yearly': {
    #     'period': '365.25d',
    #     'fourier_order': 10,
    #     'prior_scale': 0.1
    # }
}

### --- Class and Function Definitions --- ###


### --- Main Script --- ###
if __name__ == "__main__":
    basepath = Path(__file__).parent

    config = RunConfig.load_json(basepath / f"run_configs/{CONFIG_FILE}.json")

    timestamp_name = config.data_config.timestamp_name
    metric_name = config.metric_config.metric_name
    df = pd.read_csv(basepath / config.data_config.data_source).iloc[:400, :]
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

    # Add anomaly column as external regressor. Basically as if we knew where
    # anomalies will appear.
    model.add_external_regressor("ano_deviation", prior_scale=0.3)
    protocol = CalendricData(
        country="US",
        yearly_seasonality=False,
        monthly_seasonality="auto",
        holiday_event={"event_type": "Gaussian", "width": "5d"},
    )

    model.add_protocol(protocol)
    # Standard Library
    import time

    t0 = time.time()
    model.fit(
        df,
        **fit_pars,
        augmentation_config=config.metric_config.augmentation_config,
    )

    # Demonstration of model serialization and deserialization work flow
    if INCLUDE_SERIALIZATION_STEP:
        # Save the model
        model_path = basepath / "models/test_model.json"
        model_json = model.to_json(filepath=model_path, indent=2)
        # And load it
        model_new = Gloria.from_json(model_json=model_path, return_as="model")

    data = model.make_future_dataframe(periods=0)
    data = pd.concat([data, df["ano_deviation"]], axis=1).reset_index(
        drop=True
    )
    result = model.predict(data)
    mask = (
        (result[timestamp_name] - result[timestamp_name].min())
        / pd.Timedelta(config.data_config.sampling_period)
    ).apply(float.is_integer)
    result = result[mask]

    if COMPARE_TO_PROPHET:
        # Third Party
        from prophet import Prophet

        params_prophet = {
            "changepoint_prior_scale": 0.05,
            "n_changepoints": 10,
            "uncertainty_samples": 1000,
        }
        model_prophet = Prophet(**params_prophet)
        for name, props in SEASONALITIES.items():
            props["period"] = pd.Timedelta(props["period"]).days
            model_prophet.add_seasonality(name, **props)
        model_prophet.add_country_holidays("DE")
        df_prophet = df.rename(
            columns={timestamp_name: "ds", metric_name: "y"}
        )
        model_prophet.fit(df_prophet)
        result_prophet = model_prophet.predict(data)
        result_prophet = result_prophet[mask]

    fig, ax = plt.subplots(figsize=(12, 8), dpi=140)
    ax.plot(df[timestamp_name], df[metric_name], "o", label="data")
    ax.plot(result[timestamp_name], result["trend"], "black", label="trend")
    ax.plot(result[timestamp_name], result["yhat"], "red", label="fit")
    ax.plot(
        result[timestamp_name],
        result["trend_upper"],
        "black",
        label="trend_upper",
    )
    ax.plot(
        result[timestamp_name],
        result["trend_lower"],
        "black",
        label="trend_lower",
    )
    ax.fill_between(
        result[timestamp_name],
        result["observed_lower"],
        result["observed_upper"],
        color="gray",
        alpha=0.3,
        label="ci",
    )

    if COMPARE_TO_PROPHET:
        ax.plot(
            result_prophet[timestamp_name],
            result_prophet["trend"],
            "green",
            linestyle="--",
            label="trend prophet",
        )
        ax.plot(
            result_prophet[timestamp_name],
            result_prophet["yhat"],
            "green",
            linestyle="--",
            label="fit prophet",
        )

    plt.legend()
    plt.show()

    a = (
        (df.y <= result.observed_lower) | (df.y > result.observed_upper)
    ).sum()
    a /= len(df.y)
    print(a)
