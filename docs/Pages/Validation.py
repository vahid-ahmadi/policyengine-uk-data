import streamlit as st
from policyengine_uk_data.utils.download_docs_prerequisites import (
    download_data,
)

download_data()

st.set_page_config(layout="wide")

st.title("Validation")

from policyengine_uk_data import EnhancedFRS_2022_23, FRS_2022_23, SPI_2020_21
from policyengine_uk_data.utils.loss import get_loss_results
import pandas as pd


@st.cache_data
def get_validation():
    df = pd.DataFrame()
    for dataset in [FRS_2022_23, EnhancedFRS_2022_23]:
        for year in range(2022, 2029):
            print(dataset.label, year)
            loss_results = get_loss_results(dataset, year)
            loss_results["time_period"] = year
            loss_results["dataset"] = dataset.label
            df = pd.concat([df, loss_results])
    df = df.reset_index(drop=True)
    return df


df = get_validation()
truth_df = df[df.dataset == df.dataset.unique()[0]].reset_index()
truth_df["estimate"] = truth_df["target"]
truth_df["error"] = truth_df["estimate"] - truth_df["target"]
truth_df["abs_error"] = truth_df["error"].abs()
truth_df["rel_error"] = truth_df["error"] / truth_df["target"]
truth_df["abs_rel_error"] = truth_df["rel_error"].abs()
truth_df["dataset"] = "Official"
df = pd.concat([df, truth_df]).reset_index(drop=True)

st.write(
    "Calibration check: the table below shows how both the original and enhanced FRS datasets compare to over 2,000 official statistics (which the EFRS was explicitly calibrated to hit) from the OBR, DWP and HMRC."
)

st.write(
    "Since the EFRS is calibrated to these statistics, high performance is expected and achieved."
)

a, b = st.columns(2)

with a:
    frs_mean = df[df.dataset == "FRS (2022-23)"].abs_rel_error.mean()
    st.metric("FRS average error", f"{frs_mean:.2%}")
with b:
    efrs_mean = df[df.dataset == "Enhanced FRS (2022-23)"].abs_rel_error.mean()
    st.metric("Enhanced FRS average error", f"{efrs_mean:.2%}")

selected_metrics = st.selectbox("Select statistic", df.name.unique())
comparison = st.selectbox(
    "Select metric",
    ["estimate", "error", "abs_error", "rel_error", "abs_rel_error"],
)

# Bar chart showing datasets and a dotted line for actual

import plotly.express as px

comparison_df = (
    df[df.name == selected_metrics]
    .groupby(["dataset", "time_period"])[comparison]
    .mean()
    .reset_index()
)

fig = px.bar(
    comparison_df,
    x="time_period",
    y=comparison,
    color="dataset",
    barmode="group",
    title=f"{selected_metrics} {comparison} comparison",
)
st.plotly_chart(fig, use_container_width=True)


st.dataframe(df)

st.dataframe(df[df.name == selected_metrics])
