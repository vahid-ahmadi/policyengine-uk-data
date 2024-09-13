import streamlit as st
# Wide
st.set_page_config(layout="wide")

from policyengine_uk_data.utils import get_loss_results
from policyengine_uk_data import FRS_2022_23, ExtendedFRS_2022_23
from policyengine_core.model_api import Reform
import plotly.express as px
import pandas as pd

st.title("Methodology")

st.write(
    """
In this page, we'll walk through step-by-step the process we use to create PolicyEngine's dataset.
* **Family Resources Survey**: we'll start with the FRS, looking at close it is to reality. To take an actual concrete starting point, we'll assume benefit payments are as reported in the survey.
* **FRS (+ tax-benefit model)**: we need to make sure that our tax-benefit model isn't doing anything unexpected. If we turn on simulation of taxes and benefits, does anything look unexpected? If not- great, we've turned a household survey into something useful for policy analysis. We'll also take stock here of what we're missing from reality.
* **Wealth and consumption**: the most obvious thing we're missing is wealth and consumption. We'll impute those here.
* **Fine-tuning**: we'll use reweighting to make some final adjustments to make sure our dataset is as close to reality as possible.
* **Validation**: we'll compare our dataset to the UK's official statistics, and see how we're doing.
"""
)

st.subheader("Family Resources Survey")

st.write(
    """First, we'll start with the FRS as-is. Skipping over the technical details for how we actually feed this data into the model (you can find that in `policyengine_uk_data/datasets/frs/`), we need to decide how we're actually going to measure 'close to reality'. We need to define an objective function, and if our final dataset improves it a lot, we can call that a success.
         
We'll define this objective function using public statistics that we can generally agree are of high importance to describing the UK household sector. These are things that, if the survey gets them wrong, we'd expect to cause inaccuracy in our model, and if we get them all mostly right, we'd expect to have confidence that it's a pretty accurate tax-benefit model.
         
For this, we've gone through and collected:
         
* **Demographics** from the ONS: ten-year age band populations by region of the UK, national family type populations and national tenure type populations.
* **Incomes** from HMRC: for each of 14 total income bands, the number of people with income and combined income of the seven income types that account for over 99% of total income: employment, self-employment, State Pension, private pension, property, savings interest, and dividends.
* **Tax-benefit programs** from the DWP and OBR: statistics on caseloads, expenditures and revenues for all 20 major tax-benefit programs.
         
Let's first take a look at the initial FRS, our starting point, and what is generally considered the best dataset to use (mostly completely un-modified across major tax-benefit models), and see how close it is to reproducing these statistics.
         
The table below shows the result, and: it's really quite bad! Look at the relative errors.
"""
)


@st.cache_data
def get_loss(dataset, reform, time_period):
    loss_results = get_loss_results(dataset, time_period, reform)

    def get_type(name):
        if "hmrc" in name:
            return "Income"
        if "ons" in name:
            return "Demographics"
        if "obr" in name:
            return "Tax-benefit"
        return "Other"

    loss_results["type"] = loss_results.name.apply(get_type)
    return loss_results

reported_benefits = Reform.from_dict(
    {
        "gov.contrib.policyengine.disable_simulated_benefits": True,
    }
)
loss_results = get_loss(dataset=FRS_2022_23, reform=reported_benefits, time_period=2022).copy()
with st.expander(expanded=True, label="Objective function deep dive"):
    st.dataframe(loss_results, use_container_width=True)

st.write(
    "It's easier to understand 'what kind of bad' this is by splitting out the statistics into those three categories. Here's a histogram of the absolute relative errors."
)

fig = px.histogram(
    loss_results,
    x="abs_rel_error",
    nbins=25,
    title="Distribution of absolute relative errors",
    labels={
        "value": "Absolute relative error",
        "count": "Number of variables",
    },
    color="type",
)

with st.expander(expanded=True, label="Errors by category"):
    st.plotly_chart(fig, use_container_width=True)

st.write(
    """A few notes:
         
* We're comparing things in the same relevant time period (2022), and only doing a tiny amount of adjustment to the statistics: OBR statistics are taken directly from the latest EFO, ONS statistics are the most recent projections for 2022, and HMRC statistics are uprated from 2021 to 2022 using the same standard uprating factors we use in the model (and it's only one year adjustment).
* Demogaphics look basically fine: that's expected, because the DWP applies an optimisation algorithm to optimise the household weights to be as close as possible to a similar set of demographic statistics. It's a good sign that we use slightly different statistics than it was trained on and get good accuracy.
* Incomes look *not great at all*. We'll take a closer look below to understand why. But the FRS is well-known to under-report income significantly.
* Tax-benefit programs also look *not good*. And this is a concern! Because we're using this dataset to answer questions about tax-benefit programs, and the FRS isn't even providing a good representation of them under baseline law.
"""
)

with st.expander(expanded=True, label="Incomes against ground truth"):
    incomes = loss_results[loss_results.type == "Income"]
    incomes["band"] = incomes.name.apply(
        lambda x: x.split("band_")[1].split("_")[0]
    ).astype(int)
    incomes["count"] = incomes.name.apply(lambda x: "count" in x)
    incomes["variable"] = incomes.name.apply(
        lambda x: x.split("_income_band")[0]
        .split("_count")[0]
        .split("hmrc/")[-1]
    )

    variable = st.selectbox(
        "Select income variable", incomes.variable.unique()
    )
    count = st.checkbox("Count")
    variable_df = incomes[
        (incomes.variable == variable) & (incomes["count"] == count)
    ]

    fig = px.bar(
        variable_df,
        x="band",
        y=[
            "target",
            "estimate",
            "error",
            "rel_error",
            "abs_error",
            "abs_rel_error",
        ],
        barmode="group",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("""There are a few interesting things here:
             
* The FRS over-estimates incomes in the upper-middle of the distribution and under-estimates them in the top of the distribution. The reason for this is probably: the FRS misses out the top completely, and then because of the weight optimisation (which scales up the working-age age groups to hit their population targets), the middle of the distribution is inflated, overcompensating.
* Some income types are severely under-estimated across all bands: notably capital incomes. This probably reflects issues with the survey questionnaire design more than sampling bias.
""")
st.write("OK, so what can we do about it?")

st.subheader("FRS (+ tax-benefit model)")

st.write(
    "First, let's turn on the model and check nothing unexpected happens."
)


original_frs_loss = loss_results.copy()
frs_loss = get_loss(FRS_2022_23, None, 2022).copy()
combined_frs_loss = pd.merge(
    on="name",
    left=original_frs_loss,
    right=frs_loss,
    suffixes=("_original", "_simulated"),
)
combined_frs_loss["change_in_abs_rel_error"] = (
    combined_frs_loss["abs_rel_error_simulated"]
    - combined_frs_loss["abs_rel_error_original"]
)
# Sort columns
combined_frs_loss.sort_index(axis=1, inplace=True)
combined_frs_loss = combined_frs_loss.set_index("name")

with st.expander(expanded=True, label="Objective function deep dive"):
    st.dataframe(combined_frs_loss, use_container_width=True)

    st.write(
        """Again, a few notes:
            
* You might be thinking: 'why do some of the HMRC income statistics change?'. That's because of the State Pension, which is simulated in the model. The State Pension is a component of total income, so people might be moved from one income band to another if we adjust their State Pension payments slightly.
* Some of the tax-benefit statistics change, and get better and worse. This is expected for a variety of reasons- one is that incomes and benefits are often out of sync with each other in the data (the income in the survey week might not match income in the benefits assessment time period).
    """
    )

st.subheader("Adding imputations")

st.write("Now, let's add in the imputations for wealth and consumption.")

new_loss = get_loss(ExtendedFRS_2022_23, None, 2022).copy()

st.dataframe(new_loss)