import streamlit as st

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
