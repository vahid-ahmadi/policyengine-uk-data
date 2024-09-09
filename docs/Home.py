import streamlit as st

st.title("PolicyEngine-UK-Data")

st.write(
    """PolicyEngine-UK-Data is a package to create representative microdata for the UK, designed for input in the PolicyEngine tax-benefit microsimulation model."""
)

st.subheader("What does this repo do?")

st.write(
    """This package creates a (partly synthetic) dataset of households (with incomes, demographics and more) that describes the U.K. household sector. This dataset synthesises multiple sources of data (the Current Population Survey, the IRS Public Use File, and administrative statistics) to improve upon the accuracy of **any** of them."""
)
