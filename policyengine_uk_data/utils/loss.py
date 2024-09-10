import numpy as np
from policyengine_uk import Microsimulation
import pandas as pd
from policyengine_uk_data.storage import STORAGE_FOLDER

df = pd.read_csv(STORAGE_FOLDER / "statistics_by_year.csv")
dfs = []

MIN_YEAR = 2018
MAX_YEAR = 2029

for time_period in range(MIN_YEAR, MAX_YEAR + 1):
    time_period_df = df[
        ["name", "unit", "reference", str(time_period)]
    ].rename(columns={str(time_period): "value"})
    time_period_df["time_period"] = time_period
    dfs.append(time_period_df)

main_df = pd.concat(dfs)
main_df = main_df[main_df.value.notnull()]
main_df.to_csv("statistics.csv", index=False)


def create_target_matrix(
    dataset: str,
    time_period: str,
) -> np.ndarray:
    """
    Create a target matrix A, s.t. for household weights w, the target vector b and a perfectly calibrated PolicyEngine UK:

    A * w = b

    """

    sim = Microsimulation(dataset=dataset)
    sim.default_calculation_period = time_period

    family = sim.populations["benunit"]

    pe = lambda variable: sim.calculate(variable, map_to="household").values

    household_from_family = lambda values: sim.map_result(
        values, "benunit", "household"
    )
    household_from_person = lambda values: sim.map_result(
        values, "person", "household"
    )

    def pe_count(*variables):
        total = 0
        for variable in variables:
            entity = sim.tax_benefit_system.variables[variable].entity.key
            total += sim.map_result(
                sim.calculate(variable) > 0,
                entity,
                "household",
            )

        return total

    df = pd.DataFrame()

    df["attendance_allowance"] = pe("attendance_allowance")
    df["carers_allowance"] = pe("carers_allowance")
    df["dla"] = pe("dla")
    df["esa"] = pe("ESA_income") + pe("ESA_contrib")
    df["esa_contrib"] = pe("ESA_contrib")
    df["esa_income"] = pe("ESA_income")
    df["housing_benefit"] = pe("housing_benefit")
    df["pip"] = pe("pip")
    df["statutory_maternity_pay"] = pe("SMP")
    df["attendance_allowance_count"] = pe_count("attendance_allowance")
    df["carers_allowance_count"] = pe_count("carers_allowance")
    df["dla_count"] = pe_count("dla")
    df["esa_count"] = pe_count("ESA_income", "ESA_contrib")
    df["housing_benefit_count"] = pe_count("housing_benefit")
    df["pension_credit_count"] = pe_count("pension_credit")
    df["pip_count"] = pe_count("pip")

    on_uc = sim.calculate("universal_credit") > 0
    unemployed = family.any(sim.calculate("employment_status") == "UNEMPLOYED")

    df["universal_credit_jobseekers_count"] = household_from_family(
        on_uc * unemployed
    )
    df["universal_credit_non_jobseekers_count"] = household_from_family(
        on_uc * ~unemployed
    )

    df["winter_fuel_payment_count"] = pe_count("winter_fuel_allowance")
    df["capital_gains_tax"] = pe("capital_gains_tax")
    df["child_benefit"] = pe("child_benefit")

    country = sim.calculate("country")
    ct = pe("council_tax")
    df["council_tax"] = ct
    df["council_tax_england"] = ct * (country == "ENGLAND")
    df["council_tax_scotland"] = ct * (country == "SCOTLAND")
    df["council_tax_wales"] = ct * (country == "WALES")

    df["domestic_rates"] = pe("domestic_rates")
    df["fuel_duties"] = pe("fuel_duty")
    df["income_tax"] = pe("income_tax")
    df["jobseekers_allowance"] = pe("JSA_income") + pe("JSA_contrib")
    df["pension_credit"] = pe("pension_credit")
    df["stamp_duty_land_tax"] = pe("expected_sdlt")
    df["state_pension"] = pe("state_pension")
    df["tax_credits"] = pe("tax_credits")
    df["tv_licence_fee"] = pe("tv_licence")

    uc = sim.calculate("universal_credit")
    df["universal_credit"] = household_from_family(uc)
    df["universal_credit_jobseekers"] = household_from_family(uc * unemployed)
    df["universal_credit_non_jobseekers"] = household_from_family(
        uc * ~unemployed
    )

    df["vat"] = pe("vat")
    df["winter_fuel_payment"] = pe("winter_fuel_allowance")

    region = sim.calculate("region", map_to="person")
    region_to_target_name_map = {
        "NORTH_EAST": "north_east",
        "SOUTH_EAST": "south_east",
        "EAST_MIDLANDS": "east_midlands",
        "WEST_MIDLANDS": "west_midlands",
        "YORKSHIRE": "yorkshire_and_the_humber",
        "EAST_OF_ENGLAND": "east",
        "LONDON": "london",
        "SOUTH_WEST": "south_west",
        "WALES": "wales",
        "SCOTLAND": "scotland",
        "NORTHERN_IRELAND": "northern_ireland",
    }
    age = sim.calculate("age")

    for pe_region_name, region_name in region_to_target_name_map.items():
        for lower_age in range(0, 90, 10):
            upper_age = lower_age + 10
            name = f"{region_name}_age_{lower_age}_{upper_age - 1}"
            person_in_criteria = (
                (region == pe_region_name)
                & (age >= lower_age)
                & (age < upper_age)
            )
            df[name] = household_from_person(person_in_criteria)

    targets = (
        statistics[statistics.time_period == int(time_period)]
        .set_index("name")
        .loc[df.columns]
    )

    targets.value = np.select(
        [
            targets.unit == "gbp-bn",
            targets.unit == "person-m",
            targets.unit == "person-k",
            targets.unit == "benefit-unit-m",
            targets.unit == "household-k",
        ],
        [
            targets.value * 1e9,
            targets.value * 1e6,
            targets.value * 1e3,
            targets.value * 1e6,
            targets.value * 1e3,
        ],
    )

    targets

    return df, targets.value


matrix, targets = create_target_matrix("frs_2022", "2022")
target_names = matrix.columns
