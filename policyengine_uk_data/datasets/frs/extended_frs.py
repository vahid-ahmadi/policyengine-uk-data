from policyengine_core.data import Dataset
from policyengine_uk_data.utils.imputations import *
from policyengine_uk_data.storage import STORAGE_FOLDER
from typing import Type
from policyengine_uk_data.datasets.frs.frs import FRS_2022_23
from tqdm import tqdm

class ExtendedFRS(Dataset):
    input_frs: Type[Dataset]

    def generate(self):
        from policyengine_uk import Microsimulation
        create_consumption_model()
        create_vat_model()
        create_wealth_model()

        consumption = Imputation.load(STORAGE_FOLDER / "consumption.pkl")
        vat = Imputation.load(STORAGE_FOLDER / "vat.pkl")
        wealth = Imputation.load(STORAGE_FOLDER / "wealth.pkl")

        data = self.input_frs().load_dataset()
        simulation = Microsimulation(dataset=self.input_frs)
        for imputation_model in tqdm([consumption, vat, wealth], desc="Imputing data"):
            predictors = imputation_model.X_columns

            X_input = simulation.calculate_dataframe(
                predictors, map_to="household"
            )
            if imputation_model == wealth:
                # WAS doesn't sample NI -> put NI households in Wales (closest aggregate)
                X_input.loc[
                    X_input["region"] == "NORTHERN_IRELAND", "region"
                ] = "WALES"
            Y_output = imputation_model.predict(
                X_input, verbose=True
            )

            for output_variable in Y_output.columns:
                values = Y_output[output_variable].values
                data[output_variable] = {self.time_period: values}
        
        self.save_dataset(data)



class ExtendedFRS_2022_23(ExtendedFRS):
    name = "extended_frs_2022_23"
    label = "Extended FRS (2022-23)"
    file_path = STORAGE_FOLDER / "extended_frs_2022_23.h5"
    data_format = Dataset.TIME_PERIOD_ARRAYS
    input_frs = FRS_2022_23
    time_period = 2022

if __name__ == "__main__":
    ExtendedFRS_2022_23().generate()