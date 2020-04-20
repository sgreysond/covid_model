import datetime
import warnings

import pandas as pd
import numpy as np
import scipy.optimize as opt

from typing import NamedTuple


class CovidParams(NamedTuple):
    """Class to define and specify parameter values"""

    time_horizon: int = 60  # Number of days allowed for cases to resolve
    cumulative_mode: bool = True  # Indicator of whether data is cumulative instead of incremental
    epsilon: float = 1e-6  # Small number


class CovidOutcomes:
    """Class to infer death and recovery timeline distributions among reported case data"""

    def __init__(self, input_df, params):

        self.data = input_df
        self.params = params
        self.time_horizon = self.params.time_horizon

        self.validate_input_data()
        self.clean_input()

        self.constrained_history_matrix = self.build_constrained_history_matrix()
        self.right_hand_side = self.build_rhs()

        # self.solution = self.constrained_regress()
        self.raw_solution = opt.nnls(self.constrained_history_matrix, self.right_hand_side, maxiter=100000)[0]
        self.solution = pd.DataFrame({"days_from_diagnosis": list(range(self.time_horizon + 1)),
                                      "death_probability": self.raw_solution[:self.time_horizon + 1],
                                      "recovery_probability": self.raw_solution[self.time_horizon + 1:]})

        print(self.solution)

    def validate_input_data(self):
        """Run checks on input data structure"""

        try:
            assert not self.data.empty
        except AssertionError:
            raise Exception("Input dataframe is empty. Double-check any filters applied to the input.")

        try:
            assert "ObservationDate" in self.data.columns
        except AssertionError:
            raise Exception("'ObservationDate' column is missing from input dataframe")

        try:
            assert "Confirmed" in self.data.columns
        except AssertionError:
            raise Exception("'Confirmed' column is missing from input dataframe")

        try:
            assert "Deaths" in self.data.columns
        except AssertionError:
            raise Exception("'Deaths' column is missing from input dataframe")

        try:
            assert "Recovered" in self.data.columns
        except AssertionError:
            raise Exception("'Recovered' column is missing from input dataframe")

    def cumulative_to_incremental(self, col_name):
        """Convert ordered cumulative data into incremental data."""

        self.data["prev_" + col_name] = self.data[col_name].shift(1).fillna(0).astype(int)
        self.data["incr_" + col_name] = self.data[col_name] - self.data["prev_" + col_name]
        self.data.drop(columns=["prev_" + col_name, col_name], inplace=True)

        if int(self.data["incr_" + col_name].min()) < 0:
            warnings.warn("Data do not increase monotonically for column " +
                          col_name + ". Incremental values are being maxed with zero, but it is recommended that "
                                     "the cumulative nature of the input data be verified.")

            self.data["incr_" + col_name] = self.data["incr_" + col_name].clip(lower=0)

    def fill_date_gaps(self):
        """Check for any gaps in the date range, and fill them."""

        self.start_date = (self.data[self.data["incr_Confirmed"] > 0]["ObservationDate"].min() -
                           datetime.timedelta(days=self.time_horizon)).date()

        self.end_date = self.data["ObservationDate"].max().date()

        if (self.end_date - self.start_date).days < 2 * self.time_horizon:
            warnings.warn("Matrix solution may be underconstrained. More data needed for the given time horizon. "
                          "The time horizon will be shortened as needed.")

            self.time_horizon = int((self.end_date - self.start_date).days / 2) + 3

        full_dates = \
            pd.DataFrame({"ObservationDate": [self.start_date + datetime.timedelta(delta) for delta in
                                              range((self.end_date - self.start_date).days + 1)]})
        full_dates["ObservationDate"] = pd.to_datetime(full_dates["ObservationDate"])

        self.data = full_dates.merge(self.data, on="ObservationDate", how="left").fillna(0)

        # Cast dtypes since nulls create floats
        self.data["incr_Confirmed"] = self.data["incr_Confirmed"].astype(int)
        self.data["incr_Deaths"] = self.data["incr_Deaths"].astype(int)
        self.data["incr_Recovered"] = self.data["incr_Recovered"].astype(int)

    def clean_input(self):
        """Clean input data."""

        # Cast date dtype
        self.data["ObservationDate"] = pd.to_datetime(self.data["ObservationDate"])

        # Aggregate data by date. This is important if, for instance, we are considering US data, but the
        # dataframe includes a state-by-state breakdown.
        self.data = self.data.groupby("ObservationDate").agg("sum").reset_index()

        # Sort by date, lowest to highest
        self.data.sort_values(by=["ObservationDate"], inplace=True)

        # Extract incremental data from cumulative, if necessary
        if self.params.cumulative_mode:
            for col_name in ["Confirmed", "Deaths", "Recovered"]:
                self.cumulative_to_incremental(col_name)

        else:
            for col_name in ["Confirmed", "Deaths", "Recovered"]:
                self.data.rename(columns={col_name: "incr_" + col_name}, inplace=True)

        # Fill in any gaps
        self.fill_date_gaps()

    def build_constrained_history_matrix(self):
        """Build the constraint matrix"""

        num_data_rows_per_outcome = (self.end_date - self.start_date).days - self.time_horizon + 1
        half_num_cols = (self.time_horizon + 1)

        case_vec = self.data["incr_Confirmed"].to_list()

        constraint_matrix = np.ndarray(shape=[2 * num_data_rows_per_outcome + 1, 2*half_num_cols], dtype=np.float64)

        # for outcome in [0, 1]:
        for ind in range(num_data_rows_per_outcome):
            new_case_row = [case_vec[-(ind + delta + 1)] for delta in range(self.time_horizon + 1)]
            constraint_matrix[ind] = np.array(new_case_row + half_num_cols*[0])

        for ind in range(num_data_rows_per_outcome):
            new_case_row = [case_vec[-(ind + delta + 1)] for delta in range(self.time_horizon + 1)]
            constraint_matrix[num_data_rows_per_outcome + ind] = np.array(half_num_cols*[0] + new_case_row)

        constraint_matrix[2 * num_data_rows_per_outcome] = \
            (1/self.params.epsilon) * np.ones(2*half_num_cols).reshape([1, 2*half_num_cols])

        return constraint_matrix

    def build_rhs(self):
        """Construct the right-hand-side results."""

        num_data_rows_per_outcome = (self.end_date - self.start_date).days - self.time_horizon + 1

        first_case_date = self.data[self.data["incr_Confirmed"] > 0]["ObservationDate"].min()
        data_from_first_case = self.data[self.data["ObservationDate"] >= first_case_date]

        rhs = np.array([data_from_first_case["incr_Deaths"].to_list()[::-1] +
                        data_from_first_case["incr_Recovered"].to_list()[::-1] +
                        [1/self.params.epsilon]], np.float64).reshape([2 * num_data_rows_per_outcome + 1,])

        return rhs


if __name__ == "__main__":

    input_data = pd.read_csv("covid_19_data.csv")

    region = "South Korea"
    province = None

    if region is not None:
        input_data = input_data[input_data["Country/Region"] == region]

    if province is not None:
        input_data = input_data[input_data["Province/State"] == province]

    input_data = input_data[["ObservationDate", "Confirmed", "Deaths", "Recovered"]]

    covid_params = CovidParams()

    # Adjust this parameter if the data used is not cumulative.
    # covid_params.cumulative_mode = False

    outcome = CovidOutcomes(input_data, covid_params).solution

    outcome.to_csv("covid_distribitions.csv", index=False)
