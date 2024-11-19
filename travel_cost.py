import pandas as pd
import numpy as np
from scipy.special import factorial
from scipy.optimize import minimize
from tqdm import tqdm


filename = 'travel_at_6pm_1'


def to_mat(pair_data: pd.DataFrame):
    '''
    The pair_data is assumed to have (thou not exact same name):
    orig, dest, travel_time_orig2dest, travel_time_dest2orig.

    Notice that for private we assume orig2dest and dest2orig have the same
    travel time. In such case, this function also accepts data with three cols.
    '''

    # return transformed_data.values # to np 2darray

def travel_cost_single_year(
        params: list,
        public_travel_mat: np.ndarray,
        private_travel_mat: np.ndarray,
        public_transport_cnt_mat: np.ndarray,
        private_transport_cnt_mat: np.ndarray
):
    '''
    All the matrix are np 2d array, rather than pd.DataFrame.

    Parameters
    ----------
    params: dict.
        assume the keys are gamma_public, gamma_private, delta_public, 
        delta_private, v (but we are not changing v, set to 1)
    '''
    g_pub, g_pri, d_pub, d_pri = params
    v = 1  # constant in current version

    # Probabilities
    tmp_pub = np.exp((-1) * (d_pub * public_travel_mat - g_pub) / v)
    tmp_pri = np.exp((-1) * (d_pri * private_travel_mat - g_pri) / v)

    prob_pub_mat = tmp_pub / (tmp_pub + tmp_pri)
    prob_pri_mat = tmp_pri / (tmp_pub + tmp_pri)

    # Log-likelihood
    tot_transport_cnt_mat = public_transport_cnt_mat + private_transport_cnt_mat

    # apply later to the result matrix to discard the no survey data location pairs
    have_transport_cnt = tot_transport_cnt_mat > 0

    ln_Lt_mat = np.log(factorial(tot_transport_cnt_mat)) \
        - np.log(factorial(private_transport_cnt_mat)) \
        - np.log(factorial(public_transport_cnt_mat)) \
        + private_transport_cnt_mat * np.log(prob_pri_mat) \
        + public_transport_cnt_mat * np.log(prob_pub_mat)
    
    # ln_Lt_mat[have_transport_cnt] => would make the valid (True) values into a list
    ln_L = sum(ln_Lt_mat[have_transport_cnt])

    return -ln_L  # for optimization, using function min but want to find max

def ttl_travle_cost(
        params: list,
        public_travel_mat: np.ndarray,
        private_travel_mat: np.ndarray
):
    '''
    '''
    # TODO: but do we have survey of travel pairs for each year? or just combined?


def find_param(initial_params):
    g_pub, g_pri = initial_params["gamma_public"], initial_params["gamma_private"]
    d_pub, d_pri = initial_params["delta_public"], initial_params["delta_private"]
    # v = initial_params["v"]

    init_param_set = [g_pub, g_pri, d_pub, d_pri]

    # 1. get the matrix from travel time data
    # public_travel_mat = to_mat()
    # private_travel_mat = to_mat()

    # 2. not sure currently we are using survey for different years or just combined
    #    given using combined:
    # public_transport_cnt_mat = to_mat()
    # private_transport_cnt_mat = to_mat()

    # 3. start optimizing
    # constraints example
    constraints = [
        {'type': 'ineq', 'fun': lambda params: params[0] + params[1] + params[2] - 1},  # x + y + z >= 1
        {'type': 'eq', 'fun': lambda params: params[0] - params[2] - 2},                # x - z = 2
        {'type': 'ineq', 'fun': lambda params: params[1]**2 - params[0]},               # y^2 - x >= 0
        {'type': 'ineq', 'fun': lambda params: 5 - params[2]},                          # z <= 5
        {'type': 'ineq', 'fun': lambda params: params[0]}                               # x >= 0
    ]

    res = minimize(
        travel_cost_single_year,
        init_param_set,
        # args=(public_travel_mat, private_travel_mat, public_transport_cnt_mat, private_transport_cnt_mat),
        # constraints=constraints
        # can also set bounds to the params
    )

    # Extract optimal parameters
    optimal_params = res.x
    optimal_cost = res.fun


def main():
    print("Start getting travel cost.")

    params = {
        "gamma_public": 1, 
        "gamma_private": 3, 
        "delta_public": 0.1, 
        "delta_private": 0.3, 
        "v": 1
    }



if __name__ == "__main__":
    main()
