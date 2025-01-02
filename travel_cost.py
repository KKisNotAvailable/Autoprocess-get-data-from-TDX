import pandas as pd
import numpy as np
from scipy.special import factorial, gammaln
from scipy.optimize import minimize
from tqdm import tqdm

DATA_PATH = "JJinTP_data_TW/"

def travel_cost(
    params: list,
    public_travel_mat: np.ndarray,
    private_travel_mat: np.ndarray
):
    g_pub, g_pri, d_pub, d_pri = params
    v = 1  # constant in current version

    tmp_pub = np.exp((-1) * (d_pub * public_travel_mat - g_pub) / v)
    tmp_pri = np.exp((-1) * (d_pri * private_travel_mat - g_pri) / v)

    return -v * np.log(tmp_pub + tmp_pri)

def travle_log_likelihood(
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

    # since scipy's factorial cannot work with 0s, use gammaln instead
    ln_Lt_mat = gammaln(tot_transport_cnt_mat + 1) \
        - gammaln(private_transport_cnt_mat + 1) \
        - gammaln(public_transport_cnt_mat + 1) \
        + private_transport_cnt_mat * np.log(prob_pri_mat) \
        + public_transport_cnt_mat * np.log(prob_pub_mat)
    
    # ln_Lt_mat[have_transport_cnt] => would make the valid (True) values into a list
    ln_L = sum(ln_Lt_mat[have_transport_cnt])

    return -ln_L  # for optimization, using function min but want to find max


def find_param(initial_params, public_travel_mat, private_travel_mat):
    g_pub, g_pri = initial_params["gamma_public"], initial_params["gamma_private"]
    d_pub, d_pri = initial_params["delta_public"], initial_params["delta_private"]
    # v = initial_params["v"]

    init_param_set = [g_pub, g_pri, d_pub, d_pri]

    # 1. using all year combined survey:
    public_transport_cnt_mat = pd.read_csv(f"{DATA_PATH}public_mode_cnt.csv").values
    private_transport_cnt_mat = pd.read_csv(f"{DATA_PATH}private_mode_cnt.csv").values

    # 2. start optimizing
    # constraints example
    constraints = [
        {'type': 'ineq', 'fun': lambda params: params[0] + params[1] + params[2] - 1},  # x + y + z >= 1
        {'type': 'eq', 'fun': lambda params: params[0] - params[2] - 2},                # x - z = 2
        {'type': 'ineq', 'fun': lambda params: params[1]**2 - params[0]},               # y^2 - x >= 0
        {'type': 'ineq', 'fun': lambda params: 5 - params[2]},                          # z <= 5
        {'type': 'ineq', 'fun': lambda params: params[0]}                               # x >= 0
    ]

    res = minimize(
        travle_log_likelihood,
        init_param_set,
        args=(public_travel_mat, private_travel_mat, public_transport_cnt_mat, private_transport_cnt_mat),
        # constraints=constraints
        # can also set bounds to the params
    )

    print(f"Optimization is success: {res.success}")
    print(f"Nuber of iterations {res.nit}")

    return res.x  # optimal parameters


def main():
    print("Start getting travel cost.")

    params = {
        "gamma_public": 1, 
        "gamma_private": 3, 
        "delta_public": 0.1, 
        "delta_private": 0.3, 
        "v": 1
    }

    public_travel_mat = pd.read_csv(f"{DATA_PATH}public_town_travel_mat.csv").values
    public_travel_mat = public_travel_mat / 60
    private_travel_mat = pd.read_csv(f"{DATA_PATH}private_town_travel_mat.csv").values
    private_travel_mat = private_travel_mat / 60

    optimal_params = find_param(
        params,
        public_travel_mat=public_travel_mat,
        private_travel_mat=private_travel_mat
    )

    tc = travel_cost(
        optimal_params,
        public_travel_mat=public_travel_mat,
        private_travel_mat=private_travel_mat
    )

    # print(tc)


if __name__ == "__main__":
    main()
