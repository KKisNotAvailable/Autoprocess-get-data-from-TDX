import pandas as pd
import numpy as np
from scipy.special import factorial, gammaln
from scipy.optimize import minimize, basinhopping
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_PATH = "JJinTP_data_TW/"


class LargerStep:
    def __init__(self, stepsize=1.0):  # Adjust stepsize as needed
        self.stepsize = stepsize

    def __call__(self, x):
        return x + np.random.uniform(-self.stepsize, self.stepsize, size=len(x))


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

def travel_log_likelihood(
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


def find_param(initial_params, public_travel_mat, private_travel_mat, show_stats=True):
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

    # parameter bounds
    bounds = [
        (None, 0),  # gamma_public <= 0
        (None, 0),  # gamma_private <= 0
        (0, None),  # delta_public <= 0
        (0, None),  # delta_public <= 0
    ]

    res = minimize(
        travel_log_likelihood,
        init_param_set,
        args=(public_travel_mat, private_travel_mat, public_transport_cnt_mat, private_transport_cnt_mat),
        method='L-BFGS-B',
        bounds=bounds
        # constraints=constraints
    )

    # res = basinhopping(
    #     func=travel_log_likelihood,
    #     x0=init_param_set,
    #     minimizer_kwargs={
    #         "args": (public_travel_mat, private_travel_mat, public_transport_cnt_mat, private_transport_cnt_mat),
    #         "method": 'L-BFGS-B', 
    #         "bounds": bounds
    #     },
    #     niter=200,  # Number of basin-hopping iterations
    #     # take_step=LargerStep(stepsize=0.5),  # Increase step size
    #     seed=527  # For reproducibility
    # )

    if show_stats:
        print(f"Optimization is success: {res.success}")
        print(f"Number of iterations {res.nit}")
        print(f"Calculated likelihood {res.fun}")  # want this number as small as possible
        print(f"Optimal parameters: {res.x}")

    # The following is just to make sure res.fun is giving the correct result.
    # lkh = travel_log_likelihood(
    #     params=res.x,
    #     public_travel_mat=public_travel_mat,
    #     private_travel_mat=private_travel_mat,
    #     public_transport_cnt_mat=public_transport_cnt_mat,
    #     private_transport_cnt_mat=private_transport_cnt_mat
    # )
    # print(lkh)

    return res.x, res.fun  # optimal parameters and corresponding likelihood


def manual_minimize(
    init_val: float, bounds: list, 
    public_travel_mat, private_travel_mat,
    show_plot=False, **step_settings
):
    '''
    Since the minimization function seemed trapped in the local minimum, this
    function manually test different initial values and check where the global
    minimum objective function value locates.

    Parameters
    ----------
    init_val: float.
        This value must lies in the bounds, since we are finding the list
        of values to be tested within the bound based on this value and
        the step settings.
    bounds: list.
        It is expected to receive two values in the list, like this
        [lower_bound, upper_bound].
    show_plot: bool.
        To stat if need to show the likelihood series plot (to check if the
        sequence is converging).
    step_settings
        Currently consist of step_size and step_method. Will use the stated
        settings to find test values within the range, starting from init_val.
    '''
    step_size = step_settings['step_size']
    mthd = step_settings['step_method']

    if mthd == '*':
        l_cnt = abs(int(math.log(abs(bounds[0]), step_size)))
        u_cnt = abs(int(math.log(abs(bounds[1]), step_size)))

        test_vals_l = [init_val / step_size ** i for i in range(1, l_cnt+1)]
        test_vals_u = [init_val * step_size ** i for i in range(1, u_cnt+1)]
    elif mthd == '+':
        l_cnt = int(abs(bounds[0] - init_val) / step_size)
        u_cnt = int(abs(bounds[1] - init_val) / step_size)

        test_vals_l = [init_val - step_size * i for i in range(1, l_cnt+1)]
        test_vals_u = [init_val + step_size * i for i in range(1, u_cnt+1)]

    test_vals = test_vals_l[::-1] + [init_val] + test_vals_u

    # print(test_vals)

    record = []

    for v in test_vals:
        params = {
            "gamma_public": v,  # fixed cost
            "gamma_private": v, 
            "delta_public": 0.5,  # marginal cost (of travel time)
            "delta_private": 0.5, 
            "v": 1  # kind of like the likelihood of substitution between modes, close to 1 means low substitution.
        }

        optimal_params, lkh = find_param(
            params,
            public_travel_mat=public_travel_mat,
            private_travel_mat=private_travel_mat,
            show_stats=False
        )

        record.append({'params': optimal_params, 'lkh': lkh})

    srs = [(rec['lkh'] - 1639.58845) * 1000000 for rec in record]
    srs = [val for val in srs if val < 3]

    if show_plot:
        plt.plot(srs, marker='o', markersize=3, linestyle='-', color='black', label='Series')
        # plt.xticks(ticks=range(len(srs)), labels=test_vals, fontsize=8)
        plt.ylim(2.4605, 2.4615)
        plt.xticks(ticks=[0, len(srs)-1], labels=[min(test_vals), max(test_vals)], fontsize=8)
        plt.grid(axis='y', linestyle='--', alpha=0.7)  # "True" for both direction
        plt.show()

    opt_param = min(record, key=lambda x: x['lkh'])

    return opt_param


def main():
    print("Start getting travel cost...")

    public_travel_mat = pd.read_csv(f"{DATA_PATH}public_town_travel_mat.csv")
    towns = public_travel_mat.columns

    public_travel_mat = public_travel_mat.values / 60
    private_travel_mat = pd.read_csv(f"{DATA_PATH}private_town_travel_mat.csv").values
    private_travel_mat = private_travel_mat / 60

    # Manual search for 'global' minimum 
    # (tested on the range of -99 ~ -1 ~ -0.01, and found the minimum is more
    # likely lies in -1~-0.01, based on the plot of the likelihood values)
    step_setting = {
        "step_size": 0.01,  # other test: 1
        "step_method": '+'
    }

    opt = manual_minimize(
        init_val=-1, 
        bounds=[-1, -0.01],  # other test: [-99, -1]
        public_travel_mat=public_travel_mat,
        private_travel_mat=private_travel_mat,
        **step_setting
    )
    optimal_params = opt['params']


    public_vill_travel_mat = pd.read_csv(f"{DATA_PATH}public_vill_travel_mat.csv")
    vills = public_vill_travel_mat.columns

    public_vill_travel_mat = public_vill_travel_mat.values / 60
    private_vill_travel_mat = pd.read_csv(f"{DATA_PATH}private_vill_travel_mat.csv").values
    private_vill_travel_mat = private_vill_travel_mat / 60


    tc_town = travel_cost(
        optimal_params,
        public_travel_mat=public_travel_mat,
        private_travel_mat=private_travel_mat
    )

    tc_town = pd.DataFrame(tc_town, columns=towns)

    tc_vill = travel_cost(
        optimal_params,
        public_travel_mat=public_vill_travel_mat,
        private_travel_mat=private_vill_travel_mat
    )

    tc_vill = pd.DataFrame(tc_vill, columns=vills)

    print(tc_vill)

    # tc.to_csv(DATA_PATH+"towns_commuting_cost.csv", index=False)


if __name__ == "__main__":
    main()
