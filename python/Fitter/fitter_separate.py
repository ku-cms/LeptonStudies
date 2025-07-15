import argparse
import os
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot
from numpy.polynomial.chebyshev import Chebyshev
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from collections.abc import Iterable
from scipy.stats import norm
from scipy.interpolate import BPoly
from scipy.optimize import minimize
from iminuit import cost, Minuit
from scipy import special
from numba_stats import cmsshape
import os
import pandas as pd


x_min, x_max = 65, 115 # GeV

def load_histogram(root_file, hist_name, data_label):
    keys = {key.split(";")[0]: key for key in root_file.keys()}
    if hist_name in keys:
        obj = root_file[keys[hist_name]]
        if isinstance(obj, uproot.behaviors.TH1.Histogram):
            values, edges = obj.to_numpy()
            is_mc = ("MC" in data_label) or ("MC" in hist_name)
            print(f"Histogram: {hist_name}")
            return {"values": values, "edges": edges, "errors": obj.errors(), "is_mc": is_mc}
    return None

def create_fixed_param_wrapper(func, fixed_params):
    def wrapped(x, *free_params):
        full_params = []
        free_idx = 0
        for i in range(len(fixed_params) + len(free_params)):
            if i in fixed_params:
                full_params.append(fixed_params[i])
            else:
                full_params.append(free_params[free_idx])
                free_idx += 1
        return func(x, *full_params)
    return wrapped

def double_crystal_ball(x, mu, sigma, alphaL, nL, alphaR, nR):
    nL = np.clip(nL, 1, 50)
    nR = np.clip(nR, 1, 50)

    z = (x - mu) / sigma
    result = np.zeros_like(z)

    # avoid division by zero
    abs_aL = max(np.abs(alphaL), 1e-8)
    abs_aR = max(np.abs(alphaR), 1e-8)

    # core
    core = np.exp(-0.5 * z**2)
    mask_core = (z > -abs_aL) & (z < abs_aR)
    result[mask_core] = core[mask_core]

    # left tail
    mask_L = z <= -abs_aL
    # log of normalization constant
    logNL = nL * np.log(nL/abs_aL) - 0.5 * abs_aL**2
    tL = (nL/abs_aL - abs_aL - z[mask_L])
    tL = np.maximum(tL, 1e-8)
    result[mask_L] = np.exp(logNL - nL * np.log(tL))

    # right tail
    mask_R = z >= abs_aR
    logNR = nR * np.log(nR/abs_aR) - 0.5 * abs_aR**2
    tR = (nR/abs_aR - abs_aR + z[mask_R])
    tR = np.maximum(tR, 1e-8)
    result[mask_R] = np.exp(logNR - nR * np.log(tR))

    # final normalization
    norm = np.trapezoid(result, x)
    if norm <= 0 or not np.isfinite(norm):
        norm = 1e-8
    return result / norm

def double_voigtian(x, mu, sigma1, gamma1, sigma2, gamma2):
    result = (voigt_profile(x-mu, sigma1, gamma1) + 
              voigt_profile(x-mu, sigma2, gamma2))
    return result / np.trapezoid(result, x)

def double_gaussian(x, mu, sigma):
    # normalized Gaussian
    return np.exp(-0.5*((x-mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi))

def CB_G(x, mu, sigma, alpha, n, sigma2):
    def crystal_ball_unnormalized(x, mu, sigma, alpha, n):
        z = (x - mu) / sigma
        result = np.zeros_like(z)
        abs_alpha = np.abs(alpha)

        # Core region (Gaussian)
        if alpha < 0:
            mask_core = z > -abs_alpha
            mask_tail = z <= -abs_alpha
        else:
            mask_core = z < abs_alpha
            mask_tail = z >= abs_alpha

        result[mask_core] = np.exp(-0.5 * z[mask_core]**2)

        # Tail region (Power law)
        # Calculate N safely using log sum exp trick
        try:
            logN = n * np.log(n / abs_alpha) - 0.5 * abs_alpha**2
            N = np.exp(logN)
        except FloatingPointError:
            N = 1e300  # fallback large number

        base = (n / abs_alpha - abs_alpha - z[mask_tail]) if (alpha < 0) else (n / abs_alpha - abs_alpha + z[mask_tail])
        base = np.clip(base, 1e-15, np.inf)  # prevent zero or negative values

        result[mask_tail] = N * base**(-n)
        return result

    y_cb_un = crystal_ball_unnormalized(x, mu, sigma, alpha, n)

    # Normalize
    integral = np.trapezoid(y_cb_un, x)
    y_cb = y_cb_un / integral

    y_gauss = norm.pdf(x, loc=mu, scale=sigma2)

    y_total = y_cb + y_gauss
    normalization = np.trapezoid(y_total, x)
    if normalization <= 0 or np.isnan(normalization) or np.isinf(normalization):
        return np.zeros_like(y_total)
    return y_total / normalization

def phase_space(x, a, b, x_min=x_min, x_max=x_max):
    # Clip exponents into a safe range
    a_clamped = np.clip(a, 0, 20)
    b_clamped = np.clip(b, 0, 20)

    # 2) Work in log‐space
    t1 = np.clip(x - x_min, 1e-8, None)
    t2 = np.clip(x_max - x, 1e-8, None)

    log_pdf = a_clamped * np.log(t1) + b_clamped * np.log(t2)
    pdf = np.exp(log_pdf - np.max(log_pdf))   # subtract max for stability

    # zero outside
    pdf[(x <= x_min) | (x >= x_max)] = 0

    # Normalize
    norm = np.trapezoid(pdf, x)
    return pdf / (norm if norm>0 else 1e-8)

def linear(x, b, C):
    x_mid = 0.5 * (x_min + x_max)
    lin = b * (x - x_mid) + C

    # Clip negative values
    lin = np.clip(lin, 0, None)

    denom = np.trapezoid(lin, x)

    return lin / denom

def exponential(x, C):
    z = -C * x
    z_max = np.max(z)
    # subtract z_max to stabilize
    exp_z = np.exp(z - z_max)
    # normalize using log-sum-exp
    log_norm = z_max + np.log(np.trapezoid(exp_z, x))
    norm = np.exp(log_norm)

    if not np.isfinite(norm) or norm <= 0:
        return np.zeros_like(x)

    return np.exp(z) / norm

def chebyshev_background(x, *coeffs, x_min=x_min, x_max=x_max):
    x_norm = 2*(x-x_min)/(x_max-x_min) - 1
    return Chebyshev(coeffs)(x_norm) / np.trapezoid(Chebyshev(coeffs)(x_norm), x)

def bernstein_poly(x, *coeffs, x_min = x_min, x_max = x_max):
    c = np.array(coeffs).reshape(-1, 1)
    return BPoly(c, [x_min, x_max])(x)

def cms(x, beta, gamma, loc):
    y = cmsshape.pdf(x, beta, gamma, loc)
    return y

def create_combined_model_integral(fit_type, n_pass_edges):
    if fit_type not in FIT_CONFIGS:
        raise ValueError(f"Unknown fit type: {fit_type}")
    
    config = FIT_CONFIGS[fit_type]
    signal_func = config["signal_func"]
    bg_func = config["background_func"]
    param_names = config["param_names"]
    
    def model(xe, *all_pars):
        P = dict(zip(param_names, all_pars))

        # split raw array
        edges_pass = xe[:n_pass_edges]    
        
        # Get signal parameters - these are directly in P, not prefixed
        signal_params = [P[p] for p in SIGNAL_MODELS[fit_type.split('_')[0]]["params"]]
        
        # Get background parameters - these are also directly in P
        bg_pass_params = [P[p] for p in BACKGROUND_MODELS[fit_type.split('_')[1]]["params"]]

        # build integrals
        signal_pass = signal_func(edges_pass, *signal_params)
        bg_pass = P["B_p"] * bg_func(edges_pass, *bg_pass_params)

        N_p = P["N_p"]

        model_pass = N_p * signal_pass + bg_pass

        return model_pass

    return model

def print_minuit_params_table(minuit_obj):
    # Header
    header = f"{'idx':>3} | {'name':^11} | {'value':^12} | {'error':^12} | {'MINOS -':^12} | {'MINOS +':^12} | {'fixed':^5} | {'lower':^10} | {'upper':^10}"
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    # Rows
    for i, p in enumerate(minuit_obj.params):
        low  = p.lower_limit if p.lower_limit is not None else ""
        high = p.upper_limit if p.upper_limit is not None else ""
        minos_lower = f"{minuit_obj.merrors[p.name].lower:.4f}" if p.name in minuit_obj.merrors else "-"
        minos_upper = f"{minuit_obj.merrors[p.name].upper:.4f}" if p.name in minuit_obj.merrors else "-"
        print(f"{i:3d} | {p.name:11s} | {p.value:12.6f} | {p.error:12.6f} | {minos_lower:>12} | {minos_upper:>12} | "
              f"{str(p.is_fixed):^5s} | {str(low):10s} | {str(high):10s}")
    print(sep)
    
def calculate_custom_chi2(values, errors, model, n_params):
    mask = (errors > 0) & (model > 0) & (values >= 0)
    
    # Calculate Pearson chi2
    Pearson_chi2 = np.sum(((values[mask] - model[mask]) / errors[mask])**2)
    
    # Calculate Poisson chi2
    safe_ratio = np.zeros_like(values)
    ratio = np.divide(values, model, out=safe_ratio, where=(values > 0) & (model > 0))
    log_ratio = np.log(ratio, out=np.zeros_like(ratio), where=(ratio > 0))
    Poisson_terms = model - values + values * log_ratio
    valid_terms = np.isfinite(Poisson_terms) & (Poisson_terms >= 0)
    Poisson_chi2 = 2 * np.sum(Poisson_terms[valid_terms])
    
    # Calculate degrees of freedom
    ndof = mask.sum() - n_params
    
    return Pearson_chi2, Poisson_chi2, ndof

def fit_function(fit_type, hist_pass, fixed_params=None, x_min=x_min, x_max=x_max, interactive=False, args_bin=None, args_data=None, pf="Pass"):
    fixed_params = fixed_params or {}

    if fit_type not in FIT_CONFIGS:
        raise ValueError(f"Unknown fit type: {fit_type}")

    config = FIT_CONFIGS[fit_type]
    param_names = config["param_names"]

    # Prepare data
    edges_pass = hist_pass["edges"]
    values_pass = hist_pass["values"]
    values_pass[values_pass <= 0] = 0.000001
    errors_pass = hist_pass.get("errors", np.sqrt(values_pass))
    errors_pass[errors_pass <= 0] = 0.000001

    bin_mask_pass = (edges_pass[:-1] >= x_min) & (edges_pass[1:] <= x_max)
    edges_pass = edges_pass[np.r_[bin_mask_pass, False] | np.r_[False, bin_mask_pass]]
    centers_pass = 0.5 * (edges_pass[:-1] + edges_pass[1:])
    values_pass = values_pass[bin_mask_pass]
    errors_pass = errors_pass[bin_mask_pass]

    # Calculate data-based initial guesses
    N_p0 = (np.sum(values_pass)) / 2
    B_p_p0 = (np.median(values_pass[-10:]) * len(values_pass))

    for name in ['N_p', 'B_p']:
        if name in fixed_params:
            fixed_params[name]

    bounds = config["bounds"].copy()
    # Use args_data to determine if this is MC or Data
    if args_data is not None and "DATA" in args_data:
        # MC-specific bounds
        bounds.update({
            "N_p": (0, N_p0 * 0.7, np.inf),
            "B_p": (0, B_p_p0 / 2, np.inf),
        })
    elif args_data is not None and "MC" in args_data:
        # Data-specific bounds
        bounds.update({
            "N_p": (0, N_p0 * 0.5, np.inf),
            "B_p": (0, B_p_p0 / 3, np.inf),
        })
    else:
        raise ValueError(f"Unknown args_data value: {args_data}. Expected to contain 'MC' or 'DATA'.")


    # Prepare initial parameter guesses
    p0 = []
    bounds_low = []
    bounds_high = []
    initial_guesses = {}  # All initial guesses are stored here

    for name in param_names:
        if name in fixed_params:
            initial_guesses[name] = fixed_params[name]
            continue
        else:
            # Use middle value from bounds
            b = bounds[name]
            initial_guesses[name] = b[1]
        # Set bounds and add to p0 for minimization
        b = bounds[name]
        p0.append(initial_guesses[name])
        bounds_low.append(b[0])
        bounds_high.append(b[2])

    print(f"Length of edges: {len(edges_pass)}")
    print(f"Length of values: {len(values_pass)}")
    sum_values = np.sum(values_pass)
    print(f"Sum of values: {sum_values}")
    

    model_integral = create_combined_model_integral(fit_type, len(edges_pass))

    c = cost.ExtendedBinnedNLL(values_pass, edges_pass, model_integral, use_pdf='approximate')
    c.errdef = Minuit.LIKELIHOOD
    init = initial_guesses

    m = Minuit(c, *[init[name] for name in param_names], name=param_names)
    m.strategy = 2

    for name in param_names:
        if name in fixed_params:
            init[name] = fixed_params[name]
            m.fixed[name] = True
        elif name in bounds:
            lower = bounds[name][0]
            upper = bounds[name][2]
            # Set to None if not finite
            lower = lower if np.isfinite(lower) else None
            upper = upper if np.isfinite(upper) else None
            m.limits[name] = (lower, upper)


    if interactive:
        m.interactive()

    #m.simplex()
    #m.migrad()
    m.hesse()

    for param in m.parameters:
        try:
            m.minos(param)
        except Exception as e:
            print(f"MINOS failed for parameter {param}: {str(e)}")
    print_minuit_params_table(m)

    print(f"Function value at minimum: {m.fval}")
    print(f"Fit valid: {m.valid}")

    popt = m.values.to_dict()
    perr = m.errors.to_dict()
    cov = m.covariance

    # compute model predictions at bin centers or edges as appropriate
    model_pass_vals = model_integral(edges_pass, *[m.values[name] for name in param_names])    

    # Calculate Pearson and Poisson Chi2 and ndof
    n_params = len([name for name in m.parameters if not m.fixed[name]])
    Pearson_chi2, Poisson_chi2, ndof_pass = calculate_custom_chi2(values_pass, errors_pass, model_pass_vals[1:], n_params)
    total_ndof = (len(values_pass)) - n_params
    Pearson_tot_red_chi2 = Pearson_chi2 / total_ndof
    Poisson_tot_red_chi2 = Poisson_chi2 / total_ndof

    # Output fit summary to terminal
    chi2 = m.fval
    dof = m.ndof
    reduced_chi2 = m.fmin.reduced_chi2

    minos_errors = {}
    for param in m.parameters:
        if param in m.merrors:
            minos_errors[param] = {
                'lower': m.merrors[param].lower,
                'upper': m.merrors[param].upper
            }
        else:
            minos_errors[param] = {
                'lower': -m.errors[param],  # Fall back to symmetric errors
                'upper': m.errors[param]
            }

    results = {
        "minos_errors": minos_errors,
        "fit_type": fit_type,
        "bin": args_bin,
        "m": m,
        "type": fit_type,
        "popt": popt,
        "perr": perr,
        "cov": m.covariance,
        "chi_squared": chi2,
        "reduced_chi_squared": reduced_chi2,
        "dof": dof,
        "Pearson_chi2": Pearson_chi2,
        "Poisson_chi2": Poisson_chi2,
        "Pearson_tot_red_chi2": Pearson_tot_red_chi2,
        "Poisson_tot_red_chi2": Poisson_tot_red_chi2,
        "total_ndof": total_ndof,
        "success": m.valid,
        "message": m.fmin,
        "param_names": param_names,
        "centers_pass": centers_pass,
        "values_pass": values_pass,
        "errors_pass": errors_pass,
        "x_min": x_min,
        "x_max": x_max,
    }
    return results

def plot_combined_fit(results, plot_dir=".", data_type="DATA", fixed_params=None, pf="Pass"):
    if results is None:
        print("No results to plot")
        return None
    
    fixed_params = fixed_params or {}
    fit_type = results["type"]
    config = FIT_CONFIGS[fit_type]
    signal_func = config["signal_func"]
    bg_func = config["background_func"]
    params = results["popt"]
    
    # Get signal and background model names for the legend
    signal_model_name = {
        "dcb": "Double Crystal Ball",
        "dv": "Double Voigtian",
        "g": "Double Gaussian",
        "cbg": "Crystal Ball + Gaussian"
    }.get(fit_type.split('_')[0], "Unknown Signal")
    
    background_model_name = {
        "ps": "Phase Space",
        "lin": "Linear",
        "exp": "Exponential",
        "cheb": "Chebyshev Polynomial",
        "cms": "CMS Shape",
        "bpoly": "Bernstein Polynomial"
    }.get(fit_type.split('_')[1], "Unknown Background")
    
    # Create x values for plotting
    x = np.linspace(results["x_min"], results["x_max"], 1000)
    
    # Get signal parameters
    signal_params = []
    for p in SIGNAL_MODELS[fit_type.split('_')[0]]["params"]:
        signal_params.append(params[p])
    
    # Helper function to format parameters
    def format_param(name, value, error, fixed_params):
        if name in fixed_params:
            return f"{name} = {fixed_params[name]:.3f} (fixed)"
        elif np.isnan(value):
            return f"{name} = NaN"
        elif np.isinf(value):
            return f"{name} = Infinity"
        elif error == 0:
            return f"{name} = {value:.3f} (fixed)"
        else: 
            return f"{name} = {value:.3f} ± {error:.6f}"

    # Plot PASS components
    fig = plt.figure(figsize=(12, 8))  # <-- Save the figure object
    hep.style.use("CMS")
    
    # Get background parameters for pass
    bg_pass_params = []
    for p in BACKGROUND_MODELS[fit_type.split('_')[1]]["params"]:
        bg_pass_params.append(params[p])
    
    # Calculate components
    signal_pass = params["N_p"] * signal_func(x, *signal_params)
    bg_pass = params["B_p"] * bg_func(x, *bg_pass_params)
    total_pass = signal_pass + bg_pass
    
    # Plot pass data and fit with updated legend labels
    plt.errorbar(results["centers_pass"], results["values_pass"], yerr=results["errors_pass"], 
                fmt="o", color="royalblue", markersize=6, capsize=3, label="Data (Pass)")
    plt.plot(x, total_pass, 'k-', label="Total fit")
    plt.plot(x, signal_pass, 'r--', label=f"Signal ({signal_model_name})")
    plt.plot(x, bg_pass, 'g--', label=f"Background ({background_model_name})")
    
    # Formatting
    plt.xlabel("$m_{ee}$ [GeV]", fontsize=12)
    plt.ylabel("Events / GeV", fontsize=12)
    plt.title(f"{data_type.replace('_', ' ')}: {BINS_INFO[results['bin']][1]} GeV ({pf})", pad=10)
    
    # Add fit info
    chi2_red = results["reduced_chi_squared"]

    # For PASS plot:
    signal_params_text = "\n".join([
        format_param(p, params[p], results["perr"][p], fixed_params)
        for p in SIGNAL_MODELS[fit_type.split('_')[0]]["params"]
    ])
    bg_params_text = "\n".join([
        format_param(p, params[p], results["perr"][p], fixed_params)
        for p in BACKGROUND_MODELS[fit_type.split('_')[1]]["params"]
    ])

    info_text = [
        f"N_p = {params['N_p']:.1f} ± {results['perr']['N_p']:.1f}",
        f"B_p = {params['B_p']:.1f} ± {results['perr']['B_p']:.1f}",
        "",
        f"Signal yield: {params['N_p']:.1f}",
        f"Bkg yield: {params['B_p']:.1f}",
        f"NLL: χ²/ndf = {results['chi_squared']:.1f}/{results['dof']} = {chi2_red:.2f}",
        f"Pearson: χ²/ndof = {results['Pearson_chi2']:.1f}/{results['total_ndof']} = {results['Pearson_tot_red_chi2']:.2f}",
        f"Poisson: χ²/ndof = {results['Poisson_chi2']:.1f}/{results['total_ndof']} = {results['Poisson_tot_red_chi2']:.2f}",
        "",
        "Signal params:",
        signal_params_text,
        "",
        "Background params:",
        bg_params_text
    ]

    plt.legend(loc="upper right", fontsize=10)
    plt.gca().text(
        0.02, 0.98,
        "\n".join(info_text),
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.8)
    )
    
    # Save pass plot
    save_path = f"single_fits/{plot_dir}/1D_{data_type}_{results['type']}_fit_{results['bin']}_{pf}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig  # <-- Return the figure object instead of closing it
    # Close the figure
    #plt.close()

BINS_INFO = {
    f"bin{i}": (f"pt_{lo}p00To{hi}p00", f"{lo:.2f}-{hi:.2f}")
    for i, (lo, hi) in enumerate([
        (5,7), (7,10), (10,20), (20,45), (45,75), (75,100), (100,500)
    ])
}

SIGNAL_MODELS = {
    "dcb": {
        "func": double_crystal_ball,
        "params": ["mu", "sigma", "alphaL", "nL", "alphaR", "nR"],
        "bounds": {
            "mu": (88, 90.5, 92),
            "sigma": (1, 3, 6),
            "alphaL": (0, 1.0, 10),
            "nL": (0, 5.0, 30),
            "alphaR": (0, 1.0, 10),
            "nR": (0, 5.0, 30)
        }
    },
    "dv": {
        "func": double_voigtian,
        "params": ["mu", "sigma1", "gamma1", "sigma2", "gamma2"],
        "bounds": {
            "mu": (87, 89, 93),
            "sigma1": (2.0, 3.0, 4.0),
            "gamma1": (0.01, 0.5, 3.0),
            "sigma2": (1.0, 2.0, 3.0),
            "gamma2": (0.5, 1.0, 3.0)
        }
    },
    "g": {
        "func": double_gaussian,
        "params": ["mu", "sigma"],
        "bounds": {
            "mu": (88, 90, 94),
            "sigma": (1, 2.5, 4.0)
        }
    },
    "cbg": {
        "func": CB_G,
        "params": ["mu", "sigma", "alpha", "n", "sigma2"],
        "bounds": {
            "mu": (88, 90, 92),
            "sigma": (1, 3, 6),
            "alpha": (-10, -1, 10),
            "n": (0.1, 5.0, 30),
            "sigma2": (1, 3, 10)
        }
    }
}

# Then define all background models
BACKGROUND_MODELS = {
    "ps": {
        "func": lambda x, a, b: phase_space(x, a, b, x_min=x_min, x_max=x_max),  # Wrap with lambda
        "params": ["a", "b"],
        "bounds": {
            "a": (0, 0.5, 10),
            "b": (0, 1, 30)
        }
    },
    "lin": {
        "func": linear,
        "params": ["b", "C"],
        "bounds": {
            "b": (-1, 0.1, 1),
            "C": (0, 0.1, 10)
        }
    },
    "exp": {
        "func": exponential,
        "params": ["C"],
        "bounds": {
            "C": (-10, 0.1, 10)
        }
    },
    "cheb": {
        "func": chebyshev_background,
        "params": ["c0", "c1", "c2"],
        "bounds": {
            "c0": (0.001, 1, 3),
            "c1": (0.001, 1, 3),
            "c2": (0.001, 1, 3)
        }
    },
    "bpoly": {
        "func": bernstein_poly,
        "params": ["c0", "c1", "c2"],
        "bounds": {
            "c0": (0, 0.05, 10),
            "c1": (0, 0.1, 1),
            "c2": (0, 0.1, 1),
        }
    },
    "cms": {
        "func": cms,
        "params": ["beta", "gamma", "loc"],
        "bounds": {
            "beta": (0.001, 0.1, 10),
            "gamma": (-1, 0.1, 5),   
            "loc": (-100, 90, 200)     
        }
    }

}

# Now combine them into all possible combinations
FIT_CONFIGS = {}
for sig_name, sig_config in SIGNAL_MODELS.items():
    for bg_name, bg_config in BACKGROUND_MODELS.items():
        fit_type = f"{sig_name}_{bg_name}"
        
        # Build parameter names list
        param_names = ["N_p", "B_p"]
        param_names += sig_config["params"]
        param_names += bg_config["params"]
        
        # Build bounds dictionary
        bounds = {
            "N_p": (6000, 7000, np.inf),
            "B_p": (0, 10000, np.inf),
        }
        
        # Add signal bounds
        for p, b in sig_config["bounds"].items():
            bounds[p] = b
        
        # Add background bounds
        for p, b in bg_config["bounds"].items():
            bounds[p] = b
        
        FIT_CONFIGS[fit_type] = {
            "param_names": param_names,
            "bounds": bounds,
            "signal_func": sig_config["func"],
            "background_func": bg_config["func"]
        }

def main():
    parser = argparse.ArgumentParser(description="Fit ROOT histograms with different models.")
    parser.add_argument("--bin", required=True, choices=BINS_INFO.keys())
    parser.add_argument("--type", required=True, choices=FIT_CONFIGS.keys())
    parser.add_argument("--data", required=True, choices=file_paths.keys())
    parser.add_argument("--fix-pass", default="", 
                       help="Comma-separated list of parameters to fix in pass fit in format param1=value1,param2=value2")
    parser.add_argument("--fix-fail", default="", 
                       help="Comma-separated list of parameters to fix in fail fit in format param1=value1,param2=value2")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode")
    args = parser.parse_args()
    
    # NEW FILE PATH
    file_paths = {
        #"DATA_barrel_1_tag":                        "blp_3gt/DATA_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_23D_histos_pt_barrel_1.root",
        "DATA_barrel_1":                            "blp_big/DATA_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_23D_histos_pt_barrel_1.root",
        #"DATA_barrel_1_gold_blp_tag":               "gold_blp_tag/DATA_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_23D_histos_pt_barrel_1.root",
        "DATA_barrel_1_gold_blp":                   "gold_blp_big/DATA_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_23D_histos_pt_barrel_1.root",
        "DATA_barrel_1_silver_blp":                 "silver_blp_big/DATA_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_23D_histos_pt_barrel_1.root",

        #"DATA_barrel_2_tag":                        "blp_3gt/DATA_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_23D_histos_pt_barrel_2.root",
        "DATA_barrel_2":                            "blp_big/DATA_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_23D_histos_pt_barrel_2.root",
        #"DATA_barrel_2_gold_blp_tag":               "gold_blp_tag/DATA_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_23D_histos_pt_barrel_2.root",
        "DATA_barrel_2_gold_blp":                   "gold_blp_big/DATA_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_23D_histos_pt_barrel_2.root",
        "DATA_barrel_2_silver_blp":                 "silver_blp_big/DATA_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_23D_histos_pt_barrel_2.root",

        #"DATA_endcap_tag":                          "blp_3gt/DATA_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_23D_histos_pt_endcap.root",
        "DATA_endcap":                              "blp_big/DATA_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_23D_histos_pt_endcap.root",
        #"DATA_endcap_gold_blp_tag":                 "gold_blp_tag/DATA_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_23D_histos_pt_endcap.root",
        "DATA_endcap_gold_blp":                     "gold_blp_big/DATA_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_23D_histos_pt_endcap.root",
        "DATA_endcap_silver_blp":                   "silver_blp_big/DATA_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_23D_histos_pt_endcap.root",
        
        #"DATA_NEW_barrel_1_tag":                    "blp_3gt/DATA_NEW_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_23D_histos_pt_barrel_1.root",
        "DATA_OLD_barrel_1":                        "blp_big/DATA_OLD_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_OLD_23D_histos_pt_barrel_1.root",  
        #"DATA_NEW_barrel_1_gold_blp_tag":           "gold_blp_tag/DATA_NEW_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_23D_histos_pt_barrel_1.root",
        "DATA_OLD_barrel_1_gold_blp":               "gold_blp_big/DATA_OLD_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_OLD_23D_histos_pt_barrel_1.root",
        "DATA_OLD_barrel_1_silver_blp":             "silver_blp_big/DATA_OLD_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_OLD_23D_histos_pt_barrel_1.root",    

        #"DATA_NEW_barrel_2_tag":                    "blp_3gt/DATA_NEW_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_23D_histos_pt_barrel_2.root",
        "DATA_OLD_barrel_2":                        "blp_big/DATA_OLD_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_OLD_23D_histos_pt_barrel_2.root",
        #"DATA_NEW_barrel_2_gold_blp_tag":           "gold_blp_tag/DATA_NEW_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_23D_histos_pt_barrel_2.root",
        "DATA_OLD_barrel_2_gold_blp":               "gold_blp_big/DATA_OLD_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_OLD_23D_histos_pt_barrel_2.root",
        "DATA_OLD_barrel_2_silver_blp":             "silver_blp_big/DATA_OLD_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_OLD_23D_histos_pt_barrel_2.root",
        
        #"DATA_NEW_endcap_tag":                      "blp_3gt/DATA_NEW_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_23D_histos_pt_endcap.root",
        "DATA_OLD_endcap":                          "blp_big/DATA_OLD_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_OLD_23D_histos_pt_endcap.root",
        #"DATA_NEW_endcap_gold_blp_tag":             "gold_blp_tag/DATA_NEW_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_23D_histos_pt_endcap.root",
        "DATA_OLD_endcap_gold_blp":                 "gold_blp_big/DATA_OLD_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_OLD_23D_histos_pt_endcap.root",
        "DATA_OLD_endcap_silver_blp":               "silver_blp_big/DATA_OLD_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_OLD_23D_histos_pt_endcap.root",

        "DATA_NEW_barrel_1":                        "blp_big/DATA_NEW_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_23D_histos_pt_barrel_1.root",  
        "DATA_NEW_barrel_1_gold_blp":               "gold_blp_big/DATA_NEW_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_23D_histos_pt_barrel_1.root",
        "DATA_NEW_barrel_1_silver_blp":             "silver_blp_big/DATA_NEW_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_23D_histos_pt_barrel_1.root",

        "DATA_NEW_barrel_2":                        "blp_big/DATA_NEW_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_23D_histos_pt_barrel_2.root",
        "DATA_NEW_barrel_2_gold_blp":               "gold_blp_big/DATA_NEW_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_23D_histos_pt_barrel_2.root",
        "DATA_NEW_barrel_2_silver_blp":             "silver_blp_big/DATA_NEW_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_23D_histos_pt_barrel_2.root",
        
        "DATA_NEW_endcap":                          "blp_big/DATA_NEW_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_23D_histos_pt_endcap.root",
        "DATA_NEW_endcap_gold_blp":                 "gold_blp_big/DATA_NEW_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_23D_histos_pt_endcap.root",
        "DATA_NEW_endcap_silver_blp":               "silver_blp_big/DATA_NEW_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_23D_histos_pt_endcap.root",

        "DATA_NEW_2_barrel_1":                      "blp_big/DATA_NEW_2_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_2_23D_histos_pt_barrel_1.root", 
        "DATA_NEW_2_barrel_1_gold_blp":             "gold_blp_big/DATA_NEW_2_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_2_23D_histos_pt_barrel_1.root",
        "DATA_NEW_2_barrel_1_silver_blp":           "silver_blp_big/DATA_NEW_2_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_2_23D_histos_pt_barrel_1.root",

        "DATA_NEW_2_barrel_2":                      "blp_big/DATA_NEW_2_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_2_23D_histos_pt_barrel_2.root",  
        "DATA_NEW_2_barrel_2_gold_blp":             "gold_blp_big/DATA_NEW_2_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_2_23D_histos_pt_barrel_2.root",
        "DATA_NEW_2_barrel_2_silver_blp":           "silver_blp_big/DATA_NEW_2_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_2_23D_histos_pt_barrel_2.root",

        "DATA_NEW_2_endcap":                        "blp_big/DATA_NEW_2_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_2_23D_histos_pt_endcap.root",  
        "DATA_NEW_2_endcap_gold_blp":               "gold_blp_big/DATA_NEW_2_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_2_23D_histos_pt_endcap.root",
        "DATA_NEW_2_endcap_silver_blp":             "silver_blp_big/DATA_NEW_2_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_2_23D_histos_pt_endcap.root",

        #"MC_DY_barrel_1_tag":                       "blp_3gt/MC_DY_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY_23D_histos_pt_barrel_1.root",
        "MC_DY_barrel_1":                           "blp_big/MC_DY_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY_23D_histos_pt_barrel_1.root",
        #"MC_DY_barrel_1_gold_blp_tag":              "gold_blp_tag/MC_DY_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY_23D_histos_pt_barrel_1.root",
        "MC_DY_barrel_1_gold_blp":                  "gold_blp_big/MC_DY_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY_23D_histos_pt_barrel_1.root",
        "MC_DY_barrel_1_silver_blp":                "silver_blp_big/MC_DY_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY_23D_histos_pt_barrel_1.root",

        #"MC_DY_barrel_2_tag":                       "blp_3gt/MC_DY_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY_23D_histos_pt_barrel_2.root",
        "MC_DY_barrel_2":                           "blp_big/MC_DY_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY_23D_histos_pt_barrel_2.root",
        #"MC_DY_barrel_2_gold_blp_tag":              "gold_blp_tag/MC_DY_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY_23D_histos_pt_barrel_2.root",
        "MC_DY_barrel_2_gold_blp":                  "gold_blpd_big/MC_DY_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY_23D_histos_pt_barrel_2.root",
        "MC_DY_barrel_2_silver_blp":                "silver_blp_big/MC_DY_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY_23D_histos_pt_barrel_2.root",

        #"MC_DY_endcap_tag":                         "blp_3gt/MC_DY_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY_23D_histos_pt_endcap.root",
        "MC_DY_endcap":                             "blp_big/MC_DY_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY_23D_histos_pt_endcap.root",
        #"MC_DY_endcap_gold_blp_tag":                "gold_blp_tag/MC_DY_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY_23D_histos_pt_endcap.root",
        "MC_DY_endcap_gold_blp":                    "gold_blp_big/MC_DY_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY_23D_histos_pt_endcap.root",
        "MC_DY_endcap_silver_blp":                  "silver_blp_big/MC_DY_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY_23D_histos_pt_endcap.root",

        #"MC_DY2_2L_2J_barrel_1_tag":                "blp_3gt/MC_DY2_2L_2J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_2J_23D_histos_pt_barrel_1.root",
        "MC_DY2_2L_2J_barrel_1":                    "blp_big/MC_DY2_2L_2J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_2J_23D_histos_pt_barrel_1.root",
        #"MC_DY2_2L_2J_barrel_1_gold_blp_tag":       "gold_blp_tag/MC_DY2_2L_2J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_2J_23D_histos_pt_barrel_1.root",
        "MC_DY2_2L_2J_barrel_1_gold_blp":           "gold_blp_big/MC_DY2_2L_2J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_2J_23D_histos_pt_barrel_1.root",
        "MC_DY2_2L_2J_barrel_1_silver_blp":         "silver_blp_big/MC_DY2_2L_2J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_2J_23D_histos_pt_barrel_1.root",

        #"MC_DY2_2L_2J_barrel_2_tag":                "blp_3gt/MC_DY2_2L_2J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_2J_23D_histos_pt_barrel_2.root",
        "MC_DY2_2L_2J_barrel_2":                    "blp_big/MC_DY2_2L_2J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_2J_23D_histos_pt_barrel_2.root",  
        #"MC_DY2_2L_2J_barrel_2_gold_blp_tag":       "gold_blp_tag/MC_DY2_2L_2J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_2J_23D_histos_pt_barrel_2.root",
        "MC_DY2_2L_2J_barrel_2_gold_blp":           "gold_blp_big/MC_DY2_2L_2J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_2J_23D_histos_pt_barrel_2.root",
        "MC_DY2_2L_2J_barrel_2_silver_blp":         "silver_blp_big/MC_DY2_2L_2J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_2J_23D_histos_pt_barrel_2.root",

        #"MC_DY2_2L_2J_endcap_tag":                  "blp_3gt/MC_DY2_2L_2J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_2J_23D_histos_pt_endcap.root",
        "MC_DY2_2L_2J_endcap":                      "blp_big/MC_DY2_2L_2J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_2J_23D_histos_pt_endcap.root",
        #"MC_DY2_2L_2J_endcap_gold_blp_tag":         "gold_blp_tag/MC_DY2_2L_2J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_2J_23D_histos_pt_endcap.root",
        "MC_DY2_2L_2J_endcap_gold_blp":             "gold_blp_big/MC_DY2_2L_2J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_2J_23D_histos_pt_endcap.root",
        "MC_DY2_2L_2J_endcap_silver_blp":           "silver_blp_big/MC_DY2_2L_2J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_2J_23D_histos_pt_endcap.root",

        #"MC_DY2_2L_4J_barrel_1_tag":                "blp_3gt/MC_DY2_2L_4J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_4J_23D_histos_pt_barrel_1.root",
        "MC_DY2_2L_4J_barrel_1":                    "blp_big/MC_DY2_2L_4J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_4J_23D_histos_pt_barrel_1.root",
        #"MC_DY2_2L_4J_barrel_1_gold_blp_tag":       "gold_blp_tag/MC_DY2_2L_4J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_4J_23D_histos_pt_barrel_1.root", 
        "MC_DY2_2L_4J_barrel_1_gold_blp":           "gold_blp_big/MC_DY2_2L_4J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_4J_23D_histos_pt_barrel_1.root",
        "MC_DY2_2L_4J_barrel_1_silver_blp":         "silver_blp_big/MC_DY2_2L_4J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_4J_23D_histos_pt_barrel_1.root",

        #"MC_DY2_2L_4J_barrel_2_tag":                "blp_3gt/MC_DY2_2L_4J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_4J_23D_histos_pt_barrel_2.root",
        "MC_DY2_2L_4J_barrel_2":                    "blp_big/MC_DY2_2L_4J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_4J_23D_histos_pt_barrel_2.root",
        #"MC_DY2_2L_4J_barrel_2_gold_blp_tag":       "gold_blp_tag/MC_DY2_2L_4J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_4J_23D_histos_pt_barrel_2.root",
        "MC_DY2_2L_4J_barrel_2_gold_blp":           "gold_blp_big/MC_DY2_2L_4J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_4J_23D_histos_pt_barrel_2.root",
        "MC_DY2_2L_4J_barrel_2_silver_blp":         "silver_blp_big/MC_DY2_2L_4J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_4J_23D_histos_pt_barrel_2.root",

        #"MC_DY2_2L_4J_endcap_tag":                  "blp_3gt/MC_DY2_2L_4J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_4J_23D_histos_pt_endcap.root",
        "MC_DY2_2L_4J_endcap":                      "blp_big/MC_DY2_2L_4J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_4J_23D_histos_pt_endcap.root",
        #"MC_DY2_2L_4J_endcap_gold_blp_tag":         "gold_blp_tag/MC_DY2_2L_4J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_4J_23D_histos_pt_endcap.root",
        "MC_DY2_2L_4J_endcap_gold_blp":             "gold_blp_big/MC_DY2_2L_4J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_4J_23D_histos_pt_endcap.root",
        "MC_DY2_2L_4J_endcap_silver_blp":           "silver_blp_big/MC_DY2_2L_4J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_4J_23D_histos_pt_endcap.root",
    }

    # Load data
    try:
        root_file = uproot.open(file_paths[args.data])
    except Exception as e:
        print(f"Error opening file: {e}")
        return

    # Prepare directories
    plot_dir = f"{args.bin}_fits/{'DATA' if args.data.startswith('DATA') else 'MC'}"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Load histograms
    bin_suffix, bin_range = BINS_INFO[args.bin]
    hist_pass = load_histogram(root_file, f"{args.bin}_{bin_suffix}_Pass", args.data)
    hist_fail = load_histogram(root_file, f"{args.bin}_{bin_suffix}_Fail", args.data) 


    print(f"Looking for histograms:")
    print(f"Pass: {args.bin}_{bin_suffix}_Pass")
    print(f"Fail: {args.bin}_{bin_suffix}_Fail")

    if not hist_pass or not hist_fail:
        root_file.close()
        return

    # Parse fixed parameters for pass and fail
    fixed_params_pass = {}
    if args.fix_pass:
        for item in args.fix_pass.split(','):
            try:
                k, v = item.split('=')
                fixed_params_pass[k.strip()] = float(v.strip())
            except:
                print(f"Warning: Ignoring malformed parameter '{item}'")
    
    fixed_params_fail = {}
    if args.fix_fail:
        for item in args.fix_fail.split(','):
            try:
                k, v = item.split('=')
                fixed_params_fail[k.strip()] = float(v.strip())
            except:
                print(f"Warning: Ignoring malformed parameter '{item}'")
    
    # Perform combined fit
    results_pass = fit_function(args.type, hist_pass, fixed_params_pass, interactive=args.interactive, args_bin=args.bin, args_data=args.data)

    results_fail = fit_function(args.type, hist_fail, fixed_params_fail, interactive=args.interactive, args_bin=args.bin, args_data=args.data)
    
    results_pass["bin"] = args.bin 
    results_fail["bin"] = args.bin 
    
    # Plot results
    plot_pass = plot_combined_fit(results_pass, plot_dir, args.data, fixed_params_pass, pf="Pass")
    plot_fail = plot_combined_fit(results_fail, plot_dir, args.data, fixed_params_fail, pf="Fail")

    # Efficiency

    params_pass = results_pass["popt"]
    params_fail = results_fail["popt"]

    N_p = params_pass['N_p']
    N_f = params_fail['N_p']

    N_p_pos = results_pass["minos_errors"]["N_p"]["upper"]
    N_p_neg = results_pass["minos_errors"]["N_p"]["lower"]
    N_f_pos = results_fail["minos_errors"]["N_p"]["upper"]
    N_f_neg = results_fail["minos_errors"]["N_p"]["lower"]

    N_tot_pos_err = np.sqrt(N_p_pos**2 + N_f_pos*2)
    N_tot_neg_err = np.sqrt(N_p_neg**2 + N_f_neg*2)


    N_tot = results_pass["popt"]["N_p"] + results_fail["popt"]["N_p"]

    N_tot_err = np.sqrt(results_pass["perr"]["N_p"]**2 + results_fail["perr"]["N_p"]**2)

    quad_err_plus = np.sqrt( (N_p_pos / N_p)**2 + (N_tot_pos_err / N_tot)**2 )
    quad_err_minus = np.sqrt( (N_p_neg / N_p)**2 + (N_tot_neg_err / N_tot)**2 )
    quad_symmetric_err = np.sqrt( (results_pass["perr"]["N_p"]**2 / results_pass["popt"]["N_p"]**2) + (N_tot_err / N_tot)**2 )
    
    Npass, Nfail = params_pass['N_p'], params_fail['N_p']
    eff = Npass / (Npass + Nfail)
    binom_err = np.sqrt(eff * (1-eff) / (Npass + Nfail))

    root_file.close()

    return plot_pass, plot_fail, eff, quad_symmetric_err, quad_err_plus, quad_err_minus, binom_err, results_pass, results_fail

if __name__ == "__main__":
    main()
