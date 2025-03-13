import uproot
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.pyplot as plt
import mplhep as hep
import basic_tools

def load_histogram(root_file, hist_name):
    keys = {key.split(";")[0]: key for key in root_file.keys()}
    if hist_name in keys:
        obj = root_file[keys[hist_name]]
        if isinstance(obj, uproot.behaviors.TH1.Histogram):
            values, edges = obj.to_numpy()
            return {"values": values, "edges": edges}
    return None

def fit_hist(hist, hist_name, plot_dir):
    print(f"Fitting histogram '{hist_name}' ...")

    basic_tools.makeDir(plot_dir)

    plt.figure(figsize=(12, 8))
    hep.style.use("CMS")

    centers = (hist["edges"][:-1] + hist["edges"][1:]) / 2
    values = hist["values"]
    errors = np.sqrt(values) 

    # Plot data points with error bars
    plt.errorbar(
        centers,
        values,
        yerr=errors,
        fmt='o',
        label=f"Data Points ({hist_name})",
        capsize=3
    )

    # Fit a Gaussian + Exponential background model
    p0 = [max(values), 90, 10, min(values), 20]  # Initial guess: [A, mu, sigma, B, tau]
    popt, pcov = curve_fit(gaussian_exponential, centers, values, p0=p0)
    perr = np.sqrt(np.diag(pcov))  # uncertainties

    x = np.linspace(min(centers), max(centers), 100)
    gauss   = popt[0] * norm.pdf(x, popt[1], popt[2])
    expo    = popt[3] * np.exp(-x / popt[4])
    combined = gaussian_exponential(x, *popt)

    # Compute the integrated signal and background events
    signal_events, background_events = compute_signal_background_events(x, popt)
    print(f" - Estimated number of signal events: {signal_events:.2f}")
    print(f" - Estimated number of background events: {background_events:.2f}")

    # Plot the fitted components and combined fit
    plt.plot(x, gauss,      label="Gaussian (Signal)")
    plt.plot(x, expo,       label="Exponential (Background)")
    plt.plot(x, combined,   label="Combined Fit")
    
    # Display the fit parameters and event counts
    legend_text = (
        f"Gaussian (Signal): A={popt[0]:.2f}, μ={popt[1]:.2f}, σ={popt[2]:.2f}\n"
        f"Exponential (Background): B={popt[3]:.2f}, τ={popt[4]:.2f}\n"
        f"Signal Events: {signal_events:.2f}\n"
        f"Background Events: {background_events:.2f}"
    )
    
    # Create legend
    leg = plt.legend(loc='upper right', fontsize=8, frameon=True)
    leg.get_frame().set_linewidth(1)
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_alpha(1)
    
    # Add a text box in the upper left with the detailed information
    plt.text(
        0.05,
        0.95,
        legend_text,
        transform=plt.gca().transAxes,
        fontsize=8,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8)
    )
    
    # Set the plot title and label axes 
    title   = f"Fit to {hist_name}"
    xlabel  = r"$m_{ee}$ [GeV]"
    ylabel  = "Number of events"
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Save plot
    plot_name = f"fit_{hist_name}"
    basic_tools.savePlot(plot_dir, plot_name)
    
    plt.close()
    
    print("Finished fit!")

def gaussian_exponential(x, A, mu, sigma, B, tau):
    return A * norm.pdf(x, mu, sigma) + B * np.exp(-x / tau)

def compute_signal_background_events(x, popt):
    """
    Compute integrated signal and background events using the fitted parameters.
    """
    # Compute the Gaussian (signal) and  exponential (background) 
    gauss   = popt[0] * norm.pdf(x, popt[1], popt[2])
    expo    = popt[3] * np.exp(-x / popt[4])
    # Integrate each component
    signal_events       = np.trapezoid(gauss, x)
    background_events   = np.trapezoid(expo, x)
    return signal_events, background_events

def main():
    # Specify the histogram name and directory to save plots
    hist_name   = "bin11_pt_32p00To34p00_Pass"
    plot_dir    = "egamma_data_2023C_fits"
    
    # Open ROOT file
    #root_file = uproot.open("DY_2023C_pt_barrel.root")
    root_file = uproot.open("egamma_tnp_output/egamma_output_2025_02_03_run_1/HLT_Ele30_WPTight_Gsf_histos_pt_barrel.root")
    
    # Load the histogram using the opened file and specified name
    hist = load_histogram(root_file, hist_name)
    
    # Call the function to generate and save the plot, also assigning the name to match the histogram name
    fit_hist(hist, hist_name, plot_dir)

if __name__ == "__main__":
    main()

