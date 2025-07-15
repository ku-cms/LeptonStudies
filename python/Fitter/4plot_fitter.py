from fitter import *
# Ensure that the fitter.py file is in the same directory as this script!!!

from openpyxl import load_workbook
import pandas as pd
import os
from openpyxl.utils.dataframe import dataframe_to_rows

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

def data_path(args_data=None, args_bin=None, args_type=None, interactive=False):
    root_file_DATA = uproot.open(file_paths[args_data])

    bin_id = args_bin
    bin_suffix, _ = BINS_INFO[bin_id]

    args_bin = bin_id
    args_bin_suffix = bin_suffix
    args_type = args_type
    args_data = args_data

    hist_name_pass = f"{bin_id}_{bin_suffix}_Pass"
    hist_name_fail = f"{bin_id}_{bin_suffix}_Fail"

    hist_pass = load_histogram(root_file_DATA, hist_name_pass, args_data)
    hist_fail = load_histogram(root_file_DATA, hist_name_fail, args_data)

    if hist_pass and hist_fail:
        fixed_params = {}
        fit_data = fit_function(args_type, hist_pass, hist_fail, fixed_params, use_cdf=False, interactive=interactive, args_bin=args_bin, args_data=args_data)
        if fit_data['message'].is_valid:
            print(f"Fit successful for bin '{args_data_data}'.")
        else:
            print(f"Fit failed for bin '{args_data_data}'.")
            fit_data = fit_function(args_type, hist_pass, hist_fail, fixed_params, use_cdf=False, interactive=True, args_bin=args_bin, args_data=args_data)
        fit_data["bin"] = args_bin
    else:
        print(f"Could not load histograms for bin '{args_data_data}'. Exiting.")

    plot_dir = f"{args_bin}_fits/{'DATA' if args_data.startswith('DATA') else 'MC'}/{args_type}/"

    plot_pass, plot_fail = plot_combined_fit(fit_data, plot_dir, args_data)

    return plot_pass, plot_fail, fit_data['popt']['epsilon'], fit_data['perr']['epsilon'], fit_data['message'], fit_data

def MC_path(args_data=None, args_bin=None, args_type=None, interactive=False):
    root_file_DATA = uproot.open(file_paths[args_data])

    bin_id = args_bin
    bin_suffix, _ = BINS_INFO[bin_id]

    args_bin = bin_id
    args_bin_suffix = bin_suffix
    args_type = args_type
    args_data = args_data

    hist_name_pass = f"{bin_id}_{bin_suffix}_Pass"
    hist_name_fail = f"{bin_id}_{bin_suffix}_Fail"

    hist_pass = load_histogram(root_file_DATA, hist_name_pass, args_data)
    hist_fail = load_histogram(root_file_DATA, hist_name_fail, args_data)

    if hist_pass and hist_fail:
        fixed_params = {}
        fit_data = fit_function(args_type, hist_pass, hist_fail, fixed_params, use_cdf=False, interactive=interactive, args_bin=args_bin, args_data=args_data)
        if fit_data['message'].is_valid:
            print(f"Fit successful for bin '{args_data_MC}'.")
        else:
            print(f"Fit failed for bin '{args_data_MC}'.")
            fit_data = fit_function(args_type, hist_pass, hist_fail, fixed_params, use_cdf=False, interactive=True, args_bin=args_bin, args_data=args_data)
        fit_data["bin"] = args_bin
    else:
        print(f"Could not load histograms for bin '{args_data_MC}'. Exiting.")

    plot_dir = f"{args_bin}_fits/{'DATA' if args_data.startswith('DATA') else 'MC'}/{args_type}/"

    plot_pass, plot_fail = plot_combined_fit(fit_data, plot_dir, args_data)

    return plot_pass, plot_fail, fit_data['popt']['epsilon'], fit_data['perr']['epsilon'], fit_data['message'], fit_data

def four_plot_auto(plot_pass_data, plot_fail_data, plot_pass_MC, plot_fail_MC):
    def fig_to_array(fig):
        """Convert a Matplotlib figure to a NumPy array (RGBA)."""
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        # Convert ARGB to RGBA
        buf = buf[:, :, [1,2,3,0]]
        return buf

    # Convert your figs to numpy arrays
    img_pass_data = fig_to_array(plot_pass_data)
    img_fail_data = fig_to_array(plot_fail_data)
    img_pass_MC = fig_to_array(plot_pass_MC)
    img_fail_MC = fig_to_array(plot_fail_MC)

    # Close the individual figures immediately after conversion
    plt.close(plot_pass_data)
    plt.close(plot_fail_data)
    plt.close(plot_pass_MC)
    plt.close(plot_fail_MC)
    plt.close('all')

    # Create combined figure
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    images = [img_pass_data, img_fail_data, img_pass_MC, img_fail_MC]
    grid_order = [(0,0), (1,0), (0,1), (1,1)]

    for img, (row, col) in zip(images, grid_order):
        axs[row, col].imshow(img)
        axs[row, col].axis('off')

    scale_factor = epsilon_data / epsilon_MC
    scale_factor_err = np.sqrt(((epsilon_err_data / epsilon_data)**2 + 
                              (epsilon_err_MC / epsilon_MC)**2)) * scale_factor
    
    fig.text(0.5, 0.95, f"Scale Factor = {scale_factor:.6f} ± {scale_factor_err:.6f}", 
             ha="center", va="center", fontsize=15,
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"))
    
    fig.text(0.5, 0.90, 
             f"Data: {epsilon_data:.6f} ± {epsilon_err_data:.6f} | MC: {epsilon_MC:.6f} ± {epsilon_err_MC:.6f}\nbin: {args_bin}",
             ha="center", va="center", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"))

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0, top=0.85)
    os.makedirs("final_gold_blp_4plots", exist_ok=True)
    plt.savefig(f"final_gold_blp_4plots/{save_fig}.png", dpi=150, bbox_inches="tight")
    plt.close('all')

    return scale_factor, scale_factor_err

def save_to_excel(results_data, results_mc, epsilon_data, epsilon_err_data,
                 epsilon_mc, epsilon_err_mc, scale_factor, scale_factor_err,
                 args_bin, barrel, args_type, type, num_den, filename="scale_factors_basic.xlsx"):
    """
    Save fitting results with all DATA entries first, then all MC entries.
    Updates existing entries in their original positions if they match the same bin, barrel, and type.
    """
    # Create the new entries
    new_data_row = {
        "Type": "DATA",
        "bin": args_bin,
        "barrel": barrel,
        "num_den": num_den,
        "fit_type": args_type,
        "epsilon": epsilon_data,
        "epsilon_err": epsilon_err_data,
        "SF": scale_factor,
        "SF_err": scale_factor_err
    }
    
    new_mc_row = {
        "Type": "MC",
        "bin": args_bin,
        "barrel": barrel,
        "num_den": num_den,
        "fit_type": args_type,
        "epsilon": epsilon_mc,
        "epsilon_err": epsilon_err_mc,
        "SF": None,
        "SF_err": None
    }

    if os.path.exists(filename):
        existing_df = pd.read_excel(filename, sheet_name='ScaleFactors')
        
        # Split into DATA and MC parts
        data_df = existing_df[existing_df['Type'] == 'DATA']
        mc_df = existing_df[existing_df['Type'] == 'MC']
        
        data_idx = data_df[(data_df['bin'] == args_bin) & 
                         (data_df['barrel'] == barrel)].index
        
        mc_idx = mc_df[(mc_df['bin'] == args_bin) & 
                     (mc_df['barrel'] == barrel)].index
        
        # Update DATA
        if not data_idx.empty:
            data_df.loc[data_idx[0]] = new_data_row
        else:
            data_df = pd.concat([data_df, pd.DataFrame([new_data_row])], ignore_index=True)
            
        # Update MC
        if not mc_idx.empty:
            mc_df.loc[mc_idx[0]] = new_mc_row
        else:
            mc_df = pd.concat([mc_df, pd.DataFrame([new_mc_row])], ignore_index=True)
            
        # Combine with DATA then MC
        updated_df = pd.concat([data_df, mc_df], ignore_index=True)
    else:
        # Create new file with DATA then MC
        updated_df = pd.DataFrame([new_data_row, new_mc_row])

    updated_df.to_excel(filename, sheet_name='ScaleFactors', index=False)
    print(f"Results saved/updated in {filename}")

# UPDATE HERE
barrel = "barrel_1"
args_bin = "bin5"

num_den = "gold_blp"
args_data_data = f"DATA_NEW_2_{barrel}_gold_blp"
args_data_MC = f"MC_DY2_2L_2J_{barrel}_gold_blp"
interactive = True

type = "Nominal"

if type == "Nominal":
    if args_bin =="bin2":
        args_type = "cbg_cms"
    elif args_bin =="bin3" or args_bin =="bin4":
        args_type = "dcb_cms"
    elif args_bin =="bin5" or args_bin =="bin6":
        args_type = "g_cms"
elif type == "Alternative 1":
    if args_bin =="bin2":
        args_type = "g_bpoly"
    elif args_bin =="bin3" or args_bin =="bin4":
        args_type = "dcb_bpoly"
    elif args_bin =="bin5" or args_bin =="bin6":
        args_type = "g_bpoly"
elif type == "Alternative 2":
    if args_bin =="bin2":
        args_type = "cbg_bpoly"
    elif args_bin =="bin3":
        args_type = "dv_cms"
    elif args_bin =="bin4":
        args_type = "cbg_bpoly"
    elif args_bin =="bin5":
        args_type = "dv_bpoly"
    elif args_bin =="bin6":
        args_type = "dv_bpoly"

print(args_type)
save_fig = f"{args_bin}_{args_type}_{barrel}_{num_den}_4plot"

plot_pass_data, plot_fail_data, epsilon_data, epsilon_err_data, message_data, results_data = data_path(args_data_data, args_bin, args_type, interactive=interactive)
plot_pass_MC, plot_fail_MC, epsilon_MC, epsilon_err_MC, message_MC, results_MC = MC_path(args_data_MC, args_bin, args_type,interactive=interactive)

plt.close('all')

scale_factor, scale_factor_err = four_plot_auto(plot_pass_data, plot_fail_data, plot_pass_MC, plot_fail_MC)

# Save results to Excel
save_to_excel(results_data, results_MC, epsilon_data, epsilon_err_data, 
             epsilon_MC, epsilon_err_MC, scale_factor, scale_factor_err,
             args_bin, barrel, args_type, type, num_den, filename="final_gold_blp_SF_heatmap.xlsx")
