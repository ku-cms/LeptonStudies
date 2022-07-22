# plotHists.py

import os
import ROOT
import tools
import numpy as np

# Make sure ROOT.TFile.Open(fileURL) does not seg fault when $ is in sys.argv (e.g. $ passed in as argument)
ROOT.PyConfig.IgnoreCommandLineOptions = True
# Make plots faster without displaying them
ROOT.gROOT.SetBatch(ROOT.kTRUE)
# Tell ROOT not to be in charge of memory, fix issue of histograms being deleted when ROOT file is closed:
ROOT.TH1.AddDirectory(False)
# Stat box    
ROOT.gStyle.SetOptStat(111111)

# Plot with multiple GenPartFlav values
def plotMultiGenPartFlav(hists, genPartFlavs, sample, variable, plot_dir):
    # canvas
    c = ROOT.TCanvas("c", "c", 800, 800)
    c.SetLeftMargin(0.15)
    
    for genPartFlav in genPartFlavs:
        hist = hists[genPartFlav]
        hist.Draw("hist error same")
    
    # save plot
    output_name = "{0}/{1}_{2}".format(plot_dir, sample, variable)
    c.Update()
    c.SaveAs(output_name + ".pdf")

def run(sample, input_file, plot_dir):
    f = ROOT.TFile(input_file)
    variables = [
        "dxy",
        "dz",
        "ID",
        "embeddedID",
    ]
    genPartFlavs = ["genPartFlav0", "genPartFlav1", "genPartFlav5"]
    for variable in variables:
        hists = {}
        h_name_base = "h_LowPtElectron_{0}".format(variable)
        # Load hists
        for genPartFlav in genPartFlavs:
            h_name = "{0}_{1}".format(h_name_base, genPartFlav)
            hists[genPartFlav] = f.Get(h_name)
        # Plot
        plotMultiGenPartFlav(hists, genPartFlavs, sample, variable, plot_dir)

# create plots from hists saved in ROOT files
def plotHists():
    plot_dir    = "hist_plots"
    samples = {
        "SMS-T2-4bd_genMET-80_mStop-500_mLSP-490"   : "src/output/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_v1.root",
        "TTJets_DiLept"                             : "src/output/TTJets_DiLept_v1.root",
    }
    
    tools.makeDir(plot_dir)
    
    for sample in samples:
        print("Running {0}".format(sample))
        run(sample, samples[sample], plot_dir)

def main():
    plotHists()

if __name__ == "__main__":
    main()


