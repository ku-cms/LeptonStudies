# LeptonStudies

Study low transverse momenta leptons for CMS compressed SUSY searches.
- Repository: https://github.com/ku-cms/LeptonStudies
- CERN SWAN platform: https://swan.cern.ch/
- CERN SWAN info: https://swan.web.cern.ch/swan/
- CERN SWAN docs: https://swan.docs.cern.ch/
- CERN Box: https://cernbox.cern.ch/
- CERN Box docs: https://cernbox.docs.cern.ch/
- CMS Data Aggregation System (DAS): https://cmsweb.cern.ch/das/ 
- NANO AOD documentation page: https://cms-nanoaod-integration.web.cern.ch/autoDoc/
- coffea: https://github.com/CoffeaTeam/coffea
- coffea docs: https://coffea-hep.readthedocs.io/

## Setup

Setup commands to run at cmslpc or lxplus.

Checkout CMSSW_10_6_5 in your desired working area.
```
cmsrel CMSSW_10_6_5
cd CMSSW_10_6_5/src
cmsenv
```

Download this repository in the CMSSW_10_6_5/src directory.
```
git clone https://github.com/ku-cms/LeptonStudies.git
cd LeptonStudies
```

Create plots with python plotting script.
```
python python/makePlots.py
```

Create plots with the ROOT macro.
```
cd src
mkdir -p macro_plots
# start a ROOT shell:
root -l
# enter the following commands in the ROOT shell:
.L NanoClass.C
NanoClass n;
n.Loop();
```

Documentation for using ROOT Trees:

https://root.cern.ch/root/htmldoc/guides/users-guide/Trees.html#simple-analysis-using-ttreedraw

Create plots with ROOT commands:
```
root <file_path>

MyTree->Draw(<variables>, <cuts>, <options>)
```

Example for a NANO AOD v9 TTJets ROOT file, creating 1D plots:
```
root <file_path>

Events->Draw("LowPtElectron_pt", "LowPtElectron_pt<10 && LowPtElectron_genPartFlav==1")

Events->Draw("LowPtElectron_embeddedID", "LowPtElectron_pt<10 && LowPtElectron_genPartFlav==1")
```

Example for a NANO AOD v9 TTJets ROOT file, creating 2D plots:
```
root <file_path>

Events->Draw("LowPtElectron_embeddedID:LowPtElectron_pt", "LowPtElectron_pt<10 && LowPtElectron_genPartFlav==1", "colz")

Events->Draw("LowPtElectron_embeddedID:LowPtElectron_pt", "LowPtElectron_pt<10 && LowPtElectron_genPartFlav==0", "colz")
```

FNAL LPC EOS redirector:
```
root://cmseos.fnal.gov/
```

Main FNAL redirector (searches all sites):
```
root://cmsxrootd.fnal.gov/
```

Global CERN redirector (searches all sites):
```
root://cms-xrd-global.cern.ch/
```
