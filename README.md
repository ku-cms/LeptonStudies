# LeptonStudies

Study low transverse momenta leptons for CMS compressed SUSY searches.

## Resources

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

## Copying datasets

In the scripts directory, there are two scripts that can be use to copy CMS datasets.

Copying a CMS dataset with these scripts is done in two steps:
1. Make a new file list using `make_file_list.sh` 
2. Copy this file list using `copy_file_list.sh`

First, initiate your CERN grid certificate:
```
voms-proxy-init --valid 192:00 -voms cms
voms-proxy-info
```

Then, make a new file list using this syntax: 
```
cd scripts
./make_file_list.sh dataset_name output_file_name
```

Example for one EGamma 2023C Data (NANOAOD) dataset:
```
cd scripts
./make_file_list.sh /EGamma0/Run2023C-22Sep2023_v1-v1/NANOAOD EGamma0_Run2023C-22Sep2023_v1-v1_NANOAOD.txt
```

Finally, copy this file list to a directory of your choice.

If the directory is not on EOS, you can uncomment the `mkdir` command in `copy_file_list.sh`.

If the directory is on EOS, make sure the `mkdir` command is commented out in `copy_file_list.sh`.

Then, create the directory using eosmkdir.
```
eosmkdir -p <directory>
```

Example for one EGamma 2023C Data (NANOAOD) dataset (replace `username` with your username):
```
eosmkdir -p /store/user/username/datasets/2023_Data/EGamma0_Run2023C-22Sep2023_v1-v1_NANOAOD
```

Use this syntax to run interactively:
```
./copy_file_list.sh file_list.txt output_dir
```

Use this syntax to run using nohup (useful when it takes a long time): 
```
nohup ./copy_file_list.sh file_list.txt output_dir > copy_list_001.log 2>&1 & 
```

Example for one EGamma 2023C Data (NANOAOD) dataset (replace `username` with your username):
```
nohup ./copy_file_list.sh EGamma0_Run2023C-22Sep2023_v1-v1_NANOAOD.txt root://cmseos.fnal.gov//store/user/username/datasets/2023_Data/EGamma0_Run2023C-22Sep2023_v1-v1_NANOAOD > copy_list_001.log 2>&1 &
```

This should create a job running in the background.
Since we are using nohup, the job should continue running even if you logout.
You can check the job status using these commands.
```
jobs -l
ps aux | grep copy_file_list.sh
tail -f copy_list_001.log -n 1000
```

## Examples

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

## Additional Info

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

