# tools.py

import os
import colors

# creates directory if it does not exist
def makeDir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# get tree from open file
# WARNING: Do not open TFile in getTree(); if you do, the returned TTree object will be destroyed when the TFile closes.
#          Pass open TFile to getTree().
def getTree(open_file, tree_name):
    tree     = open_file.Get(tree_name)
    n_events = tree.GetEntries()
    print("tree: {0}, number of events: {1}".format(tree_name, n_events))
    return tree

def loadSignal(chain):
    files = [
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/0368FEAB-EE24-0840-82E4-6BD2ABF65BB6.root", 
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/1D0E6C19-12F9-3847-BE2F-DB0AE2A7B45D.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/44958EBA-C4CF-A342-B6B5-2A56AC0F1269.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/47B02E1A-D3AA-0841-868E-4D5491047756.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/490307DA-662D-CC44-85D1-E8A7B2DA4E41.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/4C6F3E1E-3A1D-EF45-946F-EF0A1FA6B83D.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/557E1020-0605-7948-979D-4B79FBFC676F.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/647806C4-F1F7-FF40-94E5-4CCFF8E3AFA6.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/64825D01-62C2-8943-A16D-3B9D92F67938.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/683A9572-4950-134A-AA73-B5538B4141FE.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/7F1BA7F2-0FB1-2046-8646-31F166C820B1.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/B3A08A67-DE79-3942-AC9F-03E7E6C5551A.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/BF5485EA-35FB-A146-9CB9-6A6A06C1A640.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/C14F3D2B-8765-9B4A-BBEA-446E768F8994.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/E27B793C-B911-D74D-9FD0-E69009308DCA.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/E65EF1A6-5ECC-154B-ACF9-3C9AB3767BAF.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/ED576849-C52A-2C44-B520-D20A1B8C1CFB.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/F6DE41EC-1578-F748-8199-8D8809CFCA63.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/FB6B7E47-AA87-8F4A-A490-4D6279A61029.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/FC9E2DB3-394C-6945-9A90-9A4A35CB20F1.root",
    ]
    for f in files:
        chain.Add(f)
    n_events = chain.GetEntries()
    print("loadSignal(): number of events = {0}".format(n_events))

def loadBackground(chain):
    files = [
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/040206C8-D83C-BA42-BE3F-C3CB0920BEED.root", 
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/0C483F40-C7FA-0445-B1A0-0C90E2450527.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/12D57CAF-00C5-484F-B744-5BA50CDC7540.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/137AD65A-3E27-3240-B4D7-15D63624B703.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/157E80A8-87A7-A648-84C3-DD941BFA1D7C.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/1961FF79-AA32-D94F-BD7F-303E6ED6131F.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/1A825C77-7998-F445-BAC6-F5ED8F2347D2.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/1B7BE261-DE25-1842-8349-646B3D9D3EA1.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/1DA1E53F-DCA9-BE49-BAE2-7AA2D6FDA7B4.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/238B3428-FA33-9143-8D9E-01AB528645ED.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/251D4E82-949F-E944-8C47-2C8FC3CBEBFA.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/25600B4C-6405-F448-9CD3-91EB3B33A114.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/25C8DD83-C32E-D746-9049-98DAEDA73CF7.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/2C33E603-E710-E648-92D6-506149EF2EF9.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/2FB29631-6B68-C640-A68C-CB15E697CD5A.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/342DA03F-4FBC-2E42-B44D-880B6818075C.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/36D0E49B-1FA1-854D-92D6-058CA66931D7.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/466940CB-5EBC-B34C-B650-940FBF469D07.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/4D5A89C1-1F3E-1E4F-B932-3C73F032140E.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/4D6FC96D-C6C4-6346-85CE-7122CF84FBA8.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/5457F199-A129-2A40-8127-733D51A9A3E6.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/5AB854B1-1DB6-174F-8B0A-E4725A80994C.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/5EA5D32D-1ECE-564B-9EC5-C003828D9145.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/6481B941-20E1-EC49-A059-E986B3C09DEB.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/655344C6-05B2-5D46-A608-F589CC600881.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/65794807-7D39-5A4B-9907-4E378824A488.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/6689D674-3E2E-F741-8F27-62D6FBF41EF9.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/6AF3BAC7-B767-044A-8801-4A81E3A80BF4.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/75CB6752-05D5-5747-8F6B-3B368459B701.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/76D33496-D03B-7D4B-A34E-A8DC728B912D.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/7A72DEE9-DF82-2F4B-9D7F-1BBFA2C80696.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/7C86E4E9-D1B8-E74D-B833-707229FD0558.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/7DC5D661-420D-994E-8CE8-4F1F4D2CFDBB.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/838E6251-D257-E54A-B82C-0BD681975B86.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/87EE7BE7-2EF3-6E46-974F-FE76BC6DA5BC.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/881206CE-DCE1-964C-B168-6EE491F29AC3.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/8D296C2D-E556-3E4B-A5CB-F3035351C313.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/8FC89895-4717-BE46-AB30-9E862A7F0EB4.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/9238E966-B452-3143-A0FB-9363665BF587.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/935FB2F0-EE0B-F445-8A64-5EEFD277BF20.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/937EED08-8A6E-9048-90C0-139F4907B43D.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/9A8C30C2-5E17-1040-A4E4-5D7C8BB0081F.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/A1F0A088-0931-3140-9B33-2B4FB99F8BBD.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/B188292A-75B9-4440-9CD8-98D416426499.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/B9037E7F-9158-114E-B725-6162182415A4.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/BD0AC8FE-F3C7-9846-B5E5-DFE00F4BBBD3.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/BDB9B4D0-4BEA-8447-AEA6-6953F652F285.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/C2576185-9A33-954A-AE97-9D2E48056296.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/C3C566A1-5689-854B-8B52-2A00C841677D.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/CD457AC4-ADD4-864A-82F6-B801DC396174.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/D1473181-6977-A546-8E44-1C6585E8543E.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/D17A9ED0-B887-8F4B-9480-28F00D53CE44.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/D1BF1E97-5B7B-5E42-A0DD-1DEB1493302E.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/D282A789-EED0-1641-A5AB-FB871A2C4E39.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/D3AFBB13-C368-6C49-8A55-E8AD5EEEC7AE.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/D9B4F874-366A-A347-A0C3-D4BFE122A7A4.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/DC7407F2-32E3-7546-BCA1-F3E45C0AA8DF.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/DF68DFBA-FC52-7248-8837-1073A83553A7.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/EC0743F4-AC25-BC40-AE11-ACA537848A32.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/ECC7C9BE-62C3-7B47-8A8F-5751BF69C795.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/F3DB7513-9011-3D42-BA38-7CFB4BB58CD1.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/F65C097C-C70B-5F40-B5D1-56D230DB64C5.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/F6D00887-22E0-CC42-A669-1081C3AD1FA0.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/F866BE75-2905-8E43-AB5F-FCE1F9FE0D8E.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/F91CF867-79D3-DF4A-B1A6-0D3CE6FE050C.root",
        "root://cmseos.fnal.gov//store/user/lpcsusylep/NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_UL2017_NanoAODv9/FB69BFDF-ADD5-8944-AA33-681E611906DC.root",
    ]
    for f in files:
        chain.Add(f)
    n_events = chain.GetEntries()
    print("loadBackground(): number of events = {0}".format(n_events))

def setupHist(hist, title, x_title, y_title, y_min, y_max, color, lineWidth, stats):
    # get x, y axis
    x_axis = hist.GetXaxis()
    y_axis = hist.GetYaxis()
    # set attributes
    hist.SetTitle(title)
    x_axis.SetTitle(x_title)
    y_axis.SetTitle(y_title)
    y_axis.SetRangeUser(y_min, y_max)
    hist.SetLineColor(colors.getColorIndex(color))
    hist.SetLineWidth(lineWidth)
    hist.SetStats(stats)

def setupLegend(legend):
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    legend.SetLineWidth(0)
    legend.SetNColumns(1)
    legend.SetTextFont(42)

