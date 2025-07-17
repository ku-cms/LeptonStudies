This fitter uses iminuit and is best run in terminal.

In the code, you can adjust the type of minimizer to use. m.simplex(), and m.Migrad() are both on by default.

Example: "python Fitter_SH.py --mass Z --data DATA_barrel_1 --type dcb_cms --interactive --bin bin3"

Use arguments to specify what file you are fitting and how you fit.

      -- mass: determines what mass is being fit (Z, Z_muon, JPsi, JPsi_muon).
      
      -- bin: specifies what bin is being used. (i.e. bin0, bin1, etc).
      
      -- data: determines what data type is being used (i.e. DATA_barrel_1, MC_DY_endcap, etc).
      
      -- type: defines the signal and background to use. (i.e. dcb_lin, g_exp, etc).
      
            - Signals available: double crystal ball (dcb), gaussian (g), crystal ball x gaussian (cbg), double voigtian (dv).
            - Backgrounds available:: linear (lin), exponential (exp), phase space (ps), chebyshev polynomial (cheb), Bernstein polynomial (bpoly), CMS shape (cms).
            
      -- interactive: If used, adds a visual fitter that is interactive. (must have PySide6 installed).
      
      -- fix: fixes defined parameter to the value set. (i.e. “mu = 90, sigma = 2.5, etc”).
      
      -- cdf: Use cdf for signal and background shapes, will fall back to pdf if either function doesn’t have a cdf available.

      --sigmoid: unbounds efficiency and transforms it into the correct range with a sigmoid transformation.
