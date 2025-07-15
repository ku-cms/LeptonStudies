import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from inspect import signature
from scipy.special import erf, voigt_profile
from scipy.stats import norm
from scipy.interpolate import BPoly
from numpy.polynomial.chebyshev import Chebyshev
from numba_stats import cmsshape

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
    # Normalize
    return result / np.trapezoid(result, x)

def gaussian(x, mu, sigma):
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
        try:
            logN = n * np.log(n / abs_alpha) - 0.5 * abs_alpha**2
            N = np.exp(logN)
        except FloatingPointError:
            N = 1e300  # fallback large number

        base = (n / abs_alpha - abs_alpha - z[mask_tail]) if (alpha < 0) else (n / abs_alpha - abs_alpha + z[mask_tail])
        base = np.clip(base, 1e-15, np.inf)
        result[mask_tail] = N * base**(-n)
        return result

    y_cb_un = crystal_ball_unnormalized(x, mu, sigma, alpha, n)
    integral = np.trapezoid(y_cb_un, x)
    y_cb = y_cb_un / integral
    y_gauss = norm.pdf(x, loc=mu, scale=sigma2)
    y_total = y_cb + y_gauss
    normalization = np.trapezoid(y_total, x)
    if normalization <= 0 or np.isnan(normalization) or np.isinf(normalization):
        return np.zeros_like(y_total)
    return y_total / normalization

def phase_space(x, a, b, x_min=None, x_max=None):
    if x_min is None:
        x_min = np.min(x)
    if x_max is None:
        x_max = np.max(x)
    
    a_clamped = np.clip(a, 0, 250)
    b_clamped = np.clip(b, 0, 250)

    t1 = np.clip(x - x_min, 1e-8, None)
    t2 = np.clip(x_max - x, 1e-8, None)

    log_pdf = a_clamped * np.log(t1) + b_clamped * np.log(t2)
    pdf = np.exp(log_pdf - np.max(log_pdf))
    pdf[(x <= x_min) | (x >= x_max)] = 0

    norm = np.trapezoid(pdf, x)
    return pdf / (norm if norm>0 else 1e-8)

def linear(x, b, C, x_min=80, x_max=100):
    if x_min is None:
        x_min = np.min(x)
    if x_max is None:
        x_max = np.max(x)
    
    x_mid = 0.5 * (x_min + x_max)
    lin = b * (x - x_mid) + C
    lin = np.clip(lin, 0, None)
    denom = np.trapezoid(lin, x)
    return lin / denom

def exponential(x, C, x_min=None, x_max=None):
    if x_min is None:
        x_min = np.min(x)
    if x_max is None:
        x_max = np.max(x)
    
    z = -C * x
    z_max = np.max(z)
    exp_z = np.exp(z - z_max)
    log_norm = z_max + np.log(np.trapezoid(exp_z, x))
    norm = np.exp(log_norm)

    if not np.isfinite(norm) or norm <= 0:
        return np.zeros_like(x)
    return np.exp(z) / norm

def chebyshev_background(x, *coeffs, x_min=None, x_max=None):
    if x_min is None:
        x_min = np.min(x)
    if x_max is None:
        x_max = np.max(x)
    
    x_norm = 2*(x-x_min)/(x_max-x_min) - 1
    return Chebyshev(coeffs)(x_norm) / np.trapezoid(Chebyshev(coeffs)(x_norm), x)

def bernstein_poly(x, *coeffs, x_min=None, x_max=None):
    if x_min is None:
        x_min = np.min(x)
    if x_max is None:
        x_max = np.max(x)
    
    c = np.array(coeffs).reshape(-1, 1)
    return BPoly(c, [x_min, x_max])(x)

def cms(x, beta, gamma, loc):
    y = cmsshape.pdf(x, beta, gamma, loc)
    return y

class ShapeExplorer:
    def __init__(self, shapes, x_range=(80, 100), n_points=1000):
        self.shapes = shapes
        self.current_shape = None
        self.x_range = x_range
        self.n_points = n_points
        
        self.fig = plt.figure(figsize=(14, 8))
        plt.subplots_adjust(bottom=0.35, right=0.75, top=0.925)
        
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(x_range[0], x_range[1])
        self.ax.set_xlabel('x', fontsize=12)
        self.ax.set_ylabel('y', fontsize=12)
        
        self.integral_text = self.ax.text(0.02, 0.95, "", transform=self.ax.transAxes,
                                        fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        
        self.shape_buttons = {}
        button_width = 0.15
        button_height = 0.05
        button_spacing = 0.01
        left_start = 0.80

        
        for i, (shape_name, _) in enumerate(shapes.items()):
            ax = self.fig.add_axes([left_start, 0.9 - i*(button_height + button_spacing), 
                                  button_width, button_height])
            btn = Button(ax, shape_name, color='cornflowerblue')
            btn.on_clicked(self.create_shape_select_func(shape_name))
            self.shape_buttons[shape_name] = btn
        
        self.sliders = {}
        self.reset_button = None
        
        if shapes:
            first_shape = next(iter(shapes.keys()))
            self.select_shape(first_shape)
    
    def select_shape(self, shape_name):
        if shape_name not in self.shapes:
            return
            
        self.current_shape = shape_name
        config = self.shapes[shape_name]
        
        self.fig.suptitle(config.get('title', f"{shape_name} Explorer"), fontsize=16)
        
        for slider in self.sliders.values():
            slider.ax.remove()
        self.sliders.clear()
        
        if self.reset_button:
            self.reset_button.ax.remove()
            self.reset_button = None
        
        func = config['function']
        param_ranges = config.get('param_ranges', {})
        param_steps = config.get('param_steps', {})
        param_names = config.get('param_names', None)
        auto_norm = config.get('auto_norm', True)
        x_range = config.get('x_range', self.x_range)
        
        sig = signature(func)
        self.param_names = list(sig.parameters.keys())[1:]
        
        if param_names is not None:
            if len(param_names) == len(self.param_names):
                self.param_names = param_names
        
        self.param_ranges = {}
        self.param_steps = {}
        for param in self.param_names:
            self.param_ranges[param] = param_ranges.get(param, (80, 100))
            self.param_steps[param] = param_steps.get(param, 0.001)
        
        self.x = np.linspace(x_range[0], x_range[1], self.n_points)
        self.ax.set_xlim(x_range[0], x_range[1])
        
        self.params = {name: 0.5*(self.param_ranges[name][0] + self.param_ranges[name][1]) 
                     for name in self.param_names}
        
        y = self.evaluate(func, auto_norm)
        if hasattr(self, 'line'):
            self.line.set_data(self.x, y)
        else:
            self.line, = self.ax.plot(self.x, y, 'k-', linewidth=2)
        
        y_max = np.max(y) * 1.2 if np.max(y) > 0 else 0.1
        self.ax.set_ylim(0, y_max)
        
        self.update_integral(y)
        
        slider_height = 0.02
        slider_spacing = 0.035
        bottom_start = 0.1
        
        self.param_ranges = {}
        self.param_steps = {}
        self.params = {}

        for param in self.param_names:
            self.param_ranges[param] = param_ranges.get(param, (80, 100, 120))
            self.param_steps[param] = param_steps.get(param, 0.001)
            self.params[param] = self.param_ranges[param][1]

        for i, param in enumerate(self.param_names):
            min_val, init_val, max_val = self.param_ranges[param]
            ax = self.fig.add_axes([0.25, bottom_start + i*slider_spacing, 0.5, slider_height])
            slider = Slider(ax, param, 
                            min_val, max_val,
                            valinit=init_val,
                            valstep=self.param_steps.get(param, 0.001))
            slider.on_changed(self.create_update_func(func, auto_norm))
            self.sliders[param] = slider        
                
        reset_ax = self.fig.add_axes([0.8, 0.05, 0.15, 0.04])
        self.reset_button = Button(reset_ax, 'Reset Parameters', color='cornflowerblue')
        self.reset_button.on_clicked(self.reset)
        
        self.fig.canvas.draw_idle()
    
    def evaluate(self, func, auto_norm):
        from inspect import signature, Parameter

        sig = signature(func)
        param_list = list(sig.parameters.items())

        args = []
        kwargs = {}

        args.append(self.x)

        for (name, param) in param_list[1:]:
            if name in self.params:
                if param.default is param.empty:
                    args.append(self.params[name])
                else:
                    kwargs[name] = self.params[name]

        if 'x_min' in sig.parameters:
            kwargs['x_min'] = self.x_range[0]
        if 'x_max' in sig.parameters:
            kwargs['x_max'] = self.x_range[1]

        y = func(*args, **kwargs)

        if auto_norm:
            integral = np.trapezoid(y, self.x)
            if integral > 0:
                y = y / integral
        return y
    def update_integral(self, y):
        integral = np.trapezoid(y, self.x)
        self.integral_text.set_text(f"Area under curve: {integral:.6f}")
    
    def create_update_func(self, func, auto_norm):
        def update(val):
            for param in self.param_names:
                self.params[param] = self.sliders[param].val
            
            y = self.evaluate(func, auto_norm)
            self.line.set_ydata(y)
            
            y_max = np.max(y) * 1.2 if np.max(y) > 0 else 0.1
            self.ax.set_ylim(0, y_max)
            
            self.update_integral(y)
            self.fig.canvas.draw_idle()
        return update
    
    def create_shape_select_func(self, shape_name):
        def select(event):
            self.select_shape(shape_name)
        return select
    
    def reset(self, event):
        for param, slider in self.sliders.items():
            slider.reset()
    
    def show(self):
        plt.show()

if __name__ == "__main__":
    shapes_config = {
        'Double Crystal Ball': {
            'function': double_crystal_ball,
            'param_ranges': {
                "mu": (88, 90.5, 92),
                "sigma": (1, 3, 6),
                "alphaL": (0, 1.0, 10),
                "nL": (0, 5.0, 30),
                "alphaR": (0, 1.0, 10),
                "nR": (0, 5.0, 30)
            },
            'param_steps': {
                'mu': 0.01,
                'sigma': 0.01,
                'alphaL': 0.01,
                'nL': 0.05,
                'alphaR': 0.01,
                'nR': 0.05
            },
            'x_range': (80, 100),
            'title': "Double Crystal Ball PDF Explorer",
            'auto_norm': True
        },
        'Double Voigtian': {
            'function': double_voigtian,
            'param_ranges': {
                "mu": (88, 90, 93),
                "sigma1": (2.0, 3.0, 4.0),
                "gamma1": (0.01, 0.5, 3.0),
                "sigma2": (1.0, 2.0, 3.0),
                "gamma2": (0.01, 1.0, 3.0)
            },
            'x_range': (80, 100),
            'title': "Double Voigtian Explorer",
            'auto_norm': True
        },
        'Gaussian': {
            'function': gaussian,
            'param_ranges': {
                "mu": (88, 90, 94),
                "sigma": (1, 2.5, 6)
            },
            'x_range': (80, 100),
            'title': "Gaussian Explorer",
            'auto_norm': False
        },
        'Crystal Ball x Gaussian': {
            'function': CB_G,
            'param_ranges': {
                "mu": (88, 90, 92),
                "sigma": (1, 3, 6),
                "alpha": (-10, -1, 10),
                "n": (0.1, 5.0, 30),
                "sigma2": (1, 3, 10)
            },
            'x_range': (80, 100),
            'title': "Crystall Ball x Gaussian Explorer",
            'auto_norm': True
        },
        'Phase Space': {
            'function': phase_space,
            'param_ranges': {
                "a": (0, 0.5, 10),
                "b": (0, 1, 30)
            },
            'x_range': (80, 100),
            'title': "Phase Space Explorer",
            'auto_norm': True
        },
        'Linear': {
            'function': linear,
            'param_ranges': {
                "b": (-1, 0.1, 1),
                "C": (0, 0.1, 10)
            },
            'x_range': (80, 100),
            'title': "Linear Explorer",
            'auto_norm': True
        },
        'Exponential': {
            'function': exponential,
            'param_ranges': {
                "C": (-10, 0.1, 10)
            },
            'x_range': (80, 100),
            'title': "Exponential Explorer",
            'auto_norm': True
        },
        'Chebyshev': {
            'function': lambda x, c0, c1, c2: chebyshev_background(x, c0, c1, c2),
            'param_ranges': {
                "c0": (0.001, 1, 3),
                "c1": (0.001, 1, 3),
                "c2": (0.001, 1, 3)
            },
            'x_range': (80, 100),
            'title': "Chebyshev Explorer",
            'auto_norm': True
        },
        'Bernstein Polynomial': {
            'function': lambda x, c0, c1, c2: bernstein_poly(x, c0, c1, c2),
            'param_ranges': {
                "c0": (0, 0.05, 10),
                "c1": (0, 0.1, 1),
                "c2": (0, 0.1, 1),
            },
            'x_range': (80, 100),
            'title': "Bernstein Polynomial Explorer",
            'auto_norm': True
        },
        'CMS Shape': {
            'function': cms,
            'param_ranges': {
                "beta": (-0.5, 0.1, 1.5),
                "gamma": (0, 0.1, 2),   
                "loc": (-100, 90, 200)     
            },
            'x_range': (-100, 100),
            'title': "CMS Shape Explorer",
            'auto_norm': True
        }
    }

    explorer = ShapeExplorer(shapes_config)
    explorer.show()