pub const MP_NO_ITER: isize = -1;
pub const MP_MACHEP0: f64 = 2.2204460e-16;

/// Definition of a parameter constraint structure
pub struct MPPar {
    pub fixed: bool,
    pub limited_low: bool,
    pub limited_up: bool,
    pub limit_low: f64,
    pub limit_up: f64,
    /// Step size for finite difference
    pub step: f64,
    /// Relative step size for finite difference
    pub rel_step: f64,
    /// Sidedness of finite difference derivative
    pub side: MPSide,
}

impl ::std::default::Default for MPPar {
    fn default() -> Self {
        MPPar {
            fixed: false,
            limited_low: false,
            limited_up: false,
            limit_low: 0.0,
            limit_up: 0.0,
            step: 0.0,
            rel_step: 0.0,
            side: MPSide::Auto,
        }
    }
}

/// Sidedness of finite difference derivative
pub enum MPSide {
    /// one-sided derivative computed automatically
    Auto,
    /// one-sided derivative (f(x+h) - f(x)  )/h
    Right,
    /// one-sided derivative (f(x)   - f(x-h))/h
    Left,
    /// two-sided derivative (f(x+h) - f(x-h))/(2*h)
    Both,
    /// user-computed analytical derivatives
    User,
}

/// Definition of MPFIT configuration structure
pub struct MPConfig {
    /// Relative chi-square convergence criterion  Default: 1e-10
    pub ftol: f64,
    /// Relative parameter convergence criterion   Default: 1e-10
    pub xtol: f64,
    /// Orthogonality convergence criterion        Default: 1e-10
    pub gtol: f64,
    /// Finite derivative step size                Default: MP_MACHEP0
    pub epsfcn: f64,
    /// Initial step bound                         Default: 100.0
    pub step_factor: f64,
    /// Range tolerance for covariance calculation Default: 1e-14
    pub covtol: f64,
    /// Maximum number of iterations.  If maxiter == MP_NO_ITER,
    /// then basic error checking is done, and parameter
    /// errors/covariances are estimated based on input
    /// parameter values, but no fitting iterations are done.
    pub max_iter: isize,
    /// Maximum number of function evaluations, or 0 for no limit
    /// Default: 0 (no limit)
    pub max_fev: usize,
    /// Default: true
    pub n_print: bool,
    /// Scale variables by user values?
    /// true = yes, user scale values in diag;
    /// false = no, variables scaled internally (Default)
    pub do_user_scale: bool,
    /// Disable check for infinite quantities from user?
    /// true = perform check
    /// false = do not perform check (Default)
    pub no_finite_check: bool,
}

impl ::std::default::Default for MPConfig {
    fn default() -> Self {
        MPConfig {
            ftol: 1e-10,
            xtol: 1e-10,
            gtol: 1e-10,
            epsfcn: MP_MACHEP0,
            step_factor: 100.0,
            covtol: 1e-14,
            max_iter: MP_NO_ITER,
            max_fev: 0,
            n_print: true,
            do_user_scale: false,
            no_finite_check: false,
        }
    }
}

pub enum MPStatus {}

/// Definition of results structure, for when fit completes
pub struct MPResult {
    /// Final chi^2
    pub best_norm: f64,
    /// Starting value of chi^2
    pub orig_norm: f64,
    /// Number of iterations
    pub n_iter: usize,
    /// Number of function evaluations
    pub n_fev: usize,
    /// Fitting status code
    pub status: MPStatus,
    /// Total number of parameters
    pub n_par: usize,
    /// Number of free parameters
    pub n_free: usize,
    /// Number of pegged parameters
    pub n_pegged: usize,
    /// Number of residuals (= num. of data points)
    pub n_func: usize,
    /// Final residuals nfunc-vector
    pub resid: Vec<f64>,
    /// Final parameter uncertainties (1-sigma) npar-vector
    pub xerror: Vec<f64>,
    /// Final parameter covariance matrix npar x npar array
    pub covar: Vec<f64>,
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
