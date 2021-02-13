pub const MP_NO_ITER: usize = 0;
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
    pub max_iter: usize,
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

/// MP Fit errors
pub enum MPError {
    /// General input parameter error
    Input,
    /// User function produced non-finite values
    Nan,
    /// No user data points were supplied
    Empty,
    /// No free parameters
    NoFree,
    /// Initial values inconsistent with constraints
    InitBounds,
    /// Initial constraints inconsistent
    Bounds,
    /// Not enough degrees of freedom
    DoF,
}

/// Potential success status
pub enum MPSuccess {
    /// Convergence in chi-square value
    Chi,
    /// Convergence in parameter value
    Par,
    /// Convergence in both chi-square and parameter
    Both,
    /// Convergence in orthogonality
    Dir,
    /// Maximum number of iterations reached
    MaxIter,
    /// ftol is too small; no further improvement
    Ftol,
    /// xtol is too small; no further improvement
    Xtol,
    /// gtol is too small; no further improvement
    Gtol,
}

// MP Fit Result
pub enum MPResult {
    Success(MPSuccess, MPStatus),
    Error(MPError),
}

/// Definition of results structure, for when fit completes
pub struct MPStatus {
    /// Final chi^2
    pub best_norm: f64,
    /// Starting value of chi^2
    pub orig_norm: f64,
    /// Number of iterations
    pub n_iter: usize,
    /// Number of function evaluations
    pub n_fev: usize,
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

pub trait MPFitter {
    fn eval(&self, params: &[f64], deviates: &mut [f64], derivs: Option<&mut [f64]>);

    fn n_params(&self) -> usize;
}

pub fn mpfit<T: MPFitter>(
    f: T,
    init: &mut [f64],
    params: Option<&[MPPar]>,
    config: &MPConfig,
) -> MPResult {
    if init.len() == 0 {
        return MPResult::Error(MPError::Empty);
    }
    let mut nfree: usize = 0;
    if let Some(ref pars) = params {
        if pars.len() == 0 {
            return MPResult::Error(MPError::Empty);
        }
        for p in *pars {
            if !p.fixed {
                nfree += 1;
            }
        }
        if nfree == 0 {
            return MPResult::Error(MPError::NoFree);
        }
    }
    let m = f.n_params();
    if m < nfree {
        return MPResult::Error(MPError::DoF);
    }
    let ldfjac = m;
    let mut fvec = vec![0.; m];
    f.eval(init, &mut fvec, None);
    let mut nfev: usize = 1;
    let fnorm = mp_enorm(m, &fvec);
    let orig_norm = fnorm * fnorm;
    let mut xnew = vec![0.; init.len()];
    xnew.copy_from_slice(init);
    MPResult::Success(
        MPSuccess::Both,
        MPStatus {
            best_norm: 0.0,
            orig_norm: 0.0,
            n_iter: 0,
            n_fev: 0,
            n_par: 0,
            n_free: 0,
            n_pegged: 0,
            n_func: 0,
            resid: vec![],
            xerror: vec![],
            covar: vec![],
        },
    )
}

///    function enorm
///
///    given an n-vector x, this function calculates the
///    euclidean norm of x.
///
///    the euclidean norm is computed by accumulating the sum of
///    squares in three different sums. the sums of squares for the
///    small and large components are scaled so that no overflows
///    occur. non-destructive underflows are permitted. underflows
///    and overflows do not occur in the computation of the unscaled
///    sum of squares for the intermediate components.
///    the definitions of small, intermediate and large components
///    depend on two constants, rdwarf and rgiant. the main
///    restrictions on these constants are that rdwarf**2 not
///    underflow and rgiant**2 not overflow. the constants
///    given here are suitable for every known computer.
///    the function statement is
///    double precision function enorm(n,x)
///    where
///
///    n is a positive integer input variable.
///
///    x is an input array of length n.
///
///    subprograms called
///
///    fortran-supplied ... dabs,dsqrt
///
///    argonne national laboratory. minpack project. march 1980.
///    burton s. garbow, kenneth e. hillstrom, jorge j. more
fn mp_enorm(n: usize, x: &[f64]) -> f64 {
    let mut s1 = 0.;
    let mut s2 = 0.;
    let mut s3 = 0.;
    let mut x1max = 0.;
    let mut x3max = 0.;
    let agiant = (f64::MAX.sqrt() * 0.1) / n as f64;
    let rdwarf = (f64::MIN_POSITIVE.sqrt() * 1.5) * 10.0;
    for val in x {
        let xabs = val.abs();
        if xabs > rdwarf && xabs < agiant {
            // sum for intermediate components.
            s2 += xabs * xabs;
        } else if xabs > rdwarf {
            // sum for large components.
            if xabs > x1max {
                let temp = x1max / xabs;
                s1 = 1.0 + s1 * temp * temp;
                x1max = xabs;
            } else {
                let temp = xabs / x1max;
                s1 += temp * temp;
            }
        } else if xabs > x3max {
            // sum for small components.
            let temp = x3max / xabs;
            s3 = 1.0 + s3 * temp * temp;
            x3max = xabs;
        } else if xabs != 0.0 {
            let temp = xabs / x3max;
            s3 += temp * temp;
        }
    }
    // calculation of norm.
    if s1 != 0.0 {
        x1max * (s1 + (s2 / x1max) / x1max).sqrt()
    } else if s2 != 0.0 {
        if s2 >= x3max {
            s2 * (1.0 + (x3max / s2) * (x3max * s3))
        } else {
            x3max * ((s2 / x3max) + (x3max * s3))
        }
        .sqrt()
    } else {
        x3max * s3.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use crate::{mpfit, MPFitter};

    #[test]
    fn linear() {
        struct Linear {
            x: Vec<f64>,
            y: Vec<f64>,
            ye: Vec<f64>,
        };

        impl MPFitter for Linear {
            fn eval(&self, params: &[f64], deviates: &mut [f64], derivs: Option<&mut [f64]>) {
                for (((d, x), y), ye) in deviates
                    .iter_mut()
                    .zip(self.x.iter())
                    .zip(self.y.iter())
                    .zip(self.ye.iter())
                {
                    let f = params[0] + params[1] * *x;
                    *d = (*y - f) / *ye;
                }
            }

            fn n_params(&self) -> usize {
                2
            }
        }
        let l = Linear {
            x: vec![
                -1.7237128E+00,
                1.8712276E+00,
                -9.6608055E-01,
                -2.8394297E-01,
                1.3416969E+00,
                1.3757038E+00,
                -1.3703436E+00,
                4.2581975E-02,
                -1.4970151E-01,
                8.2065094E-01,
            ],
            y: vec![
                1.9000429E-01,
                6.5807428E+00,
                1.4582725E+00,
                2.7270851E+00,
                5.5969253E+00,
                5.6249280E+00,
                0.787615,
                3.2599759E+00,
                2.9771762E+00,
                4.5936475E+00,
            ],
            ye: vec![0.07; 10],
        };
        let mut init = [1., 1.];
        let _ = mpfit(l, &mut init, None, &Default::default());
    }
}
