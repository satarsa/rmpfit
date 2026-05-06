//! Very simple pure Rust implementation of the
//! [CMPFIT](https://pages.physics.wisc.edu/~craigm/idl/cmpfit.html) library:
//! the Levenberg-Marquardt technique to solve the least-squares problem.
//!
//! The code is mainly copied directly from CMPFIT almost without changing.
//! The original CMPFIT tests (Linear (free parameters), Quad (free and fixed parameters),
//! and Gaussian (free and fixed parameters) function) are reproduced and passed.
//!
//! Just a few obvious Rust-specific optimizations are done:
//! * Removing ```goto``` (fuf).
//! * Standard Rust Result as result.
//! * A few loops are zipped to help the compiler optimize the code
//!   (no performance tests are done anyway).
//! * Using trait ```MPFitter``` to call the user code.
//! * Using ```bool``` type if possible.
//!
//! # Advantages
//! * Pure Rust.
//! * No external dependencies
//!   ([assert_approx_eq](https://docs.rs/assert_approx_eq/) just for testing).
//! * Internal Jacobian calculations.
//! * Sided, analytical or user provided derivatives are also implemented.
//! * Derivative debug mode (comparing analytical vs numerical) prints to stderr (as in cmpfit).
//!
//! # Usage Example
//! A user should implement trait ```MPFitter``` for its struct:
//! ```
//! use assert_approx_eq::assert_approx_eq;
//! use rmpfit::{MPFitter, MPResult};
//!
//! struct Linear {
//!     x: Vec<f64>,
//!     y: Vec<f64>,
//!     ye: Vec<f64>,
//! }
//!
//! impl MPFitter for Linear {
//!     fn eval(&mut self, params: &[f64], deviates: &mut [f64]) -> MPResult<()> {
//!         for (((d, x), y), ye) in deviates
//!             .iter_mut()
//!             .zip(self.x.iter())
//!             .zip(self.y.iter())
//!             .zip(self.ye.iter())
//!             {
//!                 let f = params[0] + params[1] * *x;
//!                 *d = (*y - f) / *ye;
//!             }
//!             Ok(())
//!         }
//!
//!         fn number_of_points(&self) -> usize {
//!             self.x.len()
//!         }
//!     }
//!
//! fn main() {
//!     let mut l = Linear {
//!         x: vec![
//!                 -1.7237128E+00,
//!                 1.8712276E+00,
//!                 -9.6608055E-01,
//!                 -2.8394297E-01,
//!                 1.3416969E+00,
//!                 1.3757038E+00,
//!                 -1.3703436E+00,
//!                 4.2581975E-02,
//!                 -1.4970151E-01,
//!                 8.2065094E-01,
//!         ],
//!         y: vec![
//!                 1.9000429E-01,
//!                 6.5807428E+00,
//!                 1.4582725E+00,
//!                 2.7270851E+00,
//!                 5.5969253E+00,
//!                 5.6249280E+00,
//!                 0.787615,
//!                 3.2599759E+00,
//!                 2.9771762E+00,
//!                 4.5936475E+00,
//!         ],
//!         ye: vec![0.07; 10],
//!     };
//!     // initializing input parameters
//!    let mut init = [1., 1.];
//!    let res = l.mpfit(&mut init).unwrap();
//!    assert_approx_eq!(init[0], 3.20996572); // actual 3.2
//!    assert_approx_eq!(res.xerror[0], 0.02221018);
//!    assert_approx_eq!(init[1], 1.77095420); // actual 1.78
//!    assert_approx_eq!(res.xerror[1], 0.01893756);
//! }
//! ```
//! then ```init``` will contain the refined parameters of the fitting function.
//! If user function fails to calculate residuals, it should return ```MPError::Eval```.
//!
use std::fmt;
use std::fmt::Formatter;

/// MPFIT return result
pub type MPResult<T> = Result<T, MPError>;

/// Controls how numerical derivatives are computed for a parameter, or whether
/// analytical (user-supplied) derivatives are used instead.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum MPSide {
    /// One-sided derivative computed automatically: direction chosen to stay
    /// within parameter bounds (default).
    #[default]
    Auto,
    /// One-sided forward derivative: `(f(x+h) - f(x)) / h`
    Right,
    /// One-sided backward derivative: `(f(x) - f(x-h)) / h`
    Left,
    /// Two-sided (central difference) derivative: `(f(x+h) - f(x-h)) / (2h)`
    Both,
    /// User-supplied analytical derivative. The [`MPFitter::jacobian`] method
    /// must be implemented for parameters with this variant.
    User,
}

/// Parameter constraint structure
pub struct MPPar {
    /// A boolean value, whether the parameter is to be held
    /// fixed or not. Fixed parameters are not varied by
    /// MPFIT, but are passed on to ```MPFitter``` for evaluation.
    pub fixed: bool,
    /// Is the parameter fixed at the lower boundary? If ```true```,
    /// then the parameter is bounded on the lower side.
    pub limited_low: bool,
    /// Is the parameter fixed at the upper boundary? If ```true```,
    /// then the parameter is bounded on the upper side.
    pub limited_up: bool,
    /// Gives the parameter limit on the lower side.
    pub limit_low: f64,
    /// Gives the parameter limit on the upper side.
    pub limit_up: f64,
    /// The step size to be used in calculating the numerical
    /// derivatives. If set to zero, then the step size is computed automatically.
    /// This value is superseded by the ```MPConfig::rel_step``` value.
    pub step: f64,
    /// The *relative* step size to be used in calculating
    /// the numerical derivatives.  This number is the
    /// fractional size of the step, compared to the
    /// parameter value.  This value supersedes the ```MPConfig::step```
    /// setting.  If the parameter is zero, then a default
    /// step size is chosen.
    pub rel_step: f64,
    /// Controls how the derivative is computed for this parameter.
    /// See [`MPSide`] for the available modes. Default: [`MPSide::Auto`].
    pub side: MPSide,
    /// If `true`, compute *both* the analytical derivative (via
    /// [`MPFitter::jacobian`]) *and* a numerical derivative, print a
    /// comparison to stderr, and use the numerical value. Useful for
    /// verifying that a hand-coded Jacobian is correct.
    ///
    /// **Note:** when `deriv_debug` is `true`, do *not* set `side` to
    /// [`MPSide::User`]; instead set it to the numerical variant you want to
    /// compare against (`Auto`, `Right`, `Left`, or `Both`).
    pub deriv_debug: bool,
    /// Relative tolerance for the derivative debug comparison. Differences
    /// smaller than `deriv_reltol * |analytical|` are not printed.
    pub deriv_reltol: f64,
    /// Absolute tolerance for the derivative debug comparison. Differences
    /// smaller than `deriv_abstol` are not printed.
    pub deriv_abstol: f64,
}

impl MPPar {
    pub const fn new() -> Self {
        MPPar {
            fixed: false,
            limited_low: false,
            limited_up: false,
            limit_low: 0.0,
            limit_up: 0.0,
            step: 0.0,
            rel_step: 0.0,
            side: MPSide::Auto,
            deriv_debug: false,
            deriv_reltol: 0.0,
            deriv_abstol: 0.0,
        }
    }
}

impl Default for MPPar {
    fn default() -> Self {
        Self::new()
    }
}

/// MPFIT configuration structure
#[derive(Clone)]
pub struct MPConfig {
    /// Relative chi-square convergence criterion (Default: 1e-10)
    pub ftol: f64,
    /// Relative parameter convergence criterion  (Default: 1e-10)
    pub xtol: f64,
    /// Orthogonality convergence criterion        (Default: 1e-10)
    pub gtol: f64,
    /// Finite derivative step size                (Default: f64::EPSILON)
    pub epsfcn: f64,
    /// Initial step bound                         (Default: 100.0)
    pub step_factor: f64,
    /// Range tolerance for covariance calculation (Default: 1e-14)
    pub covtol: f64,
    /// Maximum number of iterations (Default: 200).  If maxiter == 0,
    /// then basic error checking is done, and parameter
    /// errors/covariances are estimated based on input
    /// parameter values, but no fitting iterations are done.
    pub max_iter: usize,
    /// Maximum number of function evaluations, or 0 for no limit
    /// (Default: 0 (no limit))
    pub max_fev: usize,
    /// Scale variables by user values?
    /// true = yes, user scale values in diag;
    /// false = no, variables scaled internally (Default: false)
    pub do_user_scale: bool,
    /// Disable check for infinite quantities from user?
    /// true = perform check;
    /// false = do not perform check (Default: false)
    pub no_finite_check: bool,
}

impl MPConfig {
    pub const fn new() -> Self {
        MPConfig {
            ftol: 1e-10,
            xtol: 1e-10,
            gtol: 1e-10,
            epsfcn: f64::EPSILON,
            step_factor: 100.0,
            covtol: 1e-14,
            max_iter: 200,
            max_fev: 0,
            do_user_scale: false,
            no_finite_check: false,
        }
    }
}

impl Default for MPConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// MPFIT error status
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
    /// Error during evaluation by user
    Eval,
}

/// Potential success status
#[derive(PartialEq)]
pub enum MPSuccess {
    /// Not finished iterations
    NotDone,
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

/// Status structure, for fit when it completes
pub struct MPStatus {
    /// Success enum
    pub success: MPSuccess,
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

/// Trait to be implemented by user.
pub trait MPFitter {
    /// Main evaluation procedure which is called from ```mpfit```. Size of ```deviates``` is equal
    /// to the value returned by ```number_of_points```. User should compute the residuals
    /// using parameters from ```params``` and any user data that are required, and fill
    /// the ```deviates``` slice.
    /// The residuals are defined as ```(y[i] - f(x[i]))/y_error[i]```.
    fn eval(&mut self, params: &[f64], deviates: &mut [f64]) -> MPResult<()>;

    /// Number of the data points in the user private data.
    fn number_of_points(&self) -> usize;

    /// Returns the configuration for this fit.
    /// Override to supply a custom [`MPConfig`]; the default returns a
    /// reference to a shared static with all-default values.
    fn config(&self) -> &MPConfig {
        static DEFAULT: MPConfig = MPConfig::new();
        &DEFAULT
    }

    /// Returns parameter constraints for the fitted values.
    /// Override when you need fixed parameters, bounds, or custom step sizes.
    /// An empty slice (the default) means all parameters are free with no
    /// constraints — identical to the old `None` return.
    fn parameters(&self) -> &[MPPar] {
        &[]
    }

    /// Optionally supply analytical (user-computed) partial derivatives.
    /// This method is called once per Jacobian evaluation whenever at least
    /// one free parameter has [`MPSide::User`] set (or has `deriv_debug`
    /// enabled). The default implementation returns [`MPError::Eval`], which
    /// causes the fit to abort - override it when you set any parameter's
    /// `side` to [`MPSide::User`].
    ///
    /// # Arguments
    /// * `params`   - current values of *all* parameters (length = `npar`).
    /// * `deviates` - output residuals (same as [`MPFitter::eval`]).
    /// * `derivs`   - per-parameter derivative columns, length = `npar`.
    ///   - `derivs[i]` is `Some(col)` when parameter `i` needs an analytical
    ///     derivative; `col` has length `m` (number of residuals). Fill
    ///     `col[k]` with `ddeviates[k] / dparams[i]`.
    ///   - `derivs[i]` is `None` for parameters using numerical derivatives.
    ///
    /// The method must also fill `deviates` (the residual vector), exactly as
    /// [`MPFitter::eval`] would.
    #[allow(unused_variables)]
    fn jacobian(
        &mut self,
        params: &[f64],
        deviates: &mut [f64],
        derivs: &mut [Option<Vec<f64>>],
    ) -> MPResult<()> {
        Err(MPError::Eval)
    }

    /// Main function to refine the parameters.
    /// # Arguments
    /// * `xall` - A mutable slice with starting fit parameters
    fn mpfit(&mut self, xall: &mut [f64]) -> MPResult<MPStatus>
    where
        Self: Sized,
    {
        // Clone once so the immutable borrow ends before we pass `self` mutably
        // into MPFit.  Users returning &self.config avoid constructing a new
        // value; users using the default return a &'static so the clone is cheap.
        let config = self.config().clone();
        let mut fit = MPFit::new(self, xall, &config)?;
        fit.check_config()?;
        fit.parse_params()?;
        fit.init_lm()?;
        loop {
            fit.fill_xnew();
            fit.fdjac2()?;
            fit.check_limits();
            fit.qrfac();
            fit.scale();
            fit.transpose();
            if !fit.check_is_finite() {
                return Err(MPError::Nan);
            }
            let gnorm = fit.gnorm();
            if gnorm <= config.gtol {
                fit.info = MPSuccess::Dir;
            }
            if fit.info != MPSuccess::NotDone {
                return fit.terminate();
            }
            if config.max_iter == 0 {
                fit.info = MPSuccess::MaxIter;
                return fit.terminate();
            }
            fit.rescale();
            loop {
                fit.lmpar();
                let res = fit.iterate(gnorm)?;
                match res {
                    MPDone::Exit => return fit.terminate(),
                    MPDone::Inner => continue,
                    MPDone::Outer => break,
                }
            }
        }
    }
}

/// (f64::MIN_POSITIVE * 1.5).sqrt() * 10
const MP_RDWARF: f64 = 1.8269129289596699e-153;
/// f64::MAX.sqrt() * 0.1
const MP_RGIANT: f64 = 1.3407807799935083e+153;

/// Internal structure to hold calculated values.
struct MPFit<'a, T: MPFitter> {
    m: usize,
    npar: usize,
    nfree: usize,
    ifree: Vec<usize>,
    fvec: Vec<f64>,
    nfev: usize,
    xnew: Vec<f64>,
    x: Vec<f64>,
    xall: &'a mut [f64],
    qtf: Vec<f64>,
    fjac: Vec<f64>,
    step: Vec<f64>,
    dstep: Vec<f64>,
    dside: Vec<MPSide>,
    dderiv_debug: Vec<bool>,
    dderiv_reltol: Vec<f64>,
    dderiv_abstol: Vec<f64>,
    qllim: Vec<bool>,
    qulim: Vec<bool>,
    llim: Vec<f64>,
    ulim: Vec<f64>,
    qanylim: bool,
    f: &'a mut T,
    wa1: Vec<f64>,
    wa2: Vec<f64>,
    wa3: Vec<f64>,
    wa4: Vec<f64>,
    ipvt: Vec<usize>,
    diag: Vec<f64>,
    fnorm: f64,
    fnorm1: f64,
    xnorm: f64,
    delta: f64,
    info: MPSuccess,
    orig_norm: f64,
    par: f64,
    iter: usize,
    cfg: &'a MPConfig,
}

impl<'a, F: MPFitter> MPFit<'a, F> {
    fn new(f: &'a mut F, xall: &'a mut [f64], cfg: &'a MPConfig) -> MPResult<MPFit<'a, F>> {
        let m = f.number_of_points();
        let npar = xall.len();
        if m == 0 {
            Err(MPError::Empty)
        } else {
            Ok(MPFit {
                m,
                npar,
                nfree: 0,
                ifree: vec![],
                fvec: vec![0.; m],
                nfev: 1,
                xnew: vec![0.; npar],
                x: vec![],
                xall,
                qtf: vec![],
                fjac: vec![],
                step: vec![],
                dstep: vec![],
                dside: vec![],
                dderiv_debug: vec![],
                dderiv_reltol: vec![],
                dderiv_abstol: vec![],
                qllim: vec![],
                qulim: vec![],
                llim: vec![],
                ulim: vec![],
                qanylim: false,
                f,
                wa1: vec![0.; npar],
                wa2: vec![0.; m],
                wa3: vec![0.; npar],
                wa4: vec![0.; m],
                ipvt: vec![0; npar],
                diag: vec![0.; npar],
                fnorm: -1.0,
                fnorm1: -1.0,
                xnorm: -1.0,
                delta: 0.0,
                info: MPSuccess::NotDone,
                orig_norm: 0.0,
                par: 0.0,
                iter: 1,
                cfg,
            })
        }
    }

    //     subroutine fdjac2
    //
    //     this subroutine computes a forward-difference approximation
    //     to the m by n jacobian matrix associated with a specified
    //     problem of m functions in n variables.
    //
    //     the subroutine statement is
    //
    //	subroutine fdjac2(fcn,m,n,x,fvec,fjac,ldfjac,iflag,epsfcn,wa)
    //
    //     where
    //
    //	fcn is the name of the user-supplied subroutine which
    //	  calculates the functions. fcn must be declared
    //	  in an external statement in the user calling
    //	  program, and should be written as follows.
    //
    //	  subroutine fcn(m,n,x,fvec,iflag)
    //	  integer m,n,iflag
    //	  double precision x(n),fvec(m)
    //	  ----------
    //	  calculate the functions at x and
    //	  return this vector in fvec.
    //	  ----------
    //	  return
    //	  end
    //
    //	  the value of iflag should not be changed by fcn unless
    //	  the user wants to terminate execution of fdjac2.
    //	  in this case set iflag to a negative integer.
    //
    //	m is a positive integer input variable set to the number
    //	  of functions.
    //
    //	n is a positive integer input variable set to the number
    //	  of variables. n must not exceed m.
    //
    //	x is an input array of length n.
    //
    //	fvec is an input array of length m which must contain the
    //	  functions evaluated at x.
    //
    //	fjac is an output m by n array which contains the
    //	  approximation to the jacobian matrix evaluated at x.
    //
    //	ldfjac is a positive integer input variable not less than m
    //	  which specifies the leading dimension of the array fjac.
    //
    //	iflag is an integer variable which can be used to terminate
    //	  the execution of fdjac2. see description of fcn.
    //
    //	epsfcn is an input variable used in determining a suitable
    //	  step length for the forward-difference approximation. this
    //	  approximation assumes that the relative errors in the
    //	  functions are of the order of epsfcn. if epsfcn is less
    //	  than the machine precision, it is assumed that the relative
    //	  errors in the functions are of the order of the machine
    //	  precision.
    //
    //	wa is a work array of length m.
    //
    //     subprograms called
    //
    //	user-supplied ...... fcn
    //
    //	minpack-supplied ... dpmpar
    //
    //	fortran-supplied ... dabs,dmax1,dsqrt
    //
    //     argonne national laboratory. minpack project. march 1980.
    //     burton s. garbow, kenneth e. hillstrom, jorge j. more
    //
    fn fdjac2(&mut self) -> MPResult<()> {
        let eps = self.cfg.epsfcn.max(f64::EPSILON).sqrt();
        self.fjac.fill(0.);
        let has_analytical = self
            .dside
            .iter()
            .zip(&self.dderiv_debug)
            .any(|(s, d)| *s == MPSide::User || *d);
        if has_analytical {
            let m = self.m;
            let npar = self.npar;
            let mut derivs: Vec<Option<Vec<f64>>> = vec![None; npar];
            for j in 0..self.nfree {
                let free_p = self.ifree[j];
                if self.dside[j] == MPSide::User || self.dderiv_debug[j] {
                    derivs[free_p] = Some(vec![0.0; m]);
                }
            }
            self.f.jacobian(&self.xnew, &mut self.wa4, &mut derivs)?;
            self.nfev += 1;
            for j in 0..self.nfree {
                let free_p = self.ifree[j];
                if let Some(ref col) = derivs[free_p] {
                    let base = j * m;
                    self.fjac[base..base + m].copy_from_slice(col);
                }
            }
        }
        let has_debug = self.dderiv_debug.iter().any(|d| *d);
        if has_debug {
            eprintln!("FJAC DEBUG BEGIN");
            eprintln!(
                "#  {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
                "IPNT", "FUNC", "DERIV_U", "DERIV_N", "DIFF_ABS", "DIFF_REL"
            );
        }
        let mut ij = 0usize;
        for j in 0..self.nfree {
            let free_p = self.ifree[j];
            let side = self.dside[j];
            let debug = self.dderiv_debug[j];
            if side == MPSide::User && !debug {
                ij += self.m;
                continue;
            }
            if debug {
                eprintln!("FJAC PARM {}", free_p);
            }
            let temp = self.xnew[free_p];
            let mut h = eps * temp.abs();
            if free_p < self.step.len() && self.step[free_p] > 0. {
                h = self.step[free_p];
            }
            if free_p < self.dstep.len() && self.dstep[free_p] > 0. {
                h = (self.dstep[free_p] * temp).abs();
            }
            if h == 0. {
                h = eps;
            }
            if side == MPSide::Left
                || (side == MPSide::Auto
                    && j < self.qulim.len()
                    && self.qulim[j]
                    && j < self.ulim.len()
                    && temp > self.ulim[j] - h)
            {
                h = -h;
            }
            self.xnew[free_p] = temp + h;
            self.f.eval(&self.xnew, &mut self.wa4)?;
            self.nfev += 1;
            self.xnew[free_p] = temp;
            if side != MPSide::Both {
                let dr = self.dderiv_reltol[j];
                let da = self.dderiv_abstol[j];
                for i in 0..self.m {
                    let numerical = (self.wa4[i] - self.fvec[i]) / h;
                    if debug {
                        let analytical = self.fjac[ij];
                        let diff = analytical - numerical;
                        if (da == 0. && dr == 0. && (analytical != 0. || numerical != 0.))
                            || ((da != 0. || dr != 0.) && diff.abs() > da + analytical.abs() * dr)
                        {
                            eprintln!(
                                "   {:>10} {:>10.4e} {:>10.4e} {:>10.4e} {:>10.4e} {:>10.4e}",
                                i,
                                self.fvec[i],
                                analytical,
                                numerical,
                                diff,
                                if analytical == 0. {
                                    0.
                                } else {
                                    diff / analytical
                                }
                            );
                        }
                    }
                    self.fjac[ij] = numerical;
                    ij += 1;
                }
            } else {
                let m = self.m;
                self.wa2[..m].copy_from_slice(&self.wa4[..m]);
                self.xnew[free_p] = temp - h;
                self.f.eval(&self.xnew, &mut self.wa4)?; // wa4 = f(x-h)
                self.nfev += 1;
                self.xnew[free_p] = temp;
                let dr = self.dderiv_reltol[j];
                let da = self.dderiv_abstol[j];
                for i in 0..m {
                    let numerical = (self.wa2[i] - self.wa4[i]) / (2. * h);
                    if debug {
                        let analytical = self.fjac[ij];
                        let diff = analytical - numerical;
                        if (da == 0. && dr == 0. && (analytical != 0. || numerical != 0.))
                            || ((da != 0. || dr != 0.) && diff.abs() > da + analytical.abs() * dr)
                        {
                            eprintln!(
                                "   {:>10} {:>10.4e} {:>10.4e} {:>10.4e} {:>10.4e} {:>10.4e}",
                                i,
                                self.fvec[i],
                                analytical,
                                numerical,
                                diff,
                                if analytical == 0. {
                                    0.
                                } else {
                                    diff / analytical
                                }
                            );
                        }
                    }
                    self.fjac[ij] = numerical;
                    ij += 1;
                }
            }
        }
        if has_debug {
            eprintln!("FJAC DEBUG END");
        }
        Ok(())
    }

    //     subroutine qrfac
    //
    //     this subroutine uses householder transformations with column
    //     pivoting (optional) to compute a qr factorization of the
    //     m by n matrix a. that is, qrfac determines an orthogonal
    //     matrix q, a permutation matrix p, and an upper trapezoidal
    //     matrix r with diagonal elements of nonincreasing magnitude,
    //     such that a*p = q*r. the householder transformation for
    //     column k, k = 1,2,...,min(m,n), is of the form
    //
    //			    t
    //	    i - (1/u(k))*u*u
    //
    //     where u has zeros in the first k-1 positions. the form of
    //     this transformation and the method of pivoting first
    //     appeared in the corresponding linpack subroutine.
    //
    //     the subroutine statement is
    //
    //	subroutine qrfac(m,n,a,lda,pivot,ipvt,lipvt,rdiag,acnorm,wa)
    //
    //     where
    //
    //	m is a positive integer input variable set to the number
    //	  of rows of a.
    //
    //	n is a positive integer input variable set to the number
    //	  of columns of a.
    //
    //	a is an m by n array. on input a contains the matrix for
    //	  which the qr factorization is to be computed. on output
    //	  the strict upper trapezoidal part of a contains the strict
    //	  upper trapezoidal part of r, and the lower trapezoidal
    //	  part of a contains a factored form of q (the non-trivial
    //	  elements of the u vectors described above).
    //
    //	lda is a positive integer input variable not less than m
    //	  which specifies the leading dimension of the array a.
    //
    //	pivot is a logical input variable. if pivot is set true,
    //	  then column pivoting is enforced. if pivot is set false,
    //	  then no column pivoting is done.
    //
    //	ipvt is an integer output array of length lipvt. ipvt
    //	  defines the permutation matrix p such that a*p = q*r.
    //	  column j of p is column ipvt(j) of the identity matrix.
    //	  if pivot is false, ipvt is not referenced.
    //
    //	lipvt is a positive integer input variable. if pivot is false,
    //	  then lipvt may be as small as 1. if pivot is true, then
    //	  lipvt must be at least n.
    //
    //	rdiag is an output array of length n which contains the
    //	  diagonal elements of r.
    //
    //	acnorm is an output array of length n which contains the
    //	  norms of the corresponding columns of the input matrix a.
    //	  if this information is not needed, then acnorm can coincide
    //	  with rdiag.
    //
    //	wa is a work array of length n. if pivot is false, then wa
    //	  can coincide with rdiag.
    //
    //     subprograms called
    //
    //	minpack-supplied ... dpmpar,enorm
    //
    //	fortran-supplied ... dmax1,dsqrt,min0
    //
    //     argonne national laboratory. minpack project. march 1980.
    //     burton s. garbow, kenneth e. hillstrom, jorge j. more
    fn qrfac(&mut self) {
        // Compute the QR factorization of the jacobian
        // compute the initial column norms and initialize several arrays.
        let m = self.m;
        for (j, col) in self.fjac[..m * self.nfree].chunks_exact(m).enumerate() {
            let norm = col.enorm();
            self.wa1[j] = norm;
            self.wa2[j] = norm;
            self.wa3[j] = norm;
            self.ipvt[j] = j;
        }
        // reduce a to r with householder transformations.
        for j in 0..m.min(self.nfree) {
            // bring the column of largest norm into the pivot position.
            let mut kmax = j;
            for k in j + 1..self.nfree {
                if self.wa1[k] > self.wa1[kmax] {
                    kmax = k;
                }
            }
            if kmax != j {
                for i in 0..m {
                    self.fjac.swap(m * j + i, m * kmax + i);
                }
                self.wa1[kmax] = self.wa1[j];
                self.wa3[kmax] = self.wa3[j];
                self.ipvt.swap(j, kmax);
            }
            let jj = j * (m + 1);
            let jjj = m - j + jj;
            let mut ajnorm = self.fjac[jj..jjj].enorm();
            if ajnorm == 0. {
                self.wa1[j] = -ajnorm;
                continue;
            }
            if self.fjac[jj] < 0. {
                ajnorm = -ajnorm;
            }
            for fjac in self.fjac[jj..jjj].iter_mut() {
                *fjac /= ajnorm;
            }
            self.fjac[jj] += 1.;
            // apply the transformation to the remaining columns
            // and update the norms.
            let jp1 = j + 1;
            if jp1 < self.nfree {
                for k in jp1..self.nfree {
                    let mut sum = 0.;
                    for i in j..m {
                        sum += self.fjac[m * j + i] * self.fjac[m * k + i];
                    }
                    let temp = sum / self.fjac[m * j + j];
                    for i in j..m {
                        self.fjac[m * k + i] -= temp * self.fjac[m * j + i];
                    }
                    if self.wa1[k] != 0. {
                        let temp = self.fjac[j + m * k] / self.wa1[k];
                        let temp = (1. - temp.powi(2)).max(0.);
                        self.wa1[k] *= temp.sqrt();
                        let temp = self.wa1[k] / self.wa3[k];
                        if 0.05 * temp * temp < f64::EPSILON {
                            let start = jp1 + m * k;
                            self.wa1[k] = self.fjac[start..start + m - j - 1].enorm();
                            self.wa3[k] = self.wa1[k];
                        }
                    }
                }
            }
            self.wa1[j] = -ajnorm;
        }
    }

    fn parse_params(&mut self) -> MPResult<()> {
        let pars = self.f.parameters();
        if pars.is_empty() {
            // No constraints: every parameter is free with default derivative settings.
            self.nfree = self.npar;
            self.ifree = (0..self.npar).collect();
            self.dside = vec![MPSide::Auto; self.npar];
            self.dderiv_debug = vec![false; self.npar];
            self.dderiv_reltol = vec![0.0; self.npar];
            self.dderiv_abstol = vec![0.0; self.npar];
        } else {
            pars.iter().enumerate().for_each(|(i, p)| {
                if !p.fixed {
                    self.nfree += 1;
                    self.ifree.push(i);
                }
            });
            if self.nfree == 0 {
                return Err(MPError::NoFree);
            }
            for (i, p) in pars.iter().enumerate() {
                if (p.limited_low && (self.xall[i] < p.limit_low))
                    || (p.limited_up && (self.xall[i] > p.limit_up))
                {
                    return Err(MPError::InitBounds);
                }
                if !p.fixed && p.limited_low && p.limited_up && p.limit_low >= p.limit_up {
                    return Err(MPError::Bounds);
                }
            }
            self.ifree.iter().for_each(|i| {
                let p = &pars[*i];
                self.qllim.push(p.limited_low);
                self.qulim.push(p.limited_up);
                self.llim.push(p.limit_low);
                self.ulim.push(p.limit_up);
                if p.limited_low || p.limited_up {
                    self.qanylim = true;
                }
                self.step.push(p.step);
                self.dstep.push(p.rel_step);
                self.dside.push(p.side);
                self.dderiv_debug.push(p.deriv_debug);
                self.dderiv_reltol.push(p.deriv_reltol);
                self.dderiv_abstol.push(p.deriv_abstol);
            });
        }
        if self.m < self.nfree {
            return Err(MPError::DoF);
        }
        Ok(())
    }

    // Initialize Levenberg-Marquardt parameter and iteration counter
    fn init_lm(&mut self) -> MPResult<()> {
        self.f.eval(self.xall, &mut self.fvec)?;
        self.nfev += 1;
        self.fnorm = self.fvec.enorm();
        self.orig_norm = self.fnorm * self.fnorm;
        self.xnew.copy_from_slice(self.xall);
        self.x = Vec::with_capacity(self.nfree);
        for i in 0..self.nfree {
            self.x.push(self.xall[self.ifree[i]]);
        }
        self.qtf = vec![0.; self.nfree];
        self.fjac = vec![0.; self.m * self.nfree];
        Ok(())
    }

    fn check_limits(&mut self) {
        if !self.qanylim {
            return;
        }
        let m = self.m;
        for j in 0..self.nfree {
            let lpegged = j < self.qllim.len() && self.x[j] == self.llim[j];
            let upegged = j < self.qulim.len() && self.x[j] == self.ulim[j];
            let mut sum = 0.;
            // If the parameter is pegged at a limit, compute the gradient direction
            let ij = j * m;
            if lpegged || upegged {
                for i in 0..m {
                    sum += self.fvec[i] * self.fjac[ij + i];
                }
            }
            // If pegged at lower limit and gradient is toward negative then
            // reset gradient to zero
            if lpegged && sum > 0. {
                self.fjac[ij..ij + m].fill(0.0);
            }
            // If pegged at upper limit and gradient is toward positive then
            // reset gradient to zero
            if upegged && sum < 0. {
                self.fjac[ij..ij + m].fill(0.0);
            }
        }
    }

    /// On the first iteration and if user_scale is requested, scale according
    /// to the norms of the columns of the initial jacobian,
    /// calculate the norm of the scaled x, and initialize the step bound delta.
    fn scale(&mut self) {
        if self.iter != 1 {
            return;
        }
        if !self.cfg.do_user_scale {
            for j in 0..self.nfree {
                self.diag[self.ifree[j]] = if self.wa2[j] == 0. { 1. } else { self.wa2[j] };
            }
        }
        for j in 0..self.nfree {
            self.wa3[j] = self.diag[self.ifree[j]] * self.x[j];
        }
        self.xnorm = self.wa3.enorm();
        self.delta = self.cfg.step_factor * self.xnorm;
        if self.delta == 0. {
            self.delta = self.cfg.step_factor;
        }
    }

    fn fill_xnew(&mut self) {
        for i in 0..self.nfree {
            self.xnew[self.ifree[i]] = self.x[i];
        }
    }

    /// form (q transpose)*fvec and store the first n components in qtf.
    fn transpose(&mut self) {
        self.wa4.copy_from_slice(&self.fvec);
        let mut jj = 0;
        for j in 0..self.nfree {
            let temp = self.fjac[jj];
            if temp != 0. {
                let mut sum = 0.0;
                let mut ij = jj;
                for i in j..self.m {
                    sum += self.fjac[ij] * self.wa4[i];
                    ij += 1;
                }
                let temp = -sum / temp;
                ij = jj;
                for i in j..self.m {
                    self.wa4[i] += self.fjac[ij] * temp;
                    ij += 1;
                }
            }
            self.fjac[jj] = self.wa1[j];
            jj += self.m + 1;
            self.qtf[j] = self.wa4[j];
        }
    }

    /// Check for overflow.  This should be a cheap test here since FJAC
    /// has been reduced to a (small) square matrix, and the test is O(N^2).
    fn check_is_finite(&self) -> bool {
        if !self.cfg.no_finite_check {
            for val in &self.fjac {
                if !val.is_finite() {
                    return false;
                }
            }
        }
        true
    }

    /// compute the norm of the scaled gradient.
    fn gnorm(&self) -> f64 {
        let mut gnorm: f64 = 0.;
        if self.fnorm != 0. {
            let mut jj = 0;
            for j in 0..self.nfree {
                let l = self.ipvt[j];
                if self.wa2[l] != 0. {
                    let mut sum = 0.;
                    for (ij, i) in (jj..).zip(0..=j) {
                        sum += self.fjac[ij] * (self.qtf[i] / self.fnorm);
                    }
                    gnorm = gnorm.max((sum / self.wa2[l]).abs());
                }
                jj += self.m;
            }
        }
        gnorm
    }

    fn terminate(mut self) -> MPResult<MPStatus> {
        for i in 0..self.nfree {
            self.xall[self.ifree[i]] = self.x[i];
        }
        /* Compute number of pegged parameters */
        let n_pegged = self
            .f
            .parameters()
            .iter()
            .enumerate()
            .filter(|(i, p)| {
                (p.limited_low && p.limit_low == self.xall[*i])
                    || (p.limited_up && p.limit_up == self.xall[*i])
            })
            .count();
        /* Compute and return the covariance matrix and/or parameter errors */
        self = self.covar();
        let mut covar = vec![0.; self.npar * self.npar];
        for j in 0..self.nfree {
            let k = self.ifree[j] * self.npar;
            let l = j * self.m;
            for i in 0..self.nfree {
                covar[k + self.ifree[i]] = self.fjac[l + i]
            }
        }
        let mut xerror = vec![0.; self.npar];
        for j in 0..self.nfree {
            let cc = self.fjac[j * self.m + j];
            if cc > 0. {
                xerror[self.ifree[j]] = cc.sqrt();
            }
        }
        let best_norm = self.fnorm.max(self.fnorm1);
        Ok(MPStatus {
            success: self.info,
            best_norm: best_norm * best_norm,
            orig_norm: self.orig_norm,
            n_iter: self.iter,
            n_fev: self.nfev,
            n_par: self.npar,
            n_free: self.nfree,
            n_pegged,
            n_func: self.m,
            resid: self.fvec,
            xerror,
            covar,
        })
    }

    //     subroutine covar
    //
    //    given an m by n matrix a, the problem is to determine
    //    the covariance matrix corresponding to a, defined as
    //
    //                   t
    //          inverse(a *a) .
    //
    //    this subroutine completes the solution of the problem
    //    if it is provided with the necessary information from the
    //    qr factorization, with column pivoting, of a. that is, if
    //    a*p = q*r, where p is a permutation matrix, q has orthogonal
    //    columns, and r is an upper triangular matrix with diagonal
    //    elements of nonincreasing magnitude, then covar expects
    //    the full upper triangle of r and the permutation matrix p.
    //    the covariance matrix is then computed as
    //
    //                     t     t
    //          p*inverse(r *r)*p  .
    //
    //    if a is nearly rank deficient, it may be desirable to compute
    //    the covariance matrix corresponding to the linearly independent
    //    columns of a. to define the numerical rank of a, covar uses
    //    the tolerance tol. if l is the largest integer such that
    //
    //          abs(r(l,l)) .gt. tol*abs(r(1,1)) ,
    //
    //    then covar computes the covariance matrix corresponding to
    //    the first l columns of r. for k greater than l, column
    //    and row ipvt(k) of the covariance matrix are set to zero.
    //
    //    the subroutine statement is
    //
    //      subroutine covar(n,r,ldr,ipvt,tol,wa)
    //
    //    where
    //
    //      n is a positive integer input variable set to the order of r.
    //
    //      r is an n by n array. on input the full upper triangle must
    //        contain the full upper triangle of the matrix r. on output
    //        r contains the square symmetric covariance matrix.
    //
    //      ldr is a positive integer input variable not less than n
    //        which specifies the leading dimension of the array r.
    //
    //      ipvt is an integer input array of length n which defines the
    //        permutation matrix p such that a*p = q*r. column j of p
    //        is column ipvt(j) of the identity matrix.
    //
    //      tol is a nonnegative input variable used to define the
    //        numerical rank of a in the manner described above.
    //
    //      wa is a work array of length n.
    //
    //    subprograms called
    //
    //      fortran-supplied ... dabs
    //
    //    argonne national laboratory. minpack project. august 1980.
    //    burton s. garbow, kenneth e. hillstrom, jorge j. more
    fn covar(mut self) -> Self {
        /*
         * form the inverse of r in the full upper triangle of r.
         */
        let tolr = self.cfg.covtol * self.fjac[0].abs();
        let mut l: isize = -1;
        for k in 0..self.nfree {
            let k0 = k * self.m;
            let kk = k0 + k;
            if self.fjac[kk].abs() <= tolr {
                break;
            }
            self.fjac[kk] = 1.0 / self.fjac[kk];
            for j in 0..k {
                let kj = k0 + j;
                let temp = self.fjac[kk] * self.fjac[kj];
                self.fjac[kj] = 0.;
                let j0 = j * self.m;
                for i in 0..=j {
                    self.fjac[k0 + i] += -temp * self.fjac[j0 + i];
                }
            }
            l = k as isize;
        }
        /*
         * Form the full upper triangle of the inverse of (r transpose)*r
         * in the full upper triangle of r
         */
        if l >= 0 {
            let l = l as usize;
            for k in 0..=l {
                let k0 = k * self.m;
                for j in 0..k {
                    let temp = self.fjac[k0 + j];
                    let j0 = j * self.m;
                    for i in 0..=j {
                        self.fjac[j0 + i] += temp * self.fjac[k0 + i];
                    }
                }
                let temp = self.fjac[k0 + k];
                for i in 0..=k {
                    self.fjac[k0 + i] *= temp;
                }
            }
        }
        /*
         * For the full lower triangle of the covariance matrix
         * in the strict lower triangle or and in wa
         */
        for j in 0..self.nfree {
            let jj = self.ipvt[j];
            let sing = j as isize > l;
            let j0 = j * self.m;
            let jj0 = jj * self.m;
            for i in 0..=j {
                let ji = j0 + i;
                if sing {
                    self.fjac[ji] = 0.;
                }
                let ii = self.ipvt[i];
                if ii > jj {
                    self.fjac[jj0 + ii] = self.fjac[ji];
                }
                if ii < jj {
                    self.fjac[ii * self.m + jj] = self.fjac[ji];
                }
            }
            self.wa2[jj] = self.fjac[j0 + j];
        }
        /*
         * Symmetrize the covariance matrix in r
         */
        for j in 0..self.nfree {
            let j0 = j * self.m;
            for i in 0..j {
                self.fjac[j0 + i] = self.fjac[i * self.m + j];
            }
            self.fjac[j0 + j] = self.wa2[j];
        }
        self
    }

    fn rescale(&mut self) {
        if self.cfg.do_user_scale {
            return;
        }
        for j in 0..self.nfree {
            let i = self.ifree[j];
            self.diag[i] = self.diag[i].max(self.wa2[j]);
        }
    }

    // subroutine lmpar
    //
    // given an m by nfree matrix a, an nfree by nfree nonsingular diagonal
    // matrix d, an m-vector b, and a positive number delta,
    // the problem is to determine a value for the parameter
    // par such that if wa1 solves the system
    //
    //	    a*wa1 = b ,	  sqrt(par)*d*wa1 = 0 ,
    //
    // in the least squares sense, and dxnorm is the euclidean
    // norm of d*wa1, then either par is zero and
    //
    //	    (dxnorm-delta) .le. 0.1*delta ,
    //
    // or par is positive and
    //
    //	    abs(dxnorm-delta) .le. 0.1*delta .
    //
    // this subroutine completes the solution of the problem
    // if it is provided with the necessary information from the
    // qr factorization, with column pivoting, of a. that is, if
    // a*p = q*fjack, where p is a permutation matrix, q has orthogonal
    // columns, and fjack is an upper triangular matrix with diagonal
    // elements of nonincreasing magnitude, then lmpar expects
    // the full upper triangle of fjack, the permutation matrix p,
    // and the first nfree components of (q transpose)*b. on output
    // lmpar also provides an upper triangular matrix s such that
    //
    //	     t	 t		     t
    //	    p *(a *a + par*d*d)*p = s *s .
    //
    // s is employed within lmpar and may be of separate interest.
    //
    // only a few iterations are generally needed for convergence
    // of the algorithm. if, however, the limit of 10 iterations
    // is reached, then the output par will contain the best
    // value obtained so far.
    //
    // the subroutine statement is
    //
    //	subroutine lmpar(nfree,fjack,m,ipvt,diag,qtf,delta,par,wa1,wa2,
    //			 wa3,wa4)
    //
    // where
    //
    //	nfree is a positive integer input variable set to the order of fjack.
    //
    //	fjack is an nfree by nfree array. on input the full upper triangle
    //	  must contain the full upper triangle of the matrix fjack.
    //	  on output the full upper triangle is unaltered, and the
    //	  strict lower triangle contains the strict upper triangle
    //	  (transposed) of the upper triangular matrix s.
    //
    //	m is a positive integer input variable not less than nfree
    //	  which specifies the leading dimension of the array fjack.
    //
    //	ipvt is an integer input array of length nfree which defines the
    //	  permutation matrix p such that a*p = q*fjack. column j of p
    //	  is column ipvt(j) of the identity matrix.
    //
    //	diag is an input array of length nfree which must contain the
    //	  diagonal elements of the matrix d.
    //
    //	qtf is an input array of length nfree which must contain the first
    //	  nfree elements of the vector (q transpose)*b.
    //
    //	delta is a positive input variable which specifies an upper
    //	  bound on the euclidean norm of d*wa1.
    //
    //	par is a nonnegative variable. on input par contains an
    //	  initial estimate of the levenberg-marquardt parameter.
    //	  on output par contains the final estimate.
    //
    //	wa1 is an output array of length nfree which contains the least
    //	  squares solution of the system a*wa1 = b, sqrt(par)*d*wa1 = 0,
    //	  for the output par.
    //
    //	wa2 is an output array of length nfree which contains the
    //	  diagonal elements of the upper triangular matrix s.
    //
    //	wa3 and wa4 are work arrays of length nfree.
    //
    // subprograms called
    //
    //	minpack-supplied ... dpmpar,mp_enorm,qrsolv
    //
    //	fortran-supplied ... dabs,mp_dmax1,dmin1,dsqrt
    //
    // argonne national laboratory. minpack project. march 1980.
    // burton s. garbow, kenneth e. hillstrom, jorge j. more
    fn lmpar(&mut self) {
        /*
         *     compute and store in wa1 the gauss-newton direction. if the
         *     jacobian is rank-deficient, obtain a least squares solution.
         */
        let m = self.m;
        let mut nsing = self.nfree;
        let mut jj = 0;
        for j in 0..self.nfree {
            self.wa3[j] = self.qtf[j];
            if self.fjac[jj] == 0. && nsing == self.nfree {
                nsing = j;
            }
            if nsing < self.nfree {
                self.wa3[j] = 0.;
            }
            jj += m + 1;
        }
        for j in (0..nsing).rev() {
            let jj = m * j;
            self.wa3[j] /= self.fjac[jj + j];
            let temp = self.wa3[j];
            for i in 0..j {
                self.wa3[i] -= self.fjac[jj + i] * temp;
            }
        }
        for j in 0..self.nfree {
            self.wa1[self.ipvt[j]] = self.wa3[j];
        }
        /*
         *     initialize the iteration counter.
         *     evaluate the function at the origin, and test
         *     for acceptance of the gauss-newton direction.
         */
        for j in 0..self.nfree {
            self.wa4[j] = self.diag[self.ifree[j]] * self.wa1[j];
        }
        let mut dxnorm = self.wa4[0..self.nfree].enorm();
        let mut fp = dxnorm - self.delta;
        if fp <= 0.1 * self.delta {
            self.par = 0.;
            return;
        }
        /*
         *     if the jacobian is not rank deficient, the newton
         *     step provides a lower bound, parl, for the zero of
         *     the function. otherwise set this bound to zero.
         */
        let mut parl = 0.;
        if nsing >= self.nfree {
            self.newton_correction(dxnorm);
            let mut jj = 0;
            for j in 0..self.nfree {
                let mut sum = 0.0;
                for i in 0..j {
                    sum += self.fjac[jj + i] * self.wa3[i];
                }
                self.wa3[j] = (self.wa3[j] - sum) / self.fjac[jj + j];
                jj += m;
            }
            let temp = self.wa3[0..self.nfree].enorm();
            parl = ((fp / self.delta) / temp) / temp;
        }
        /*
         *     calculate an upper bound, paru, for the zero of the function.
         */
        let mut jj = 0;
        for j in 0..self.nfree {
            let mut sum = 0.0;
            for i in 0..=j {
                sum += self.fjac[jj + i] * self.qtf[i];
            }
            let l = self.ipvt[j];
            self.wa3[j] = sum / self.diag[self.ifree[l]];
            jj += m;
        }
        let gnorm = self.wa3[0..self.nfree].enorm();
        let mut paru = gnorm / self.delta;
        if paru == 0. {
            paru = f64::MIN_POSITIVE / self.delta.min(0.1);
        }
        /*
         *     if the input par lies outside of the interval (parl,paru),
         *     set par to the closer endpoint.
         */
        self.par = self.par.max(parl);
        self.par = self.par.max(paru);
        if self.par == 0. {
            self.par = gnorm / dxnorm;
        }
        let mut iter = 0;
        loop {
            iter += 1;
            if self.par == 0. {
                self.par = f64::MIN_POSITIVE.max(0.001 * paru);
            }
            let temp = self.par.sqrt();
            for j in 0..self.nfree {
                self.wa3[j] = temp * self.diag[self.ifree[j]];
            }
            self.qrsolv();
            for j in 0..self.nfree {
                self.wa4[j] = self.diag[self.ifree[j]] * self.wa1[j];
            }
            dxnorm = self.wa4[0..self.nfree].enorm();
            let temp = fp;
            fp = dxnorm - self.delta;
            /*
             *	 if the function is small enough, accept the current value
             *	 of par. also test for the exceptional cases where parl
             *	 is zero or the number of iterations has reached 10.
             */
            if fp.abs() <= 0.1 * self.delta || (parl == 0. && fp <= temp && temp < 0.) || iter >= 10
            {
                return;
            }
            self.newton_correction(dxnorm);
            jj = 0;
            for j in 0..self.nfree {
                self.wa3[j] /= self.wa2[j];
                let temp = self.wa3[j];
                for i in j + 1..self.nfree {
                    self.wa3[i] -= self.fjac[jj + i] * temp;
                }
                jj += m;
            }
            let temp = self.wa3[0..self.nfree].enorm();
            let parc = ((fp / self.delta) / temp) / temp;
            /*
             *	 depending on the sign of the function, update parl or paru.
             */
            if fp > 0.0 {
                parl = parl.max(self.par);
            }
            if fp < 0.0 {
                paru = paru.min(self.par);
            }
            /*
             *	 compute an improved estimate for par.
             */
            self.par = parl.max(self.par + parc);
        }
    }

    /// compute the newton correction.
    fn newton_correction(&mut self, dxnorm: f64) {
        for j in 0..self.nfree {
            let l = self.ipvt[j];
            self.wa3[j] = self.diag[self.ifree[l]] * (self.wa4[l] / dxnorm);
        }
    }

    // subroutine qrsolv
    //
    // given an m by n matrix a, an n by n diagonal matrix d,
    // and an m-vector b, the problem is to determine an x which
    // solves the system
    //
    //	a*x = b ,	  d*x = 0 ,
    //
    // in the least squares sense.
    //
    // this subroutine completes the solution of the problem
    // if it is provided with the necessary information from the
    // qr factorization, with column pivoting, of a. that is, if
    // a*p = q*r, where p is a permutation matrix, q has orthogonal
    // columns, and r is an upper triangular matrix with diagonal
    // elements of nonincreasing magnitude, then qrsolv expects
    // the full upper triangle of r, the permutation matrix p,
    // and the first n components of (q transpose)*b. the system
    // a*x = b, d*x = 0, is then equivalent to
    //
    //		   t	   t
    //	r*z = q *b ,  p *d*p*z = 0 ,
    //
    // where x = p*z. if this system does not have full rank,
    // then a least squares solution is obtained. on output qrsolv
    // also provides an upper triangular matrix s such that
    //
    //	 t	 t		 t
    //	p *(a *a + d*d)*p = s *s .
    //
    // s is computed within qrsolv and may be of separate interest.
    //
    // the subroutine statement is
    //
    //	subroutine qrsolv(n,r,ldr,ipvt,diag,qtb,x,sdiag,wa)
    //
    // where
    //
    //	n is a positive integer input variable set to the order of r.
    //
    //	r is an n by n array. on input the full upper triangle
    //	  must contain the full upper triangle of the matrix r.
    //	  on output the full upper triangle is unaltered, and the
    //	  strict lower triangle contains the strict upper triangle
    //	  (transposed) of the upper triangular matrix s.
    //
    //	ldr is a positive integer input variable not less than n
    //	  which specifies the leading dimension of the array r.
    //
    //	ipvt is an integer input array of length n which defines the
    //	  permutation matrix p such that a*p = q*r. column j of p
    //	  is column ipvt(j) of the identity matrix.
    //
    //	diag is an input array of length n which must contain the
    //	  diagonal elements of the matrix d.
    //
    //	qtb is an input array of length n which must contain the first
    //	  n elements of the vector (q transpose)*b.
    //
    //	x is an output array of length n which contains the least
    //	  squares solution of the system a*x = b, d*x = 0.
    //
    //	sdiag is an output array of length n which contains the
    //	  diagonal elements of the upper triangular matrix s.
    //
    //	wa is a work array of length n.
    //
    // subprograms called
    //
    //	fortran-supplied ... dabs,dsqrt
    //
    // argonne national laboratory. minpack project. march 1980.
    // burton s. garbow, kenneth e. hillstrom, jorge j. more
    fn qrsolv(&mut self) {
        /*
         *     copy r and (q transpose)*b to preserve input and initialize s.
         *     in particular, save the diagonal elements of r in x.
         */
        let m = self.m;
        let mut kk = 0;
        for j in 0..self.nfree {
            // mirror row j of R into column j
            for i in j..self.nfree {
                self.fjac[m * j + i] = self.fjac[j + m * i];
            }
            self.wa1[j] = self.fjac[kk]; // save R[j,j]
            self.wa4[j] = self.qtf[j];
            kk += m + 1;
        }
        /*
         *     eliminate the diagonal matrix d using a givens rotation.
         */
        for j in 0..self.nfree {
            /*
             *	 prepare the row of d to be eliminated, locating the
             *	 diagonal element using p from the qr factorization.
             */
            let l = self.ipvt[j];
            if self.wa3[l] != 0. {
                self.wa2[j] = self.wa3[l];
                self.wa2[j + 1..self.nfree].fill(0.0);
                /*
                 *	 the transformations to eliminate the row of d
                 *	 modify only a single element of (q transpose)*b
                 *	 beyond the first n, which is initially zero.
                 */
                let mut qtbpj = 0.;
                for k in j..self.nfree {
                    /*
                     *	    determine a givens rotation which eliminates the
                     *	    appropriate element in the current row of d.
                     */
                    if self.wa2[k] == 0. {
                        continue;
                    }
                    let kk = k * (m + 1);
                    let (sinx, cosx) = if self.fjac[kk].abs() < self.wa2[k].abs() {
                        let cotan = self.fjac[kk] / self.wa2[k];
                        let sinx = 0.5 / (0.25 + 0.25 * cotan * cotan).sqrt();
                        let cosx = sinx * cotan;
                        (sinx, cosx)
                    } else {
                        let tanx = self.wa2[k] / self.fjac[kk];
                        let cosx = 0.5 / (0.25 + 0.25 * tanx * tanx).sqrt();
                        let sinx = cosx * tanx;
                        (sinx, cosx)
                    };
                    /*
                     *	    compute the modified diagonal element of r and
                     *	    the modified element of ((q transpose)*b,0).
                     */
                    self.fjac[kk] = cosx * self.fjac[kk] + sinx * self.wa2[k];
                    let temp = cosx * self.wa4[k] + sinx * qtbpj;
                    qtbpj = -sinx * self.wa4[k] + cosx * qtbpj;
                    self.wa4[k] = temp;
                    /*
                     *	    accumulate the tranformation in the row of s.
                     */
                    for i in k + 1..self.nfree {
                        let j = m * k + i;
                        let f = self.fjac[j];
                        let w = self.wa2[i];
                        self.fjac[j] = cosx * f + sinx * w;
                        self.wa2[i] = cosx * w - sinx * f;
                    }
                }
            }
            /*
             *	 store the diagonal element of s and restore
             *	 the corresponding diagonal element of r.
             */
            let kk = j * (m + 1);
            self.wa2[j] = self.fjac[kk];
            self.fjac[kk] = self.wa1[j];
        }
        /*
         *     solve the triangular system for z. if the system is
         *     singular, then obtain a least squares solution.
         */
        let mut nsing = self.nfree;
        for j in 0..self.nfree {
            if self.wa2[j] == 0. && nsing == self.nfree {
                nsing = j;
            }
            if nsing < self.nfree {
                self.wa4[j] = 0.;
            }
        }
        for j in (0..nsing).rev() {
            let mut sum = 0.0;
            for i in j + 1..nsing {
                sum += self.fjac[m * j + i] * self.wa4[i];
            }
            self.wa4[j] = (self.wa4[j] - sum) / self.wa2[j];
        }
        /*
         *     permute the components of z back to components of x.
         */

        for j in 0..self.nfree {
            self.wa1[self.ipvt[j]] = self.wa4[j];
        }
    }

    fn iterate(&mut self, gnorm: f64) -> MPResult<MPDone> {
        for j in 0..self.nfree {
            self.wa1[j] = -self.wa1[j];
        }
        let mut alpha: f64 = 1.0;
        if !self.qanylim {
            /* No parameter limits, so just move to new position WA2 */
            for j in 0..self.nfree {
                self.wa2[j] = self.x[j] + self.wa1[j];
            }
        } else {
            /* Respect the limits.  If a step were to go out of bounds, then
             * we should take a step in the same direction but shorter distance.
             * The step should take us right to the limit in that case.
             */
            for j in 0..self.nfree {
                let lpegged = self.qllim[j] && self.x[j] <= self.llim[j];
                let upegged = self.qulim[j] && self.x[j] >= self.ulim[j];
                let dwa1 = self.wa1[j].abs() > f64::EPSILON;
                if lpegged && self.wa1[j] < 0. {
                    self.wa1[j] = 0.;
                }
                if upegged && self.wa1[j] > 0. {
                    self.wa1[j] = 0.;
                }
                if dwa1 && self.qllim[j] && self.x[j] + self.wa1[j] < self.llim[j] {
                    alpha = alpha.min((self.llim[j] - self.x[j]) / self.wa1[j]);
                }
                if dwa1 && self.qulim[j] && self.x[j] + self.wa1[j] > self.ulim[j] {
                    alpha = alpha.min((self.ulim[j] - self.x[j]) / self.wa1[j]);
                }
            }
            /* Scale the resulting vector, advance to the next position */
            for j in 0..self.nfree {
                self.wa1[j] *= alpha;
                self.wa2[j] = self.x[j] + self.wa1[j];
                /*
                 * Adjust the output values.  If the step put us exactly
                 * on a boundary, make sure it is exact.
                 */
                let sgnu = if self.ulim[j] >= 0. { 1. } else { -1. };
                let sgnl = if self.llim[j] >= 0. { 1. } else { -1. };
                let ulim1 = self.ulim[j] * (1. - sgnu * f64::EPSILON)
                    - if self.ulim[j] == 0. { f64::EPSILON } else { 0. };
                let llim1 = self.llim[j] * (1. + sgnl * f64::EPSILON)
                    + if self.llim[j] == 0. { f64::EPSILON } else { 0. };
                if self.qulim[j] && self.wa2[j] >= ulim1 {
                    self.wa2[j] = self.ulim[j];
                }
                if self.qllim[j] && self.wa2[j] <= llim1 {
                    self.wa2[j] = self.llim[j];
                }
            }
        }
        for j in 0..self.nfree {
            self.wa3[j] = self.diag[self.ifree[j]] * self.wa1[j];
        }
        let pnorm = self.wa3[0..self.nfree].enorm();
        /*
         *	    on the first iteration, adjust the initial step bound.
         */
        if self.iter == 1 {
            self.delta = self.delta.min(pnorm);
        }
        /*
         *	    evaluate the function at x + p and calculate its norm.
         */
        for i in 0..self.nfree {
            self.xnew[self.ifree[i]] = self.wa2[i];
        }
        self.f.eval(&self.xnew, &mut self.wa4)?;
        self.nfev += 1;
        self.fnorm1 = self.wa4[0..self.m].enorm();
        /*
         *	    compute the scaled actual reduction.
         */
        let actred = if 0.1 * self.fnorm1 < self.fnorm {
            let temp = self.fnorm1 / self.fnorm;
            1.0 - temp * temp
        } else {
            -1.0
        };
        /*
         *	    compute the scaled predicted reduction and
         *	    the scaled directional derivative.
         */
        let mut jj = 0;
        for j in 0..self.nfree {
            self.wa3[j] = 0.;
            let l = self.ipvt[j];
            let temp = self.wa1[l];
            let mut ij = jj;
            for i in 0..=j {
                self.wa3[i] += self.fjac[ij] * temp;
                ij += 1;
            }
            jj += self.m;
        }
        /*
         * Remember, alpha is the fraction of the full LM step actually
         * taken
         */
        let temp1 = self.wa3[0..self.nfree].enorm() * alpha / self.fnorm;
        let temp2 = ((alpha * self.par).sqrt() * pnorm) / self.fnorm;
        let temp11 = temp1 * temp1;
        let temp22 = temp2 * temp2;
        let prered = temp11 + temp22 / 0.5;
        let dirder = -(temp11 + temp22);
        /*
         *	    compute the ratio of the actual to the predicted
         *	    reduction.
         */
        let ratio = if prered != 0. { actred / prered } else { 0. };
        /*
         *	    update the step bound.
         */
        if ratio <= 0.25 {
            let mut temp = if actred >= 0. {
                0.5
            } else {
                0.5 * dirder / (dirder + 0.5 * actred)
            };
            if 0.1 * self.fnorm1 >= self.fnorm || temp < 0.1 {
                temp = 0.1;
            }
            self.delta = temp * self.delta.min(pnorm / 0.1);
            self.par /= temp;
        } else {
            if self.par == 0. || ratio >= 0.75 {
                self.delta = pnorm / 0.5;
                self.par *= 0.5;
            }
        }
        /*
         *	    test for successful iteration.
         */
        if ratio >= 1e-4 {
            /*
             *	    successful iteration. update x, fvec, and their norms.
             */
            for j in 0..self.nfree {
                self.x[j] = self.wa2[j];
                self.wa2[j] = self.diag[self.ifree[j]] * self.x[j];
            }
            for i in 0..self.m {
                self.fvec[i] = self.wa4[i];
            }
            self.xnorm = self.wa2[0..self.nfree].enorm();
            self.fnorm = self.fnorm1;
            self.iter += 1;
        }
        /*
         *	    tests for convergence.
         */
        if actred.abs() <= self.cfg.ftol && prered <= self.cfg.ftol && 0.5 * ratio <= 1.0 {
            self.info = MPSuccess::Chi;
        }
        if self.delta <= self.cfg.xtol * self.xnorm {
            self.info = MPSuccess::Par;
        }
        if actred.abs() <= self.cfg.ftol
            && prered <= self.cfg.ftol
            && 0.5 * ratio <= 1.0
            && self.info == MPSuccess::Par
        {
            self.info = MPSuccess::Both;
        }
        if self.info != MPSuccess::NotDone {
            return Ok(MPDone::Exit);
        }
        /*
         *	    tests for termination and stringent tolerances.
         */
        if self.cfg.max_fev > 0 && self.nfev >= self.cfg.max_fev {
            /* Too many function evaluations */
            self.info = MPSuccess::MaxIter;
        }
        if self.iter >= self.cfg.max_iter {
            /* Too many iterations */
            self.info = MPSuccess::MaxIter;
        }
        if actred.abs() <= f64::EPSILON && prered <= f64::EPSILON && 0.5 * ratio <= 1.0 {
            self.info = MPSuccess::Ftol;
        }
        if self.delta <= f64::EPSILON * self.xnorm {
            self.info = MPSuccess::Xtol;
        }
        if gnorm <= f64::EPSILON {
            self.info = MPSuccess::Gtol;
        }
        if self.info != MPSuccess::NotDone {
            return Ok(MPDone::Exit);
        }
        if ratio < 1e-4 {
            Ok(MPDone::Inner)
        } else {
            Ok(MPDone::Outer)
        }
    }

    fn check_config(&self) -> MPResult<()> {
        if self.cfg.ftol <= 0.
            || self.cfg.xtol <= 0.
            || self.cfg.gtol <= 0.
            || self.cfg.step_factor <= 0.
        {
            Err(MPError::Input)
        } else if self.m < self.nfree {
            Err(MPError::DoF)
        } else {
            Ok(())
        }
    }
}

enum MPDone {
    Exit,
    Inner,
    Outer,
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
trait ENorm {
    fn enorm(&self) -> f64;
}

impl ENorm for [f64] {
    fn enorm(&self) -> f64 {
        let mut s1 = 0.;
        let mut s2 = 0.;
        let mut s3 = 0.;
        let mut x1max = 0.;
        let mut x3max = 0.;
        let agiant = MP_RGIANT / self.len() as f64;
        for val in self {
            let xabs = val.abs();
            if xabs > MP_RDWARF && xabs < agiant {
                // sum for intermediate components.
                s2 += xabs * xabs;
            } else if xabs > MP_RDWARF {
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
}

impl fmt::Display for MPError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                MPError::Input => "general input parameter error",
                MPError::Nan => "user function produced non-finite values",
                MPError::Empty => "no user data points were supplied",
                MPError::NoFree => "no free parameters",
                MPError::InitBounds => "initial values inconsistent with constraints",
                MPError::Bounds => "initial constraints inconsistent",
                MPError::DoF => "not enough degrees of freedom",
                MPError::Eval => "error during user evaluation",
            }
        )
    }
}

impl fmt::Debug for MPError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for MPSuccess {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                MPSuccess::NotDone => "unknown error",
                MPSuccess::Chi => "convergence in chi-square value",
                MPSuccess::Par => "convergence in parameter value",
                MPSuccess::Both => "convergence in chi-square and parameter values",
                MPSuccess::Dir => "convergence in orthogonality",
                MPSuccess::MaxIter => "maximum number of iterations reached",
                MPSuccess::Ftol => "ftol is too small; no further improvement",
                MPSuccess::Xtol => "xtol is too small; no further improvement",
                MPSuccess::Gtol => "gtol is too small; no further improvement",
            }
        )
    }
}

impl fmt::Debug for MPSuccess {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[cfg(test)]
mod tests {
    use crate::{MPFitter, MPPar, MPResult, MPSuccess};
    use assert_approx_eq::assert_approx_eq;
    use std::f64::consts::{LN_2, PI};

    #[test]
    fn linear() {
        struct Linear {
            x: Vec<f64>,
            y: Vec<f64>,
            ye: Vec<f64>,
        }

        impl MPFitter for Linear {
            fn eval(&mut self, params: &[f64], deviates: &mut [f64]) -> MPResult<()> {
                for (((d, x), y), ye) in deviates
                    .iter_mut()
                    .zip(self.x.iter())
                    .zip(self.y.iter())
                    .zip(self.ye.iter())
                {
                    let f = params[0] + params[1] * *x;
                    *d = (*y - f) / *ye;
                }
                Ok(())
            }

            fn number_of_points(&self) -> usize {
                self.x.len()
            }
        }
        let mut l = Linear {
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
        let res = l.mpfit(&mut init);
        match res {
            Ok(status) => {
                assert_eq!(status.success, MPSuccess::Chi);
                assert_eq!(status.n_iter, 3);
                assert_eq!(status.n_fev, 8);
                assert_approx_eq!(status.best_norm, 2.75628498);
                assert_approx_eq!(init[0], 3.20996572);
                assert_approx_eq!(init[1], 1.77095420);
                assert_approx_eq!(status.xerror[0], 0.02221018);
                assert_approx_eq!(status.xerror[1], 0.01893756);
            }
            Err(err) => {
                panic!("Error in Linear fit: {}", err);
            }
        }
    }

    #[test]
    fn quad() {
        struct Quad {
            x: Vec<f64>,
            y: Vec<f64>,
            ye: Vec<f64>,
            params: Option<[MPPar; 3]>,
        }

        impl MPFitter for Quad {
            fn eval(&mut self, params: &[f64], deviates: &mut [f64]) -> MPResult<()> {
                for (((d, x), y), ye) in deviates
                    .iter_mut()
                    .zip(self.x.iter())
                    .zip(self.y.iter())
                    .zip(self.ye.iter())
                {
                    let x = *x;
                    let f = params[0] + params[1] * x + params[2] * x * x;
                    *d = (*y - f) / *ye;
                }
                Ok(())
            }

            fn number_of_points(&self) -> usize {
                self.x.len()
            }

            fn parameters(&self) -> &[MPPar] {
                match &self.params {
                    None => &[],
                    Some(p) => p,
                }
            }
        }
        let mut l = Quad {
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
                2.3095947E+01,
                2.6449392E+01,
                1.0204468E+01,
                5.40507,
                1.5787588E+01,
                1.6520903E+01,
                1.5971818E+01,
                4.7668524E+00,
                4.9337711E+00,
                8.7348375E+00,
            ],
            ye: vec![0.2; 10],
            params: None,
        };
        let mut init = [1., 1., 1.];
        let res = l.mpfit(&mut init);
        match res {
            Ok(status) => {
                assert_eq!(status.success, MPSuccess::Chi);
                assert_eq!(status.n_iter, 3);
                assert_eq!(status.n_fev, 10);
                assert_approx_eq!(status.best_norm, 5.67932273);
                assert_approx_eq!(init[0], 4.70382909);
                assert_approx_eq!(init[1], 0.06258629);
                assert_approx_eq!(init[2], 6.16308723);
                assert_approx_eq!(status.xerror[0], 0.09751164);
                assert_approx_eq!(status.xerror[1], 0.05480195);
                assert_approx_eq!(status.xerror[2], 0.05443275);
            }
            Err(err) => {
                panic!("Error in Quad fit: {}", err);
            }
        }
        l.params = Some([
            MPPar::default(),
            MPPar {
                fixed: true,
                ..MPPar::new()
            },
            MPPar::default(),
        ]);
        let mut init = [1., 0., 1.];
        let res = l.mpfit(&mut init);
        match res {
            Ok(status) => {
                assert_eq!(status.success, MPSuccess::Chi);
                assert_eq!(status.n_iter, 3);
                assert_eq!(status.n_fev, 8);
                assert_approx_eq!(status.best_norm, 6.98358800);
                assert_approx_eq!(init[0], 4.69625430);
                assert_approx_eq!(init[1], 0.00000000);
                assert_approx_eq!(init[2], 6.17295360);
                assert_approx_eq!(status.xerror[0], 0.09728581);
                assert_approx_eq!(status.xerror[1], 0.00000000);
                assert_approx_eq!(status.xerror[2], 0.05374279);
            }
            Err(err) => {
                panic!("Error in Quad fixed fit: {}", err);
            }
        }
    }

    #[test]
    fn gaussian() {
        struct Gaussian {
            x: Vec<f64>,
            y: Vec<f64>,
            ye: Vec<f64>,
            pars: Option<[MPPar; 4]>,
        }

        impl MPFitter for Gaussian {
            fn eval(&mut self, params: &[f64], deviates: &mut [f64]) -> MPResult<()> {
                let sig2 = params[3] * params[3];
                for (((d, x), y), ye) in deviates
                    .iter_mut()
                    .zip(self.x.iter())
                    .zip(self.y.iter())
                    .zip(self.ye.iter())
                {
                    let xc = *x - params[2];
                    let f = params[1] * (-0.5 * xc * xc / sig2).exp() + params[0];
                    *d = (*y - f) / *ye;
                }
                Ok(())
            }

            fn number_of_points(&self) -> usize {
                self.x.len()
            }

            fn parameters(&self) -> &[MPPar] {
                match &self.pars {
                    None => &[],
                    Some(p) => p,
                }
            }
        }
        let mut l = Gaussian {
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
                -4.4494256E-02,
                8.7324673E-01,
                7.4443483E-01,
                4.7631559E+00,
                1.7187297E-01,
                1.1639182E-01,
                1.5646480E+00,
                5.2322268E+00,
                4.2543168E+00,
                6.2792623E-01,
            ],
            ye: vec![0.5; 10],
            pars: None,
        };
        let mut init = [0., 1., 1., 1.];
        let res = l.mpfit(&mut init);
        match res {
            Ok(status) => {
                assert_eq!(status.success, MPSuccess::Chi);
                assert_eq!(status.n_iter, 27);
                assert_eq!(status.n_fev, 134);
                assert_approx_eq!(status.best_norm, 10.35003196);
                assert_approx_eq!(init[0], 0.48044336);
                assert_approx_eq!(init[1], 4.55075247);
                assert_approx_eq!(init[2], -0.06256246);
                assert_approx_eq!(init[3], 0.39747174);
                assert_approx_eq!(status.xerror[0], 0.23223493);
                assert_approx_eq!(status.xerror[1], 0.39543448);
                assert_approx_eq!(status.xerror[2], 0.07471491);
                assert_approx_eq!(status.xerror[3], 0.08999568);
            }
            Err(err) => {
                panic!("Error in Quad fit: {}", err);
            }
        }
        let mut init = [0., 1., 0., 0.1];
        l.pars = Some([
            MPPar {
                fixed: true,
                ..MPPar::new()
            },
            MPPar::default(),
            MPPar {
                fixed: true,
                ..MPPar::new()
            },
            MPPar::default(),
        ]);
        let res = l.mpfit(&mut init);
        match res {
            Ok(status) => {
                assert_eq!(status.success, MPSuccess::Chi);
                assert_eq!(status.n_iter, 12);
                assert_eq!(status.n_fev, 35);
                assert_approx_eq!(status.best_norm, 15.51613428);
                assert_approx_eq!(init[0], 0.00000000);
                assert_approx_eq!(init[1], 5.05924391);
                assert_approx_eq!(init[2], 0.00000000);
                assert_approx_eq!(init[3], 0.47974647);
                assert_approx_eq!(status.xerror[0], 0.00000000);
                assert_approx_eq!(status.xerror[1], 0.32930696);
                assert_approx_eq!(status.xerror[2], 0.00000000);
                assert_approx_eq!(status.xerror[3], 0.05380360);
            }
            Err(err) => {
                panic!("Error in Quad fit: {}", err);
            }
        }
    }

    fn gauss(x: f64, xc: f64, w: f64) -> f64 {
        (4. * LN_2).sqrt() / (PI.sqrt() * w) * (-4. * LN_2 / w.powi(2) * (x - xc).powi(2)).exp()
    }

    fn lorentz(x: f64, xc: f64, w: f64) -> f64 {
        2. / PI * w / (4. * (x - xc).powi(2) + w.powi(2))
    }

    fn pseudovoigt(x: f64, p: &[f64]) -> f64 {
        let xc = p[0];
        let w = p[1];
        let a = p[2];
        let y0 = p[3];
        let mu = p[4];
        let g = gauss(x, xc, w);
        let l = lorentz(x, xc, w);
        y0 + a * (mu * l + (1. - mu) * g)
    }

    struct Pseudovoigt {
        x: Vec<f64>,
        y: Vec<f64>,
        ye: Vec<f64>,
    }

    const PARS: &[MPPar; 5] = &[
        MPPar::new(),
        MPPar::new(),
        MPPar::new(),
        MPPar::new(),
        MPPar {
            limited_low: true,
            limited_up: true,
            limit_low: 0.0,
            limit_up: 1.0,
            ..MPPar::new()
        },
    ];

    impl Pseudovoigt {
        const fn pars() -> &'static [MPPar] {
            PARS
        }
    }

    impl MPFitter for Pseudovoigt {
        fn eval(&mut self, params: &[f64], deviates: &mut [f64]) -> MPResult<()> {
            for (((d, x), y), ye) in deviates
                .iter_mut()
                .zip(self.x.iter())
                .zip(self.y.iter())
                .zip(self.ye.iter())
            {
                let x = *x;
                let y = *y;
                let ye = *ye;
                let pv = pseudovoigt(x, params);
                let resid = (y - pv) / ye;
                *d = resid;
            }
            Ok(())
        }

        fn number_of_points(&self) -> usize {
            self.x.len()
        }

        fn parameters(&self) -> &[MPPar] {
            Self::pars()
        }
    }

    #[test]
    fn test_pseudovoigt() {
        let mut l = Pseudovoigt {
            x: vec![
                45.48130544450339,
                45.49617104593113,
                45.511036647358864,
                45.5259022487866,
                45.54076785021434,
                45.55563345164207,
                45.57049905306981,
                45.585364654497546,
                45.60023025592528,
                45.615095857353026,
                45.629961458780755,
                45.64482706020849,
                45.65969266163623,
                45.674558263063965,
                45.68942386449171,
                45.704289465919445,
                45.71915506734718,
                45.73402066877491,
                45.74888627020265,
                45.76375187163039,
                45.77861747305813,
                45.79348307448586,
                45.8083486759136,
                45.823214277341336,
                45.83807987876907,
                45.85294548019681,
                45.867811081624545,
                45.88267668305228,
                45.89754228448002,
                45.91240788590776,
                45.9272734873355,
                45.94213908876323,
                45.95700469019096,
                45.9718702916187,
                45.98673589304644,
                46.00160149447418,
                46.016467095901916,
                46.03133269732965,
                46.04619829875738,
                46.061063900185125,
                46.07592950161286,
                46.0907951030406,
                46.105660704468335,
                46.12052630589607,
                46.135391907323815,
                46.150257508751544,
                46.16512311017928,
                46.17998871160702,
                46.19485431303475,
                46.2097199144625,
                46.22458551589023,
                46.23945111731797,
                46.2543167187457,
                46.269182320173435,
                46.28404792160118,
                46.298913523028915,
                46.31377912445665,
                46.32864472588439,
                46.343510327312124,
                46.35837592873986,
                46.3732415301676,
                46.38810713159533,
                46.40297273302307,
                46.417838334450806,
                46.43270393587855,
                46.447569537306286,
                46.462435138734016,
            ],
            y: vec![
                782.9381965112784,
                785.9953096826335,
                783.502095047636,
                781.8478754078232,
                786.5586751999884,
                790.6722286020803,
                795.62248764412,
                795.097884130258,
                799.194201620961,
                808.4468792234753,
                811.0505980447331,
                809.8543648061078,
                813.3515498973136,
                816.6842223486614,
                818.4962324229795,
                824.9028803637333,
                834.069696303739,
                841.6539793557772,
                853.5715299785493,
                869.7538160514533,
                877.395076590247,
                889.6775409243694,
                909.5194442162739,
                937.5729137263957,
                977.3289837738814,
                1020.6997653554964,
                1086.3643444254128,
                1194.694799707516,
                1364.637343902714,
                1667.2254730749685,
                2299.913139055621,
                3728.2971104942458,
                6538.224597833223,
                10726.924311797535,
                14063.85433952567,
                14146.859802962677,
                11294.095996910199,
                7482.043816631519,
                4398.290451186299,
                2670.3413870183867,
                1859.1848310024075,
                1481.3978814815955,
                1279.282063387968,
                1151.3201119770513,
                1063.148522287353,
                1010.3296031455463,
                980.2632514947052,
                949.8782908568008,
                920.585197863486,
                891.6124281566953,
                880.6980614269305,
                872.5485462506051,
                858.3331153524963,
                849.5566888279196,
                839.3413923545357,
                833.8182111395416,
                829.9830591235499,
                827.4737256563571,
                829.5348938252345,
                822.4325715230892,
                820.9871700287264,
                818.1349164141059,
                818.7359717234702,
                818.9869745651724,
                815.624564738269,
                814.4356077460651,
                813.3298216118234,
            ],
            ye: vec![
                0.8434045362508102,
                0.8440282564506858,
                0.8415629781020887,
                0.8394082618068363,
                0.8411119905429378,
                0.8610586993450439,
                0.8780963697576099,
                0.8875028631142169,
                0.8975136641339376,
                0.909194544030465,
                0.9159901617993966,
                0.9201542731914041,
                0.9269623051929742,
                0.9336228997573229,
                0.9390176080723663,
                0.946835903636598,
                0.955824154845803,
                0.9641784951989889,
                0.9746094160965306,
                0.9874393099280484,
                0.9957774299748927,
                1.0075898793919262,
                1.023645892682499,
                1.0441327342197362,
                1.0708149461644139,
                1.0991060019865837,
                1.1391041698306787,
                1.1866600916485794,
                1.2356391252495653,
                1.3493735660002284,
                1.5725251282178219,
                1.9883901352011322,
                2.618342185198151,
                3.3290125974774036,
                3.7844701144175716,
                3.7811871432006514,
                3.3680967716707126,
                2.733920132004131,
                2.0903354213623446,
                1.6248939190613658,
                1.3528790030972153,
                1.2051933083891002,
                1.1176711112964255,
                1.0583440303729854,
                1.0153282477822558,
                0.9882173099333285,
                0.9718860648687835,
                0.9552458101160581,
                0.9391281078884468,
                0.9230110939050422,
                0.9162172417842753,
                0.9106644040198394,
                0.902218619582364,
                0.8965877993506405,
                0.8910110109445686,
                0.8896058320357073,
                0.8877747948959651,
                0.8857705188656704,
                0.8860369000321661,
                0.8818259233257201,
                0.8802658858092582,
                0.8778390619432762,
                0.8769921888256682,
                0.8764039739284314,
                0.8737460539494736,
                0.872356022121994,
                0.8711681408643424,
            ],
        };
        let mut init = [
            45.98749603354855,
            0.18935719230294046,
            14146.859802962677,
            781.8478754078232,
            0.5,
        ];
        let res = l.mpfit(&mut init);
        match res {
            Ok(status) => {
                assert_eq!(status.success, MPSuccess::Chi);
                assert_eq!(status.n_iter, 12);
                assert_eq!(status.n_fev, 69);
                assert_approx_eq!(status.best_norm, 37480.11190046);
                assert_approx_eq!(init[0], 45.99597613);
                assert_approx_eq!(init[1], 0.06848724);
                assert_approx_eq!(init[2], 1200.62523271);
                assert_approx_eq!(init[3], 763.71495089);
                assert_approx_eq!(init[4], 0.47813424);
                assert_approx_eq!(status.xerror[0], 0.00000456);
                assert_approx_eq!(status.xerror[1], 0.00001329);
                assert_approx_eq!(status.xerror[2], 0.19927477);
                assert_approx_eq!(status.xerror[3], 0.16325942);
                assert_approx_eq!(status.xerror[4], 0.00041317);
            }
            Err(err) => {
                panic!("Error in Pseudovoigt fit: {}", err);
            }
        };
    }

    #[test]
    fn test_pseudovoigt1() {
        let mut l = Pseudovoigt {
            x: vec![
                52.837_083_463_783_88,
                52.849_611_521_510_916,
                52.862_139_579_237_954,
                52.874_667_636_964_99,
                52.887_195_694_692_03,
                52.899_723_752_419_07,
                52.912_251_810_146_11,
                52.924_779_867_873_15,
                52.937_307_925_600_19,
                52.949_835_983_327_226,
                52.962_364_041_054_265,
                52.974_892_098_781_304,
                52.987_420_156_508_34,
                52.999_948_214_235_38,
                53.012_476_271_962_42,
                53.025_004_329_689_46,
                53.037_532_387_416_5,
                53.050_060_445_143_54,
                53.062_588_502_870_575,
                53.075_116_560_597_614,
                53.087_644_618_324_65,
                53.100_172_676_051_69,
                53.112_700_733_778_73,
                53.125_228_791_505_77,
                53.137_756_849_232_81,
                53.150_284_906_959_85,
                53.162_812_964_686_886,
                53.175_341_022_413_924,
                53.187_869_080_140_96,
                53.200_397_137_868,
                53.212_925_195_595_04,
                53.225_453_253_322_08,
                53.237_981_311_049_12,
                53.250_509_368_776_16,
                53.263_037_426_503_196,
                53.275_565_484_230_235,
                53.288_093_541_957_274,
                53.300_621_599_684_31,
                53.313_149_657_411_35,
                53.325_677_715_138_39,
                53.338_205_772_865_43,
                53.350_733_830_592_47,
                53.363_261_888_319_51,
                53.375_789_946_046_545,
                53.388_318_003_773_584,
                53.400_846_061_500_62,
                53.413_374_119_227_66,
                53.425_902_176_954_7,
                53.438_430_234_681_74,
                53.450_958_292_408_78,
                53.463_486_350_135_82,
                53.476_014_407_862_856,
                53.488_542_465_589_894,
                53.501_070_523_316_93,
                53.513_598_581_043_97,
                53.526_126_638_771_01,
                53.538_654_696_498_05,
                53.551_182_754_225_09,
                53.563_710_811_952_13,
                53.576_238_869_679_166,
                53.588_766_927_406_205,
                53.601_294_985_133_244,
                53.613_823_042_860_28,
                53.626_351_100_587_32,
                53.638_879_158_314_36,
                53.651_407_216_041_4,
                53.663_935_273_768_44,
                53.676_463_331_495_476,
                53.688_991_389_222_515,
                53.701_519_446_949_554,
                53.714_047_504_676_59,
                53.726_575_562_403_63,
                53.739_103_620_130_67,
                53.751_631_677_857_71,
                53.764_159_735_584_75,
                53.776_687_793_311_79,
                53.789_215_851_038_826,
                53.801_743_908_765_864,
                53.814_271_966_492_9,
                53.826_800_024_219_94,
            ],
            y: vec![
                178.987_438_648_311_44,
                178.601_925_441_924_15,
                178.230_843_586_504_62,
                177.951_978_186_375_67,
                177.616_624_546_348_2,
                177.476_413_570_285_13,
                177.098_221_281_638_25,
                177.461_479_528_291_96,
                177.193_478_559_630_45,
                176.316_354_701_232_65,
                175.887_710_193_495_05,
                177.046_517_369_018_86,
                177.685_708_586_221_1,
                177.848_235_305_765_74,
                176.599_838_197_142_53,
                176.913_271_582_268_83,
                178.195_994_358_753_18,
                177.996_352_432_245_1,
                177.776_574_261_242_53,
                177.709_455_098_463_88,
                177.335_390_278_114_8,
                178.237_816_532_019_45,
                179.893_256_663_941_37,
                179.372_389_563_483_38,
                178.910_598_459_418_4,
                178.991_938_452_104_68,
                178.636_643_460_850_3,
                178.753_525_107_775_6,
                178.639_327_989_735_88,
                178.721_324_722_853_18,
                179.038_981_909_383_35,
                179.557_238_252_202_7,
                179.010_346_249_775_07,
                178.934_505_913_512_6,
                179.630_672_240_794_87,
                179.289_057_480_863_62,
                179.422_304_633_534_42,
                179.938_579_801_727_43,
                179.623_286_830_184_04,
                179.842_319_902_770_9,
                180.487_975_565_693_5,
                181.115_942_419_818_42,
                182.453_531_267_354_3,
                183.687_781_083_862_43,
                183.602_223_085_692_46,
                184.263_041_408_326_8,
                184.869_698_374_413_1,
                185.509_340_245_788_54,
                185.629_155_590_496_2,
                186.051_331_030_921_46,
                188.354_820_023_733_5,
                191.480_072_818_505_53,
                195.485_580_967_991_75,
                202.783_975_875_700_52,
                225.292_920_077_447_18,
                281.547_773_616_094,
                391.189_239_304_060_8,
                554.248_726_887_628_1,
                749.535_832_853_957_3,
                969.841_764_419_908,
                1_201.463_538_714_477_2,
                1_370.878_308_475_559_2,
                1_326.971_558_082_223,
                1_132.042_002_274_113_9,
                900.739_067_432_381,
                681.163_346_395_108,
                473.900_562_579_887_4,
                319.451_987_037_619_86,
                230.924_951_465_948_5,
                204.584_443_092_548_55,
                195.699_343_901_824_3,
                191.817_891_458_470_06,
                190.112_059_297_907_7,
                188.270_495_536_016_74,
                186.351_795_226_883_2,
                184.918_104_477_306_54,
                185.054_910_029_775_8,
                184.733_798_314_174_72,
                183.168_455_209_723_53,
                182.505_556_446_722_24,
            ],
            ye: vec![
                0.5170943934449542,
                0.5179997395349097,
                0.5190267737147933,
                0.5201979316997363,
                0.521_044_862_119_817,
                0.5220782217580224,
                0.5227562087267094,
                0.5246175285309543,
                0.5255535768308761,
                0.5254782999575789,
                0.5253802404836828,
                0.515_446_596_362_622,
                0.5092425903709408,
                0.5045145103803138,
                0.498_637_076_201_441,
                0.4961887439899552,
                0.4953786487707595,
                0.4931381814159416,
                0.490_987_392_879_383,
                0.4892996776568037,
                0.4875593352897711,
                0.4874550737214612,
                0.4887125244539284,
                0.4867985144169186,
                0.4852581169476606,
                0.4844860975749162,
                0.4830456365779631,
                0.4824865878565433,
                0.4814730317192217,
                0.4809268169116051,
                0.4805948353610307,
                0.4806023850469225,
                0.4792644316303579,
                0.4784282303922414,
                0.4784070425690038,
                0.4768118372258096,
                0.4762821089726825,
                0.4763157887530701,
                0.475_450_326_606_755,
                0.475_189_132_683_48,
                0.4756012511808683,
                0.476_047_255_929_986,
                0.4776247525492051,
                0.4794773782719995,
                0.4790699236418459,
                0.4795103303506842,
                0.4799589579208453,
                0.4804261951429987,
                0.4802690157615006,
                0.4801150680749974,
                0.4828229112980472,
                0.4864291570457189,
                0.4910561363817306,
                0.4998681263598743,
                0.5263212774722177,
                0.5881269105998965,
                0.6925030118847442,
                0.8237322085714922,
                0.9570134673509345,
                1.0868526047365892,
                1.2086214645578408,
                1.2893034358985047,
                1.2681065476159368,
                1.1711902860779513,
                1.044_156_476_195_068,
                0.9077748898292016,
                0.755_980_570_139_086,
                0.6202041050537231,
                0.5266842939659873,
                0.4950926046898963,
                0.4839612205334828,
                0.4783952106180395,
                0.4759508695218888,
                0.4731293401860461,
                0.4701709517704612,
                0.4680880353549415,
                0.4675371876039536,
                0.4668807647290181,
                0.4644066448749752,
                0.4630597776058855,
            ],
        };
        let mut init = [
            53.601294985133244,
            0.314_574_909_088_767_2,
            1194.9905982820642,
            175.88771019349505,
            0.5,
        ];
        let res = l.mpfit(&mut init);
        match res {
            Ok(status) => {
                assert_approx_eq!(init[0], 53.605029);
                assert_approx_eq!(init[1], 0.081662);
                assert_approx_eq!(init[2], 105.675544);
                assert_approx_eq!(init[3], 178.305680);
                assert_approx_eq!(init[4], 0.113177);
                assert_approx_eq!(status.xerror[0], 0.000018);
                assert_approx_eq!(status.xerror[1], 0.000046);
                assert_approx_eq!(status.xerror[2], 0.081090);
                assert_approx_eq!(status.xerror[3], 0.079527);
                assert_approx_eq!(status.xerror[4], 0.002053);
            }
            Err(err) => {
                panic!("Error in Pseudovoigt fit: {}", err);
            }
        };
    }

    #[test]
    fn lin_sided() {
        use crate::MPSide;

        struct Linear {
            x: Vec<f64>,
            y: Vec<f64>,
            ye: Vec<f64>,
            pars: [MPPar; 2],
        }

        impl MPFitter for Linear {
            fn eval(&mut self, params: &[f64], deviates: &mut [f64]) -> MPResult<()> {
                for (((d, x), y), ye) in deviates
                    .iter_mut()
                    .zip(self.x.iter())
                    .zip(self.y.iter())
                    .zip(self.ye.iter())
                {
                    *d = (*y - (params[0] + params[1] * *x)) / *ye;
                }
                Ok(())
            }

            fn number_of_points(&self) -> usize {
                self.x.len()
            }

            fn parameters(&self) -> &[MPPar] {
                &self.pars
            }
        }

        let mut l = Linear {
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
            pars: [
                MPPar {
                    side: MPSide::Both,
                    ..MPPar::new()
                },
                MPPar {
                    side: MPSide::Both,
                    ..MPPar::new()
                },
            ],
        };
        let mut init = [1., 1.];
        let res = l.mpfit(&mut init);
        match res {
            Ok(status) => {
                assert_eq!(status.success, MPSuccess::Chi);
                assert_eq!(status.n_iter, 3);
                assert_eq!(status.n_fev, 12);
                assert_approx_eq!(status.best_norm, 2.75628498);
                assert_approx_eq!(init[0], 3.20996572);
                assert_approx_eq!(init[1], 1.77095420);
                assert_approx_eq!(status.xerror[0], 0.02221018);
                assert_approx_eq!(status.xerror[1], 0.01893756);
            }
            Err(err) => {
                panic!("Error in lin_sided fit: {}", err);
            }
        }
    }

    #[test]
    fn gauss_analytical() {
        use crate::MPSide;

        struct Gaussian {
            x: Vec<f64>,
            y: Vec<f64>,
            ye: Vec<f64>,
            pars: [MPPar; 4],
        }

        impl MPFitter for Gaussian {
            fn eval(&mut self, params: &[f64], deviates: &mut [f64]) -> MPResult<()> {
                let sig2 = params[3] * params[3];
                for (((d, x), y), ye) in deviates
                    .iter_mut()
                    .zip(self.x.iter())
                    .zip(self.y.iter())
                    .zip(self.ye.iter())
                {
                    let xc = *x - params[2];
                    let f = params[1] * (-0.5 * xc * xc / sig2).exp() + params[0];
                    *d = (*y - f) / *ye;
                }
                Ok(())
            }

            fn number_of_points(&self) -> usize {
                self.x.len()
            }

            fn parameters(&self) -> &[MPPar] {
                &self.pars
            }

            fn jacobian(
                &mut self,
                params: &[f64],
                deviates: &mut [f64],
                derivs: &mut [Option<Vec<f64>>],
            ) -> MPResult<()> {
                let sig2 = params[3] * params[3];
                let sig = params[3];
                for (i, (((d, x), y), ye)) in deviates
                    .iter_mut()
                    .zip(self.x.iter())
                    .zip(self.y.iter())
                    .zip(self.ye.iter())
                    .enumerate()
                {
                    let xc = *x - params[2];
                    let e = (-0.5 * xc * xc / sig2).exp();
                    let f = params[1] * e + params[0];
                    *d = (*y - f) / *ye;
                    if let Some(col) = &mut derivs[0] {
                        col[i] = -1.0 / *ye;
                    }
                    if let Some(col) = &mut derivs[1] {
                        col[i] = -e / *ye;
                    }
                    if let Some(col) = &mut derivs[2] {
                        col[i] = -params[1] * e * xc / (sig2 * *ye);
                    }
                    if let Some(col) = &mut derivs[3] {
                        col[i] = -params[1] * e * xc * xc / (sig2 * sig * *ye);
                    }
                }
                Ok(())
            }
        }

        let mut l = Gaussian {
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
                -4.4494256E-02,
                8.7324673E-01,
                7.4443483E-01,
                4.7631559E+00,
                1.7187297E-01,
                1.1639182E-01,
                1.5646480E+00,
                5.2322268E+00,
                4.2543168E+00,
                6.2792623E-01,
            ],
            ye: vec![0.5; 10],
            pars: [
                MPPar {
                    side: MPSide::User,
                    ..MPPar::new()
                },
                MPPar {
                    side: MPSide::User,
                    ..MPPar::new()
                },
                MPPar {
                    side: MPSide::User,
                    ..MPPar::new()
                },
                MPPar {
                    side: MPSide::User,
                    ..MPPar::new()
                },
            ],
        };
        let mut init = [0., 1., 1., 1.];
        let res = l.mpfit(&mut init);
        match res {
            Ok(status) => {
                assert_eq!(status.success, MPSuccess::Chi);
                assert_eq!(status.n_iter, 27);
                assert_eq!(status.n_fev, 56);
                assert_approx_eq!(status.best_norm, 10.35003196);
                assert_approx_eq!(init[0], 0.48044336);
                assert_approx_eq!(init[1], 4.55075247);
                assert_approx_eq!(init[2], -0.06256246);
                assert_approx_eq!(init[3], 0.39747174);
                assert_approx_eq!(status.xerror[0], 0.23223493);
                assert_approx_eq!(status.xerror[1], 0.39543448);
                assert_approx_eq!(status.xerror[2], 0.07471491);
                assert_approx_eq!(status.xerror[3], 0.08999568);
            }
            Err(err) => {
                panic!("Error in gauss_analytical fit: {}", err);
            }
        }
    }
}
