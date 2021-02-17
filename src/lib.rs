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
#[derive(Copy, Clone)]
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
    /// Finite derivative step size                Default: f64::EPSILON
    pub epsfcn: f64,
    /// Initial step bound                         Default: 100.0
    pub step_factor: f64,
    /// Range tolerance for covariance calculation Default: 1e-14
    pub covtol: f64,
    /// Maximum number of iterations.  If maxiter == 0,
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
            epsfcn: f64::EPSILON,
            step_factor: 100.0,
            covtol: 1e-14,
            max_iter: 0,
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
    /// Error
    Error,
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
    fn eval(&self, params: &[f64], deviates: &mut [f64]);

    fn number_of_points(&self) -> usize;
}

/// (f64::MIN_POSITIVE * 1.5).sqrt() * 10
const MP_RDWARF: f64 = 1.8269129289596699e-153;
/// f64::MAX.sqrt() * 0.1
const MP_RGIANT: f64 = 1.3407807799935083e+153;

struct MPFit<'a, F: MPFitter> {
    m: usize,
    npar: usize,
    nfree: usize,
    ifree: Vec<usize>,
    fvec: Vec<f64>,
    nfev: usize,
    xnew: Vec<f64>,
    x: Vec<f64>,
    xall: &'a [f64],
    qtf: Vec<f64>,
    fjack: Vec<f64>,
    side: Vec<MPSide>,
    step: Vec<f64>,
    dstep: Vec<f64>,
    qllim: Vec<bool>,
    qulim: Vec<bool>,
    llim: Vec<f64>,
    ulim: Vec<f64>,
    qanylim: bool,
    f: &'a F,
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
}

impl<'a, F: MPFitter> MPFit<'a, F> {
    fn new(m: usize, xall: &'a [f64], f: &'a F) -> Option<MPFit<'a, F>> {
        let npar = xall.len();
        if m == 0 {
            None
        } else {
            Some(MPFit {
                m,
                npar,
                nfree: 0,
                ifree: vec![],
                fvec: vec![0.; m],
                nfev: 1,
                xnew: vec![0.; npar],
                x: vec![],
                xall: &xall,
                qtf: vec![],
                fjack: vec![],
                side: vec![],
                step: vec![],
                dstep: vec![],
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
                info: MPSuccess::Error,
            })
        }
    }

    ///     subroutine fdjac2
    ///
    ///     this subroutine computes a forward-difference approximation
    ///     to the m by n jacobian matrix associated with a specified
    ///     problem of m functions in n variables.
    ///
    ///     the subroutine statement is
    ///
    ///	subroutine fdjac2(fcn,m,n,x,fvec,fjac,ldfjac,iflag,epsfcn,wa)
    ///
    ///     where
    ///
    ///	fcn is the name of the user-supplied subroutine which
    ///	  calculates the functions. fcn must be declared
    ///	  in an external statement in the user calling
    ///	  program, and should be written as follows.
    ///
    ///	  subroutine fcn(m,n,x,fvec,iflag)
    ///	  integer m,n,iflag
    ///	  double precision x(n),fvec(m)
    ///	  ----------
    ///	  calculate the functions at x and
    ///	  return this vector in fvec.
    ///	  ----------
    ///	  return
    ///	  end
    ///
    ///	  the value of iflag should not be changed by fcn unless
    ///	  the user wants to terminate execution of fdjac2.
    ///	  in this case set iflag to a negative integer.
    ///
    ///	m is a positive integer input variable set to the number
    ///	  of functions.
    ///
    ///	n is a positive integer input variable set to the number
    ///	  of variables. n must not exceed m.
    ///
    ///	x is an input array of length n.
    ///
    ///	fvec is an input array of length m which must contain the
    ///	  functions evaluated at x.
    ///
    ///	fjac is an output m by n array which contains the
    ///	  approximation to the jacobian matrix evaluated at x.
    ///
    ///	ldfjac is a positive integer input variable not less than m
    ///	  which specifies the leading dimension of the array fjac.
    ///
    ///	iflag is an integer variable which can be used to terminate
    ///	  the execution of fdjac2. see description of fcn.
    ///
    ///	epsfcn is an input variable used in determining a suitable
    ///	  step length for the forward-difference approximation. this
    ///	  approximation assumes that the relative errors in the
    ///	  functions are of the order of epsfcn. if epsfcn is less
    ///	  than the machine precision, it is assumed that the relative
    ///	  errors in the functions are of the order of the machine
    ///	  precision.
    ///
    ///	wa is a work array of length m.
    ///
    ///     subprograms called
    ///
    ///	user-supplied ...... fcn
    ///
    ///	minpack-supplied ... dpmpar
    ///
    ///	fortran-supplied ... dabs,dmax1,dsqrt
    ///
    ///     argonne national laboratory. minpack project. march 1980.
    ///     burton s. garbow, kenneth e. hillstrom, jorge j. more
    ///
    fn fdjack2(&mut self, config: &MPConfig) {
        let eps = config.epsfcn.max(f64::EPSILON).sqrt();
        // TODO: sides are not going to be used, probably clean them up after
        // TODO: probably analytical derivatives should be implemented at some point
        let mut ij = 0;
        for j in 0..self.nfree {
            let free_p = self.ifree[j];
            let temp = self.x[free_p];
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
            if j < self.qulim.len()
                && self.qulim[j]
                && j < self.ulim.len()
                && temp > self.ulim[j] - h
            {
                h = -h;
            }
            self.x[self.ifree[j]] = temp + h;
            self.f.eval(&self.x, &mut self.wa4);
            self.nfev += 1;
            self.x[self.ifree[j]] = temp;
            for i in 0..self.m {
                self.fjack[ij] = (self.wa4[i] - self.fvec[i]) / h;
                ij += 1;
            }
        }
    }

    ///     subroutine qrfac
    ///
    ///     this subroutine uses householder transformations with column
    ///     pivoting (optional) to compute a qr factorization of the
    ///     m by n matrix a. that is, qrfac determines an orthogonal
    ///     matrix q, a permutation matrix p, and an upper trapezoidal
    ///     matrix r with diagonal elements of nonincreasing magnitude,
    ///     such that a*p = q*r. the householder transformation for
    ///     column k, k = 1,2,...,min(m,n), is of the form
    ///
    ///			    t
    ///	    i - (1/u(k))*u*u
    ///
    ///     where u has zeros in the first k-1 positions. the form of
    ///     this transformation and the method of pivoting first
    ///     appeared in the corresponding linpack subroutine.
    ///
    ///     the subroutine statement is
    ///
    ///	subroutine qrfac(m,n,a,lda,pivot,ipvt,lipvt,rdiag,acnorm,wa)
    ///
    ///     where
    ///
    ///	m is a positive integer input variable set to the number
    ///	  of rows of a.
    ///
    ///	n is a positive integer input variable set to the number
    ///	  of columns of a.
    ///
    ///	a is an m by n array. on input a contains the matrix for
    ///	  which the qr factorization is to be computed. on output
    ///	  the strict upper trapezoidal part of a contains the strict
    ///	  upper trapezoidal part of r, and the lower trapezoidal
    ///	  part of a contains a factored form of q (the non-trivial
    ///	  elements of the u vectors described above).
    ///
    ///	lda is a positive integer input variable not less than m
    ///	  which specifies the leading dimension of the array a.
    ///
    ///	pivot is a logical input variable. if pivot is set true,
    ///	  then column pivoting is enforced. if pivot is set false,
    ///	  then no column pivoting is done.
    ///
    ///	ipvt is an integer output array of length lipvt. ipvt
    ///	  defines the permutation matrix p such that a*p = q*r.
    ///	  column j of p is column ipvt(j) of the identity matrix.
    ///	  if pivot is false, ipvt is not referenced.
    ///
    ///	lipvt is a positive integer input variable. if pivot is false,
    ///	  then lipvt may be as small as 1. if pivot is true, then
    ///	  lipvt must be at least n.
    ///
    ///	rdiag is an output array of length n which contains the
    ///	  diagonal elements of r.
    ///
    ///	acnorm is an output array of length n which contains the
    ///	  norms of the corresponding columns of the input matrix a.
    ///	  if this information is not needed, then acnorm can coincide
    ///	  with rdiag.
    ///
    ///	wa is a work array of length n. if pivot is false, then wa
    ///	  can coincide with rdiag.
    ///
    ///     subprograms called
    ///
    ///	minpack-supplied ... dpmpar,enorm
    ///
    ///	fortran-supplied ... dmax1,dsqrt,min0
    ///
    ///     argonne national laboratory. minpack project. march 1980.
    ///     burton s. garbow, kenneth e. hillstrom, jorge j. more
    fn qrfac(&mut self) {
        // compute the initial column norms and initialize several arrays.
        let mut ij = 0;
        for j in 0..self.nfree {
            self.wa2[j] = self.fjack[ij..ij + self.m].enorm();
            self.wa1[j] = self.wa2[j];
            self.wa3[j] = self.wa1[j];
            self.ipvt[j] = j;
            ij += self.m;
        }
        // reduce a to r with householder transformations.
        for j in 0..self.m.min(self.nfree) {
            // bring the column of largest norm into the pivot position.
            let mut kmax = j;
            for k in j..self.nfree {
                if self.wa1[k] > self.wa1[kmax] {
                    kmax = k;
                }
            }
            if kmax != j {
                let mut ij = self.m * j;
                let mut jj = self.m * kmax;
                for _ in 0..self.m {
                    self.fjack.swap(jj, ij);
                    ij += 1;
                    jj += 1;
                }
                self.wa1[kmax] = self.wa1[j];
                self.wa3[kmax] = self.wa3[j];
                self.ipvt.swap(j, kmax);
            }
            let jj = j + self.m * j;
            let mut ajnorm = self.fjack[jj..self.m - j + jj].enorm();
            if ajnorm == 0. {
                self.wa1[j] = -ajnorm;
                continue;
            }
            if self.fjack[jj] < 0. {
                ajnorm = -ajnorm;
            }
            ij = jj;
            for _ in j..self.m {
                self.fjack[ij] /= ajnorm;
                ij += 1;
            }
            self.fjack[jj] += 1.;
            // apply the transformation to the remaining columns
            // and update the norms.
            let jp1 = j + 1;
            if jp1 < self.nfree {
                for k in jp1..self.nfree {
                    let mut sum = 0.;
                    ij = j + self.m * k;
                    let mut jj = j + self.m * j;
                    for _ in j..self.m {
                        sum += self.fjack[jj] * self.fjack[ij];
                        ij += 1;
                        jj += 1;
                    }
                    let temp = sum / self.fjack[j + self.m * j];
                    ij = j + self.m * k;
                    jj = j + self.m * j;
                    for _ in j..self.m {
                        self.fjack[ij] -= temp * self.fjack[jj];
                        ij += 1;
                        jj += 1;
                    }
                    if self.wa1[k] != 0. {
                        let temp = self.fjack[j + self.m * k] / self.wa1[k];
                        let temp = (1. - temp.powi(2)).max(0.);
                        self.wa1[k] *= temp.sqrt();
                        let temp = self.wa1[k] / self.wa3[k];
                        if 0.05 * temp * temp < f64::EPSILON {
                            let start = jp1 + self.m * k;
                            self.wa1[k] = self.fjack[start..start + self.m - j - 1].enorm();
                            self.wa3[k] = self.wa1[k];
                        }
                    }
                }
            }
            self.wa1[j] = -ajnorm;
        }
    }
}

pub fn mpfit<T: MPFitter>(
    f: T,
    xall: &mut [f64],
    params: Option<&[MPPar]>,
    config: &MPConfig,
) -> MPResult {
    let mut fit = match MPFit::new(f.number_of_points(), xall, &f) {
        None => return MPResult::Error(MPError::Empty),
        Some(v) => v,
    };
    match &params {
        None => {
            fit.nfree = fit.npar;
            fit.ifree = (0..fit.npar).collect();
        }
        Some(pars) => {
            if pars.len() == 0 {
                return MPResult::Error(MPError::Empty);
            }
            for (i, p) in pars.iter().enumerate() {
                if p.fixed {
                    fit.qllim.push(p.limited_low);
                    fit.qulim.push(p.limited_up);
                    fit.llim.push(p.limit_low);
                    fit.ulim.push(p.limit_up);
                    if p.limited_low || p.limited_up {
                        fit.qanylim = true;
                    }
                } else {
                    fit.nfree += 1;
                    fit.ifree.push(i);
                }
                fit.side.push(p.side);
                fit.step.push(p.step);
                fit.dstep.push(p.rel_step);
            }
            if fit.nfree == 0 {
                return MPResult::Error(MPError::NoFree);
            }
        }
    };
    if fit.m < fit.nfree {
        return MPResult::Error(MPError::DoF);
    }
    f.eval(fit.xall, &mut fit.fvec);
    fit.fnorm = fit.fvec.enorm();
    let orig_norm = fit.fnorm * fit.fnorm;
    fit.xnew.copy_from_slice(fit.xall);
    fit.x = Vec::with_capacity(fit.nfree);
    for i in 0..fit.nfree {
        fit.x.push(fit.xall[fit.ifree[i]]);
    }
    // Initialize Levenberg-Marquardt parameter and iteration counter
    let par = 0.0;
    let mut iter = 1;
    fit.qtf = vec![0.; fit.nfree];
    fit.fjack = vec![0.; fit.m * fit.nfree];
    loop {
        for i in 0..fit.nfree {
            fit.xnew[fit.ifree[i]] = fit.x[i];
        }
        // Calculate the Jacobian matrix
        fit.fdjack2(&config);
        if fit.qanylim {
            for j in 0..fit.nfree {
                let lpegged = j < fit.qllim.len() && fit.x[j] == fit.llim[j];
                let upegged = j < fit.qulim.len() && fit.x[j] == fit.ulim[j];
                let mut sum = 0.;
                // If the parameter is pegged at a limit, compute the gradient direction
                let ij = j * fit.m;
                if lpegged || upegged {
                    for i in 0..fit.m {
                        sum += fit.fvec[i] * fit.fjack[ij + i];
                    }
                }
                // If pegged at lower limit and gradient is toward negative then
                // reset gradient to zero
                if lpegged && sum > 0. {
                    for i in 0..fit.m {
                        fit.fjack[ij + i] = 0.;
                    }
                }
                // If pegged at upper limit and gradient is toward positive then
                // reset gradient to zero
                if upegged && sum < 0. {
                    for i in 0..fit.m {
                        fit.fjack[ij + i] = 0.;
                    }
                }
            }
        }
        // Compute the QR factorization of the jacobian
        fit.qrfac();
        /*
         *	 on the first iteration and if mode is 1, scale according
         *	 to the norms of the columns of the initial jacobian.
         */
        if iter == 1 {
            if !config.do_user_scale {
                for j in 0..fit.nfree {
                    fit.diag[fit.ifree[j]] = if fit.wa2[j] == 0. { 1. } else { fit.wa2[j] };
                }
            }
            /*
             *	 on the first iteration, calculate the norm of the scaled x
             *	 and initialize the step bound delta.
             */
            for j in 0..fit.nfree {
                fit.wa3[j] = fit.diag[fit.ifree[j]] * fit.x[j];
            }
            fit.xnorm = fit.wa3.enorm();
            fit.delta = config.step_factor * fit.xnorm;
            if fit.delta == 0. {
                fit.delta = config.step_factor;
            }
        }
        /*
         *	 form (q transpose)*fvec and store the first n components in
         *	 qtf.
         */
        fit.wa4.copy_from_slice(&fit.fvec);
        let mut jj = 0;
        for j in 0..fit.nfree {
            let temp = fit.fjack[jj];
            if temp != 0. {
                let mut sum = 0.0;
                let mut ij = jj;
                for i in j..fit.m {
                    sum += fit.fjack[ij] * fit.wa4[i];
                    ij += 1;
                }
                let temp = -sum / temp;
                ij = jj;
                for i in j..fit.m {
                    fit.wa4[i] += fit.fjack[ij] * temp;
                    ij += 1;
                }
            }
            fit.fjack[jj] = fit.wa1[j];
            jj += fit.m + 1;
            fit.qtf[j] = fit.wa4[j];
        }
        /* ( From this point on, only the square matrix, consisting of the
        triangle of R, is needed.) */
        if config.no_finite_check {
            /* Check for overflow.  This should be a cheap test here since FJAC
            has been reduced to a (small) square matrix, and the test is
            O(N^2). */
            for val in &fit.fjack {
                if !val.is_finite() {
                    return MPResult::Error(MPError::Nan);
                }
            }
        }
        /*
         *	 compute the norm of the scaled gradient.
         */
        let mut gnorm: f64 = 0.;
        if fit.fnorm != 0. {
            let mut jj = 0;
            for j in 0..fit.nfree {
                let l = fit.ipvt[j];
                if fit.wa2[l] != 0. {
                    let mut sum = 0.;
                    let mut ij = jj;
                    for i in 0..=j {
                        sum += fit.fjack[ij] * (fit.qtf[i] / fit.fnorm);
                        ij += 1;
                    }
                    gnorm = gnorm.max((sum / fit.wa2[l]).abs());
                }
                jj += fit.m;
            }
        }
        if gnorm <= config.gtol {
            fit.info = MPSuccess::Dir;
        }
        break;
    }
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
            fn eval(&self, params: &[f64], deviates: &mut [f64]) {
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

            fn number_of_points(&self) -> usize {
                self.x.len()
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
