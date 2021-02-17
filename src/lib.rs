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
        }
    }
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
            max_iter: 200,
            max_fev: 0,
            n_print: true,
            do_user_scale: false,
            no_finite_check: false,
        }
    }
}

/// MP Fit errors
pub enum MPError {
    NoError,
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
#[derive(PartialEq)]
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
    xall: &'a mut [f64],
    qtf: Vec<f64>,
    fjack: Vec<f64>,
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
    orig_norm: f64,
    par: f64,
    iter: usize,
    cfg: &'a MPConfig,
}

impl<'a, F: MPFitter> MPFit<'a, F> {
    fn new(m: usize, xall: &'a mut [f64], f: &'a F, cfg: &'a MPConfig) -> Option<MPFit<'a, F>> {
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
                xall,
                qtf: vec![],
                fjack: vec![],
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
                orig_norm: 0.0,
                par: 0.0,
                iter: 1,
                cfg,
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
        // Calculate the Jacobian matrix
        let eps = config.epsfcn.max(f64::EPSILON).sqrt();
        // TODO: probably sides and analytical derivatives should be implemented at some point
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
        // Compute the QR factorization of the jacobian
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

    fn parse_params(&mut self, params: Option<&[MPPar]>) -> MPError {
        match &params {
            None => {
                self.nfree = self.npar;
                self.ifree = (0..self.npar).collect();
            }
            Some(pars) => {
                if pars.len() == 0 {
                    return MPError::Empty;
                }
                for (i, p) in pars.iter().enumerate() {
                    if p.fixed {
                        self.qllim.push(p.limited_low);
                        self.qulim.push(p.limited_up);
                        self.llim.push(p.limit_low);
                        self.ulim.push(p.limit_up);
                        if p.limited_low || p.limited_up {
                            self.qanylim = true;
                        }
                    } else {
                        self.nfree += 1;
                        self.ifree.push(i);
                    }
                    self.step.push(p.step);
                    self.dstep.push(p.rel_step);
                }
                if self.nfree == 0 {
                    return MPError::NoFree;
                }
            }
        };
        if self.m < self.nfree {
            return MPError::DoF;
        }
        MPError::NoError
    }

    // Initialize Levenberg-Marquardt parameter and iteration counter
    fn init_lm(&mut self) {
        self.f.eval(self.xall, &mut self.fvec);
        self.fnorm = self.fvec.enorm();
        self.orig_norm = self.fnorm * self.fnorm;
        self.xnew.copy_from_slice(self.xall);
        self.x = Vec::with_capacity(self.nfree);
        for i in 0..self.nfree {
            self.x.push(self.xall[self.ifree[i]]);
        }
        self.qtf = vec![0.; self.nfree];
        self.fjack = vec![0.; self.m * self.nfree];
    }

    fn check_limits(&mut self) {
        if !self.qanylim {
            return;
        }
        for j in 0..self.nfree {
            let lpegged = j < self.qllim.len() && self.x[j] == self.llim[j];
            let upegged = j < self.qulim.len() && self.x[j] == self.ulim[j];
            let mut sum = 0.;
            // If the parameter is pegged at a limit, compute the gradient direction
            let ij = j * self.m;
            if lpegged || upegged {
                for i in 0..self.m {
                    sum += self.fvec[i] * self.fjack[ij + i];
                }
            }
            // If pegged at lower limit and gradient is toward negative then
            // reset gradient to zero
            if lpegged && sum > 0. {
                for i in 0..self.m {
                    self.fjack[ij + i] = 0.;
                }
            }
            // If pegged at upper limit and gradient is toward positive then
            // reset gradient to zero
            if upegged && sum < 0. {
                for i in 0..self.m {
                    self.fjack[ij + i] = 0.;
                }
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
            let temp = self.fjack[jj];
            if temp != 0. {
                let mut sum = 0.0;
                let mut ij = jj;
                for i in j..self.m {
                    sum += self.fjack[ij] * self.wa4[i];
                    ij += 1;
                }
                let temp = -sum / temp;
                ij = jj;
                for i in j..self.m {
                    self.wa4[i] += self.fjack[ij] * temp;
                    ij += 1;
                }
            }
            self.fjack[jj] = self.wa1[j];
            jj += self.m + 1;
            self.qtf[j] = self.wa4[j];
        }
    }

    /// Check for overflow.  This should be a cheap test here since FJAC
    /// has been reduced to a (small) square matrix, and the test is O(N^2).
    fn check_is_finite(&self) -> bool {
        if !self.cfg.no_finite_check {
            for val in &self.fjack {
                if !val.is_finite() {
                    return false;
                }
            }
        }
        true
    }

    ///	 compute the norm of the scaled gradient.
    fn gnorm(&self) -> f64 {
        let mut gnorm: f64 = 0.;
        if self.fnorm != 0. {
            let mut jj = 0;
            for j in 0..self.nfree {
                let l = self.ipvt[j];
                if self.wa2[l] != 0. {
                    let mut sum = 0.;
                    let mut ij = jj;
                    for i in 0..=j {
                        sum += self.fjack[ij] * (self.qtf[i] / self.fnorm);
                        ij += 1;
                    }
                    gnorm = gnorm.max((sum / self.wa2[l]).abs());
                }
                jj += self.m;
            }
        }
        gnorm
    }

    fn terminate(&self) -> MPResult {
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

    fn rescale(&mut self) {
        if self.cfg.do_user_scale {
            return;
        }
        for j in 0..self.nfree {
            let i = self.ifree[j];
            self.diag[i] = self.diag[i].max(self.wa2[j]);
        }
    }

    /// subroutine lmpar
    ///
    /// given an m by nfree matrix a, an nfree by nfree nonsingular diagonal
    /// matrix d, an m-vector b, and a positive number delta,
    /// the problem is to determine a value for the parameter
    /// par such that if wa1 solves the system
    ///
    ///	    a*wa1 = b ,	  sqrt(par)*d*wa1 = 0 ,
    ///
    /// in the least squares sense, and dxnorm is the euclidean
    /// norm of d*wa1, then either par is zero and
    ///
    ///	    (dxnorm-delta) .le. 0.1*delta ,
    ///
    /// or par is positive and
    ///
    ///	    abs(dxnorm-delta) .le. 0.1*delta .
    ///
    /// this subroutine completes the solution of the problem
    /// if it is provided with the necessary information from the
    /// qr factorization, with column pivoting, of a. that is, if
    /// a*p = q*fjack, where p is a permutation matrix, q has orthogonal
    /// columns, and fjack is an upper triangular matrix with diagonal
    /// elements of nonincreasing magnitude, then lmpar expects
    /// the full upper triangle of fjack, the permutation matrix p,
    /// and the first nfree components of (q transpose)*b. on output
    /// lmpar also provides an upper triangular matrix s such that
    ///
    ///	     t	 t		     t
    ///	    p *(a *a + par*d*d)*p = s *s .
    ///
    /// s is employed within lmpar and may be of separate interest.
    ///
    /// only a few iterations are generally needed for convergence
    /// of the algorithm. if, however, the limit of 10 iterations
    /// is reached, then the output par will contain the best
    /// value obtained so far.
    ///
    /// the subroutine statement is
    ///
    ///	subroutine lmpar(nfree,fjack,m,ipvt,diag,qtf,delta,par,wa1,wa2,
    ///			 wa3,wa4)
    ///
    /// where
    ///
    ///	nfree is a positive integer input variable set to the order of fjack.
    ///
    ///	fjack is an nfree by nfree array. on input the full upper triangle
    ///	  must contain the full upper triangle of the matrix fjack.
    ///	  on output the full upper triangle is unaltered, and the
    ///	  strict lower triangle contains the strict upper triangle
    ///	  (transposed) of the upper triangular matrix s.
    ///
    ///	m is a positive integer input variable not less than nfree
    ///	  which specifies the leading dimension of the array fjack.
    ///
    ///	ipvt is an integer input array of length nfree which defines the
    ///	  permutation matrix p such that a*p = q*fjack. column j of p
    ///	  is column ipvt(j) of the identity matrix.
    ///
    ///	diag is an input array of length nfree which must contain the
    ///	  diagonal elements of the matrix d.
    ///
    ///	qtf is an input array of length nfree which must contain the first
    ///	  nfree elements of the vector (q transpose)*b.
    ///
    ///	delta is a positive input variable which specifies an upper
    ///	  bound on the euclidean norm of d*wa1.
    ///
    ///	par is a nonnegative variable. on input par contains an
    ///	  initial estimate of the levenberg-marquardt parameter.
    ///	  on output par contains the final estimate.
    ///
    ///	wa1 is an output array of length nfree which contains the least
    ///	  squares solution of the system a*wa1 = b, sqrt(par)*d*wa1 = 0,
    ///	  for the output par.
    ///
    ///	wa2 is an output array of length nfree which contains the
    ///	  diagonal elements of the upper triangular matrix s.
    ///
    ///	wa3 and wa4 are work arrays of length nfree.
    ///
    /// subprograms called
    ///
    ///	minpack-supplied ... dpmpar,mp_enorm,qrsolv
    ///
    ///	fortran-supplied ... dabs,mp_dmax1,dmin1,dsqrt
    ///
    /// argonne national laboratory. minpack project. march 1980.
    /// burton s. garbow, kenneth e. hillstrom, jorge j. more
    fn lmpar(&mut self) {
        /*
         *     compute and store in wa1 the gauss-newton direction. if the
         *     jacobian is rank-deficient, obtain a least squares solution.
         */
        let mut nsing = self.nfree;
        let mut jj = 0;
        for j in 0..self.nfree {
            self.wa3[j] = self.qtf[j];
            if self.fjack[jj] == 0. && nsing == self.nfree {
                nsing = j;
            }
            if nsing < self.nfree {
                self.wa3[j] = 0.;
            }
            jj += self.m + 1;
        }
        if nsing >= 1 {
            for k in 0..nsing {
                let j = nsing - k - 1;
                let mut ij = self.m * j;
                self.wa3[j] /= self.fjack[j + ij];
                let temp = self.wa3[j];
                if j > 0 {
                    for i in 0..j {
                        self.wa3[i] -= self.fjack[ij] * temp;
                        ij += 1;
                    }
                }
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
        let dxnorm = self.wa4[0..self.nfree].enorm();
        let fp = dxnorm - self.delta;
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
            for j in 0..self.nfree {
                let l = self.ipvt[j];
                self.wa3[j] = self.diag[self.ifree[l]] * (self.wa4[l] / dxnorm);
            }
            let mut jj = 0;
            for j in 0..self.nfree {
                let mut sum = 0.;
                if j > 0 {
                    let mut ij = jj;
                    for i in 0..j {
                        sum += self.fjack[ij] * self.wa3[i];
                        ij += 1;
                    }
                }
                self.wa3[j] = (self.wa3[j] - sum) / self.fjack[j + self.m * j];
                jj += self.m;
            }
            let temp = self.wa3[0..self.nfree].enorm();
            parl = ((fp / self.delta) / temp) / temp;
        }
        /*
         *     calculate an upper bound, paru, for the zero of the function.
         */
        let mut jj = 0;
        for j in 0..self.nfree {
            let mut sum = 0.;
            let mut ij = jj;
            for i in 0..=j {
                sum += self.fjack[ij] * self.qtf[i];
                ij += 1;
            }
            let l = self.ipvt[j];
            self.wa3[j] = sum / self.diag[self.ifree[l]];
            jj += self.m;
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
        }
    }

    /// subroutine qrsolv
    ///
    /// given an m by n matrix a, an n by n diagonal matrix d,
    /// and an m-vector b, the problem is to determine an x which
    /// solves the system
    ///
    ///	a*x = b ,	  d*x = 0 ,
    ///
    /// in the least squares sense.
    ///
    /// this subroutine completes the solution of the problem
    /// if it is provided with the necessary information from the
    /// qr factorization, with column pivoting, of a. that is, if
    /// a*p = q*r, where p is a permutation matrix, q has orthogonal
    /// columns, and r is an upper triangular matrix with diagonal
    /// elements of nonincreasing magnitude, then qrsolv expects
    /// the full upper triangle of r, the permutation matrix p,
    /// and the first n components of (q transpose)*b. the system
    /// a*x = b, d*x = 0, is then equivalent to
    ///
    ///		   t	   t
    ///	r*z = q *b ,  p *d*p*z = 0 ,
    ///
    /// where x = p*z. if this system does not have full rank,
    /// then a least squares solution is obtained. on output qrsolv
    /// also provides an upper triangular matrix s such that
    ///
    ///	 t	 t		 t
    ///	p *(a *a + d*d)*p = s *s .
    ///
    /// s is computed within qrsolv and may be of separate interest.
    ///
    /// the subroutine statement is
    ///
    ///	subroutine qrsolv(n,r,ldr,ipvt,diag,qtb,x,sdiag,wa)
    ///
    /// where
    ///
    ///	n is a positive integer input variable set to the order of r.
    ///
    ///	r is an n by n array. on input the full upper triangle
    ///	  must contain the full upper triangle of the matrix r.
    ///	  on output the full upper triangle is unaltered, and the
    ///	  strict lower triangle contains the strict upper triangle
    ///	  (transposed) of the upper triangular matrix s.
    ///
    ///	ldr is a positive integer input variable not less than n
    ///	  which specifies the leading dimension of the array r.
    ///
    ///	ipvt is an integer input array of length n which defines the
    ///	  permutation matrix p such that a*p = q*r. column j of p
    ///	  is column ipvt(j) of the identity matrix.
    ///
    ///	diag is an input array of length n which must contain the
    ///	  diagonal elements of the matrix d.
    ///
    ///	qtb is an input array of length n which must contain the first
    ///	  n elements of the vector (q transpose)*b.
    ///
    ///	x is an output array of length n which contains the least
    ///	  squares solution of the system a*x = b, d*x = 0.
    ///
    ///	sdiag is an output array of length n which contains the
    ///	  diagonal elements of the upper triangular matrix s.
    ///
    ///	wa is a work array of length n.
    ///
    /// subprograms called
    ///
    ///	fortran-supplied ... dabs,dsqrt
    ///
    /// argonne national laboratory. minpack project. march 1980.
    /// burton s. garbow, kenneth e. hillstrom, jorge j. more
    fn qrsolv(&mut self) {
        /*
         *     copy r and (q transpose)*b to preserve input and initialize s.
         *     in particular, save the diagonal elements of r in x.
         */
        let mut kk = 0;
        for j in 0..self.nfree {
            let mut ij = kk;
            let mut ik = kk;
            for _ in j..self.nfree {
                self.fjack[ij] = self.fjack[ik];
                ij += 1;
                ik += self.m;
            }
            self.wa1[j] = self.fjack[kk];
            self.wa4[j] = self.qtf[j];
            kk += self.m + 1;
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
                for k in j..self.nfree {
                    self.wa2[k] = 0.;
                }
                self.wa2[j] = self.wa3[l];
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
                    let kk = k + self.m * k;
                    let (sinx, cosx) = if self.fjack[kk].abs() < self.wa2[k].abs() {
                        let cotan = self.fjack[kk] / self.wa2[k];
                        let sinx = 0.5 / (0.25 + 0.25 * cotan * cotan).sqrt();
                        let cosx = sinx * cotan;
                        (sinx, cosx)
                    } else {
                        let tanx = self.wa2[k] / self.fjack[kk];
                        let cosx = 0.5 / (0.25 + 0.25 * tanx * tanx).sqrt();
                        let sinx = cosx * tanx;
                        (sinx, cosx)
                    };
                    /*
                     *	    compute the modified diagonal element of r and
                     *	    the modified element of ((q transpose)*b,0).
                     */
                    self.fjack[kk] = cosx * self.fjack[kk] + sinx * self.wa2[k];
                    let temp = cosx * self.wa4[k] + sinx * qtbpj;
                    qtbpj = -sinx * self.wa4[k] + cosx * qtbpj;
                    self.wa4[k] = temp;
                    /*
                     *	    accumulate the tranformation in the row of s.
                     */
                    let kp1 = k + 1;
                    if self.nfree > kp1 {
                        let mut ik = kk + 1;
                        for i in kp1..self.nfree {
                            let temp = cosx * self.fjack[ik] + sinx * self.wa2[i];
                            self.wa2[i] = -sinx * self.fjack[ik] + cosx * self.wa2[i];
                            self.fjack[ik] = temp;
                            ik += 1;
                        }
                    }
                }
            }
            /*
             *	 store the diagonal element of s and restore
             *	 the corresponding diagonal element of r.
             */
            let kk = j + self.m * j;
            self.wa2[j] = self.fjack[kk];
            self.fjack[kk] = self.wa1[j];
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
        if nsing > 0 {
            for k in 0..nsing {
                let j = nsing - k - 1;
                let mut sum = 0.;
                let jp1 = j + 1;
                if nsing > jp1 {
                    let mut ij = jp1 + self.m * j;
                    for i in jp1..nsing {
                        sum += self.fjack[ij] * self.wa4[i];
                        ij += 1;
                    }
                }
                self.wa4[j] = (self.wa4[j] - sum) / self.wa2[j];
            }
        }
        /*
         *     permute the components of z back to components of x.
         */
        for j in 0..self.nfree {
            self.wa1[self.ipvt[j]] = self.wa4[j];
        }
    }
}

pub fn mpfit<T: MPFitter>(
    f: T,
    xall: &mut [f64],
    params: Option<&[MPPar]>,
    config: &MPConfig,
) -> MPResult {
    let mut fit = match MPFit::new(f.number_of_points(), xall, &f, config) {
        None => return MPResult::Error(MPError::Empty),
        Some(v) => v,
    };
    let params_error = fit.parse_params(params);
    match &params_error {
        MPError::NoError => (),
        _ => return MPResult::Error(params_error),
    }
    fit.init_lm();
    loop {
        fit.fill_xnew();
        fit.fdjack2(&config);
        fit.check_limits();
        fit.qrfac();
        fit.scale();
        fit.transpose();
        if !fit.check_is_finite() {
            return MPResult::Error(MPError::Nan);
        }
        let gnorm = fit.gnorm();
        if gnorm <= config.gtol {
            fit.info = MPSuccess::Dir;
        }
        if fit.info != MPSuccess::Error {
            return fit.terminate();
        }
        if config.max_iter == 0 {
            fit.info = MPSuccess::MaxIter;
            return fit.terminate();
        }
        fit.rescale();
        loop {
            fit.lmpar();
            return fit.terminate();
        }
    }
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
