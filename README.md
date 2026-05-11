# rmpfit

Very simple pure Rust implementation of the
[CMPFIT](https://pages.physics.wisc.edu/~craigm/idl/cmpfit.html) library:
the Levenberg-Marquardt technique to solve the least-squares problem.

The code is mainly copied directly from CMPFIT almost without changing.
The original CMPFIT tests (Linear (free parameters), Quad (free and fixed parameters),
and Gaussian (free and fixed parameters) function) are reproduced and passed.

Just a few obvious Rust-specific optimizations are done:

* Removing ```goto``` (fuf).
* Using `MPResult` for error handling.
* A few loops are zipped to help the compiler optimize the code (no performance tests are done anyway).
* Using trait ```MPFitter``` to call the user code.

## Advantages

* Pure Rust.
* No external dependencies.
* Internal Jacobian calculations.
* Sided, analytical or user provided derivatives are also implemented.
* Derivative debug mode (comparing analytical vs numerical) prints to stderr (as in cmpfit).

## Usage Example

A user should implement trait ```MPFitter``` for its struct:

```rust
use assert_approx_eq::assert_approx_eq;
use rmpfit::{MPFitter, MPResult};

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

fn main() {
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
    // initializing input parameters
    let mut init = [1., 1.];
    let res = l.mpfit(&mut init).unwrap();
    assert_close(init[0], 3.20996572); // actual 3.2
    assert_close(res.xerror[0], 0.02221018);
    assert_close(init[1], 1.77095420); // actual 1.78
    assert_close(res.xerror[1], 0.01893756);
}

fn assert_close(left: f64, right: f64) {
    let abs_tol: f64 = 1e-6;
    let rel_tol: f64 = 1e-9;
    let diff = (left - right).abs();
    let scale = left.abs().max(right.abs()).max(1.0);
    assert!(diff <= abs_tol.max(rel_tol * scale));
}
```

then ```init``` will contain the refined parameters of the fitting function.
If user function fails to calculate residuals, it should return ```MPError::Eval```.

License: MIT
