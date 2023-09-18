# 29/8/23

posterior using closed form solution for 5 points and then for 4 points after dleeting 1 point and they should match

bayesian linear regression again

use **solve** (bishop's book code must be present)

Martin craser blr (for code)


https://nbviewer.org/github/krasserm/bayesian-machine-learning/blob/dev/bayesian-linear-regression/bayesian_linear_regression.ipynb

if ax=b given then instead of x = A-1b use x = solve(A,b)

gregory gunderson - read blog (solve & inverse)

https://gregorygundersen.com/blog/2020/12/09/matrix-inversion/

laplace torch library -> simplest nn is logistic regression and then use laplace approx to find posterior

type 2 likelihood

unnormalized posterior approx using laplace approx instead of closed form solution

precision matrix, cholesky decomposition -> use instead of inverse of covariance matrix (better conditioned)

avoid computing inverses

wolfram alpha -> use, or use monte carlo sampling

<hr>

# 12/9/23

## Report 
- show prior after adding and removing 1-1 points
- equations basics
- diagram from christopher bishop book

## Notes
- probit approximation and logistic regression
- logistic regression true poaterior distribution using grid search
- piyush rai logistic rgression : bayesian averaging, iitk
- do shape manipulation earlier before passing
- calculating teh p(Ddel) -> use monte carlo estimate
- laplace torch library for nn
- linear and logistic regression using laplace torch
- torch and tree utils (present in jax)
- hamil torch search look into it -> for weights across nn -> flatten and unflatten utils
- outlier detection -> 
- outliers using moon datatset-> point not in training set but once u account for it you get a worsse fir -> something like this
- liver vala do 2-d classification
- backdoor defenses - look at
- look at log pdf of each point-> remove that point -> see how it changes -> heuristic to identify outliers (for now) we are choosing to unlearn


