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
- probit approximation in logistic regression
- logistic regression true posterior distribution using grid search
- piyush rai logistic rgression : bayesian averaging, iitk
- do shape manipulation earlier before passing
- calculating the p(Ddel) -> use monte carlo estimate
- laplace torch library for nn
- linear and logistic regression using laplace torch
- torch and tree utils (present in jax)
- hamil torch search look into it -> for weights across nn -> flatten and unflatten utils
- outlier detection -> 
- outliers using moon datatset-> point not in training set but once u account for it you get a worse fit -> something like this
- liver vala do 2-d classification
- backdoor defenses - look at
- look at log pdf of each point-> remove that point -> see how it changes -> heuristic to identify outliers (for now) we are choosing to unlearn


# 19/9/23

## Notes
- Go back to og paper and what is actually that we need and can we use anything other than laplace approx, maybe rejection sampling
- We need original posterior -> we cannot comput in closed form in nn -> we need to approximate it -> laplace approx is one way to do it or we can get sample through mcmc
- get posterior using laplace torch -> then we get ~p, 
- go through sirs notebook and replicate it for classifciation and regression and then do it for unlearning task.
- reduce nn to 2 layers and tehn visualize it using contour plots.
- nb for important sampling of sir -> different data that u could have deleted find out what p(ddel|theta) -> calculate
curve: x->p(x), y->p(del|theta)
- visualization of teh formula -> unlearned posterior updated
- unlearning for linear regression solidify

# 26/9/23

## Notes

- flatten the parameters from the dictionary -> use hamiltorch (use flatten and unflatten like in assign 3) or use jtu (jax tree utils)
- jax tree utils -> stores the exact structure of dict ->> use similar on torch -> tree flatten and unflatten in hamiltorch
- deepminds tree library may work -> checkout
- make inder code better
- linear regression thing post on github issues -> see if possibe afte rmaking more complex nn

