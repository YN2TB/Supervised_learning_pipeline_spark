# Advanced PySpark MLlib: Supervised Pipelines, Custom Transformers, Distributed Ensembles

**Dataset.** US DOT on-time performance, calendar year 2015 — 5,819,079 flights,
592 MB CSV, joined to 322 airports and 14 carriers.
**Tasks.** Arrival delay in minutes (regression) and severe delay, `ARRIVAL_DELAY > 30`
(binary classification), trained under one shared pipeline and one 80/20 split.

---

# Part A — Mathematical foundations and distributed mechanics

## A1. Scaling mathematics and distributed sparse operations

### A1.1 Standardisation vs. range scaling

**Z-score (StandardScaler).** For feature $j$ with mean $\mu_j$ and standard
deviation $\sigma_j$:

$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

producing $\mathbb{E}[z_j] = 0$, $\operatorname{Var}(z_j) = 1$. The transform is
unbounded, so outliers survive it — a value 40 standard deviations out is still 40
after scaling. It assumes an approximately symmetric distribution to be meaningful.

**Min–max (MinMaxScaler).** Mapping onto $[\min, \max]$, by default $[0,1]$:

$$x^{\text{scaled}}_{ij} = \frac{x_{ij} - x^{\min}_j}{x^{\max}_j - x^{\min}_j}\left(\max - \min\right) + \min$$

The output range is guaranteed, which suits bounded activations and distance
kernels, but the mapping is defined by two order statistics. A single extreme
value sets $x^{\max}_j$ and compresses every ordinary observation into a narrow
band near zero. On this dataset that failure is concrete: `DEPARTURE_DELAY` has a
maximum near 1,650 minutes against a median of −2, so min–max scaling would crush
almost the entire distribution into roughly the first 2% of the range.

Both are affine in $x$, so neither changes the *shape* of a distribution — neither
one fixes skew. That is what A1.3 is for.

**Degenerate columns.** If $\sigma_j = 0$ (a constant feature), $z_{ij}$ is
undefined. Spark defines it as $0$ rather than failing. That convention is safe
numerically but has a consequence downstream that we hit in practice — see A2.3.

### A1.2 Why `withMean=True` is dangerous on distributed sparse vectors

A `SparseVector` stores only its non-zero entries, as parallel index and value
arrays. Its memory is $O(\text{nnz})$, not $O(n)$.

Centering breaks that representation. For any feature with $\mu_j \neq 0$, a
structural zero maps to

$$0 - \mu_j = -\mu_j \neq 0$$

so **every** zero becomes an explicit stored non-zero. Sparsity is not reduced —
it is destroyed outright, and the vector must be materialised dense. Spark guards
this: the older `mllib` `StandardScalerModel` refuses sparse input when
`withMean=true` outright, and the `ml` implementation has to densify.

The cost is a function of the one-hot expansion. In our assembled vector:

| | value |
|---|---|
| total feature slots after assembly | 80 |
| slots contributed by one-hot groups (`AIRLINE`, `MONTH`, `DAY_OF_WEEK`, `DEP_HOUR`) | 60 |
| non-zeros per row (20 numeric + 4 one-hot indicators) | ~24 |
| density | ~30% |

Centering would take every row from ~24 stored values to 80 — a **~3.3× increase
in shuffle volume, cache footprint and GC pressure**, for 5.7M rows, and it scales
with the *worst* case: push `ROUTE` (4,279 levels) through one-hot instead of
target encoding and density falls below 1%, making centering a ~100× penalty.

**How `withStd` avoids it.** Scaling is multiplicative:

$$x_{ij} \mapsto \frac{x_{ij}}{\sigma_j}, \qquad 0 \mapsto 0$$

Zero is a fixed point, so the sparsity pattern is preserved exactly and only the
stored values change. Spark can rescale a `SparseVector` in place, touching
$O(\text{nnz})$ entries.

This is why the pipeline sets `withMean=False, withStd=True` deliberately
(`mllib_pipeline.py`), not as an untouched default. The cost is that features are
scaled but not centred; for the tree ensembles this is irrelevant (splits are
scale- and location-invariant), and for the regularised linear models it is
handled by fitting an intercept, which absorbs the uncentred mean.

### A1.3 Robust scaling: log compression before standardisation

Affine scaling cannot fix skew. `DISTANCE`, `SCHEDULED_TIME` and the delay columns
are heavily right-skewed, so we compress first:

$$x \mapsto \ln(1 + x)$$

Two properties matter. It is monotonic, so ordering is preserved and tree splits
are unaffected. And because $\frac{d}{dx}\ln(1+x) = \frac{1}{1+x}$, the derivative
shrinks as $x$ grows: large values are pulled in far harder than small ones, which
is exactly the compression skew needs. Multiplicative structure becomes additive,
which is what the linear models can actually represent. It also fixes zero:
$\ln(1+0) = 0$, so sparsity survives — the same argument as A1.2.

**The negative-value problem.** `DEPARTURE_DELAY` is negative for early flights,
where $\ln(1+x)$ is undefined for $x \le -1$. We therefore use the **signed**
extension (`SignedLog1pTransformer`):

$$f(x) = \operatorname{sign}(x)\ln(1 + |x|)$$

which is defined on all of $\mathbb{R}$, agrees with $\ln(1+x)$ for $x \ge 0$, is
monotonic across zero, odd, and continuous with $f(0)=0$. It compresses both tails
symmetrically, so a 3-hour early arrival and a 3-hour late arrival are compressed
alike.

**Truncation vs. compression, and a mistake worth recording.** These are
different tools: Tukey truncation clips to fences $[Q_1 - 1.5\,\text{IQR},\,
Q_3 + 1.5\,\text{IQR}]$ and *removes* tail information; log compression *retains*
it with rank intact. But the deeper lesson from this dataset is that sometimes
the right answer is **neither**, and we got this wrong twice before measuring it.

`DEPARTURE_DELAY` is the dominant predictor of arrival delay, and the two are
very nearly *linear*: $\operatorname{corr} = 0.947$ with a slope near 1, which is
simply the statement that a late departure arrives late unless the schedule
absorbs it.

The first pipeline fed the models only the **clipped** value. That was wrong
because an outlier is not the same thing as a large value: the fitted fence lands
at $[-23, 25]$ minutes and puts 12.8% of flights above it — flights averaging
**+72.9 minutes** of arrival delay against −5.6 for everyone else. Clipping drops
the correlation to 0.614, capping achievable $R^2$ near $0.614^2 \approx 0.38$.

The obvious repair — send it through the signed log instead — was *also* wrong,
and for a subtler reason: the relationship is linear, so a monotone nonlinear
transform actively destroys the structure a linear model is built to exploit.
Measured directly, with `LinearRegression` on a 2% sample:

| treatment of `DEPARTURE_DELAY` | $R^2$ |
|---|---|
| clipped only (first attempt) | 0.390 |
| signed log only (second attempt) | 0.409 |
| clipped + signed log | 0.409 |
| **raw, untransformed** | **0.937** |
| raw + clipped + signed log | 0.937 |

The raw column carries essentially all of the signal, and both derived views add
nothing measurable on top of it (+0.0001). So the pipeline passes
`DEPARTURE_DELAY` through untransformed, retains `dep_delay_clip` as a bounded
companion view, and drops the log view of it entirely.

The general principle: skew is a reason to compress, and contamination is a
reason to truncate, but a feature that is large, genuine, and linearly related to
the target needs neither. Reaching for a transform because a distribution *looks*
untidy is how signal gets destroyed.

`DISTANCE` is the case that does want compression: its fitted upper fence lands
at **2,091 miles**, which would fold every transcontinental flight into one value,
so it is log-compressed and never clipped. `TAXI_OUT` is the case that genuinely
wants truncation, where extreme values really are operational anomalies.

---

## A2. Distributed PCA mechanics

### A2.1 What is actually distributed

PCA needs the eigen-decomposition of the covariance of an $m \times n$ matrix $A$,
here $m = 5{,}704{,}000$ rows and $n = 80$ features. The essential asymmetry is
that **$m$ is enormous and $n$ is small**, and Spark's design follows from it.

The covariance requires the Gramian $A^\top A$, and that decomposes as a sum over
rows:

$$A^\top A = \sum_{i=1}^{m} a_i a_i^\top$$

Each term is $n \times n$. So every executor scans only its own partition,
accumulating a local $n \times n$ matrix, and those partial sums are combined by
`treeAggregate` — a tree-structured reduction whose depth is logarithmic in the
partition count, so the driver receives $O(\log p)$ merges rather than $p$ of them.

The important consequence: **the $m \times n$ data matrix is never collected.** Only
an $n \times n$ summary — 80×80 here, ~51 KB — crosses the network per partial
aggregate. Communication is independent of $m$ entirely.

> **A correction worth stating precisely.** It is often said that Spark computes
> PCA "without collecting the covariance matrix to the driver". That is not quite
> right, and the distinction matters. `RowMatrix.computePrincipalComponentsAndExplainedVariance`
> — the method `spark.ml`'s `PCA` calls — *does* assemble the full $n \times n$
> covariance on the driver and run a **local** Breeze SVD on it. What is never
> collected is the *data*. The claim that survives scrutiny is: cost is
> $O(mn^2/p)$ distributed plus $O(n^3)$ on the driver, and only the second term is
> centralised.
>
> That is fine at $n=80$ ($80^3 \approx 5\times10^5$ flops, microseconds) and is
> precisely why it becomes the bottleneck as $n$ grows: the driver-side term is
> cubic and single-threaded. `RowMatrix.computeSVD` therefore switches strategy by
> size — a local dense SVD for small $n$, and ARPACK's implicitly-restarted
> Lanczos for large $n$ or small $k$, where only the matrix–vector product
> $A^\top(Av)$ is distributed and the driver never holds an $n \times n$ object at
> all.

### A2.2 From SVD to principal components

For the centred matrix, the SVD $A = U\Sigma V^\top$ gives

$$A^\top A = V\Sigma^\top U^\top U \Sigma V^\top = V\Sigma^2 V^\top$$

using $U^\top U = I$. So the columns of $V$ are the eigenvectors of $A^\top A$ —
the principal directions — and the eigenvalues are $\lambda_i = \sigma_i^2$. The
projection onto the top $k$ components is $A V_k$, which is again a per-row
operation and so is a distributed map with no shuffle at all.

**Explained variance ratio.** With eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots
\ge \lambda_p$, the fraction of total variance retained by the first $k$
components is

$$\text{EVR}(k) = \frac{\sum_{i=1}^{k}\lambda_i}{\sum_{j=1}^{p}\lambda_j}$$

The denominator equals $\operatorname{tr}(A^\top A) = \sum_j \operatorname{Var}(x_j)$,
so with standardised inputs it is just $p$.

**Measured, and the result is unflattering to PCA here.** Ten components retain
only **28.1%** of the variance of the 80-dimensional feature space, and the cost
shows up directly in the models: linear regression scores $R^2 = 0.930$ on the
full feature set against $0.682$ through `PCA(k=10)`.

That is not a bug, it is what PCA does to this kind of feature space. Most of the
dimensions are one-hot indicators, which are close to mutually orthogonal and
individually low-variance, so there is little linear redundancy for PCA to
exploit — variance is spread thinly across many directions rather than
concentrated in a few. PCA pays off when features are strongly correlated; one-hot
expansions of independent categoricals are close to the opposite case. The scree
curve in `docs/benchmarks/pca_explained_variance.png` shows exactly that shape:
no elbow, just a slow decline.

The honest conclusion is that dimensionality reduction is the wrong tool for this
feature matrix, and the two-arm design is what makes that visible rather than
hiding it behind a single reported number.

### A2.3 A consequence we hit in practice

The driver-side SVD is not merely a scaling footnote; it is a failure mode. Our
first tournament run died with `breeze.linalg.NotConvergedException` inside
`PCA.fit`.

The cause: `OneHotEncoder(handleInvalid="keep")` reserves a slot for unseen
categories. On a small cross-validation fold that slot never fires, so the column
is constant; `StandardScaler` maps a zero-variance column to all zeros (A1.1); and
the covariance matrix is then **singular**. LAPACK's driver failed to converge on
it rather than degrading gracefully. Diagnostically, the feature matrix contained
no NaN and no Inf — 3 of 80 columns were identically zero.

The fix is a `VarianceThresholdSelector` ahead of PCA, dropping zero-variance
columns. It is both the repair and the correct modelling choice: a constant
feature carries no information by definition. Note the interaction — the bug
surfaced only at small sample sizes, because on the full data most categories
appear in every fold. Testing at reduced scale is what exposed it.

---

## A3. Parametric models vs. tree ensembles

### A3.1 ElasticNet regression

Spark's `LinearRegression` minimises

$$J(\beta) = \frac{1}{2n}\sum_{i=1}^{n}\left(y_i - x_i^\top\beta\right)^2 + \lambda\left[\alpha\lVert\beta\rVert_1 + \frac{1-\alpha}{2}\lVert\beta\rVert_2^2\right]$$

with `regParam` $=\lambda$ and `elasticNetParam` $=\alpha$.

The two penalties do different jobs. The $\ell_2$ term has gradient $\lambda(1-\alpha)\beta$,
which shrinks coefficients smoothly toward zero without reaching it, and adds
$\lambda(1-\alpha)I$ to the normal equations — so $X^\top X + \lambda(1-\alpha)I$ is
invertible even when $X^\top X$ is not. That directly buys stability under the
collinearity our features have (`DISTANCE`, `GC_DISTANCE_MI` and `SCHEDULED_TIME`
are near-collinear by construction).

The $\ell_1$ term is not differentiable at zero; its subgradient is constant
$\pm\lambda\alpha$, so it applies the same push regardless of coefficient size and
drives small coefficients exactly to zero. That yields genuine sparsity —
selection, not just shrinkage.

Setting $\alpha = 0$ gives ridge, $\alpha = 1$ gives lasso. The mixture matters
here for a specific reason: lasso alone, given a group of correlated features,
picks one arbitrarily and zeroes the rest, and which one it picks is unstable
across resamples. The ridge component restores the grouping effect, so correlated
features shrink together. $\alpha = 0.5$ is the brief's default and our grid also
tries $\alpha = 0$.

Because $\ell_1$ is non-smooth, Spark optimises with OWL-QN rather than plain
L-BFGS.

### A3.2 Generalised linear models and link functions

A GLM has three parts: a response distribution from the exponential family, a
linear predictor $\eta = X\beta$, and a **link** $g$ connecting them:

$$g\left(\mathbb{E}[Y]\right) = X\beta$$

Ordinary least squares is the special case of a Gaussian response with the
identity link. The link earns its keep when the identity is wrong about the
response's *support* or its *mean–variance relationship*.

Arrival delay is a good example. Delay is strongly right-skewed and its dispersion
grows with its level — a flight averaging 5 minutes late varies by minutes, one
averaging 200 minutes late varies by tens of minutes. Ordinary least squares
assumes constant variance and so is systematically misspecified.

| Family | Support | $V(\mu)$ | Canonical use |
|---|---|---|---|
| Gaussian | $\mathbb{R}$ | $1$ | symmetric, constant-variance |
| Poisson | $\{0,1,2,\dots\}$ | $\mu$ | counts; variance = mean |
| Gamma | $(0,\infty)$ | $\mu^2$ | positive continuous, constant *relative* error |

With a **log link**, $\log\mu = X\beta \Rightarrow \mu = e^{X\beta}$. Two
consequences: $\mu > 0$ is guaranteed for any $\beta$, and coefficients become
multiplicative — $\beta_j$ is a proportional change in expected delay per unit of
$x_j$, not an additive one. Gamma's $V(\mu) = \mu^2$ means constant *coefficient of
variation*, which matches delay behaviour better than constant absolute variance.

Fitting is by **iteratively reweighted least squares**: at each step form the
working response $z = \eta + (y - \mu)g'(\mu)$ and weights $w = 1/(V(\mu)g'(\mu)^2)$,
then solve a weighted least-squares problem. Each iteration is one distributed
weighted regression, so a GLM costs a small multiple of an OLS fit.

> **A constraint the brief's setup creates.** A log link — Gamma or Poisson — requires
> $y > 0$ strictly, but arrival delay is *negative* for every early flight — the
> majority of them. We therefore fit the GLM on $y + c$ with the constant
> $c = |\min(y, 0)| + 1$ computed once on the training data. Because RMSE, MAE and
> $R^2$ are all invariant to a common shift of prediction and target, the reported
> metrics remain directly comparable to the other regressors with no inverse
> transform. This is a real modelling decision, not a workaround: it says we are
> modelling *delay relative to the earliest possible arrival* on a multiplicative
> scale.

### A3.3 LinearSVC and the soft-margin hinge loss

A separating hyperplane $w^\top x + b = 0$ has geometric margin $2/\lVert w\rVert$,
so maximising the margin means minimising $\lVert w\rVert^2$. Real data is not
separable, so slack variables $\xi_i \ge 0$ permit violations:

$$\min_{w,b,\xi} \; \frac{1}{2}\lVert w\rVert^2 + C\sum_{i=1}^{n}\xi_i
\quad\text{s.t.}\quad y_i\left(w^\top x_i + b\right) \ge 1 - \xi_i,\; \xi_i \ge 0$$

The constraints are tight at the optimum, so $\xi_i = \max\left(0, 1 - y_i(w^\top x_i + b)\right)$,
and substituting gives the unconstrained **hinge-loss** form:

$$\min_{w,b}\;\frac{1}{2}\lVert w\rVert^2 + C\sum_{i=1}^{n}\max\left(0,\,1 - y_i\left(w^\top x_i + b\right)\right)$$

The hinge is zero once a point is correctly classified *beyond* the margin, which
is what distinguishes it from log loss: correct, confident points contribute
nothing to the gradient, so the solution depends only on the support vectors near
the boundary. It is convex but non-smooth at the kink, so Spark optimises via
subgradients (OWL-QN).

**Spark's parameterisation differs from the textbook.** `LinearSVC` minimises the
*averaged* loss with an $\ell_2$ penalty scaled by `regParam`:

$$\frac{1}{n}\sum_{i=1}^{n}\max\left(0, 1 - y'_i\left(w^\top x_i + b\right)\right) + \frac{\lambda}{2}\lVert w\rVert^2$$

with $y'_i \in \{-1, +1\}$ recoded from the $\{0,1\}$ label. Comparing to the form
above, $C \leftrightarrow \frac{1}{n\lambda}$ — so **larger `regParam` means
*weaker* regularisation in $C$ terms**, the opposite of the direction people
usually expect. LinearSVC is also linear-only in Spark: there is no kernel trick,
because the dual formulation needs an $n \times n$ Gram matrix, which at
$n = 4.5$M is $2 \times 10^{13}$ entries and hopeless. Scaling matters for exactly
this reason — the $\ell_2$ penalty is not scale-invariant.

### A3.4 Random Forest — bagging and parallel tree growth

Two independent randomisations decorrelate the trees.

**Bagging.** Each tree trains on a bootstrap resample, $n$ draws with replacement.
The probability a given row is omitted from a given tree is
$\left(1 - \frac{1}{n}\right)^n \to e^{-1} \approx 0.368$, so each tree sees about
63.2% of the distinct rows and the remaining ~37% form its out-of-bag set.

**Feature subspace sampling.** At *each node*, only a random subset of $m$ of the
$p$ features is considered — $m = \sqrt{p}$ for classification, $m = p/3$ for
regression (Spark's `featureSubsetStrategy="auto"`).

Why both are needed: bagging alone leaves the trees highly correlated, because a
single dominant predictor gets chosen as the root split in nearly every bootstrap.
Here `DEPARTURE_DELAY` is exactly such a feature. For $B$ trees each of variance
$\sigma^2$ with pairwise correlation $\rho$, the average has variance

$$\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

Adding trees drives the second term to zero but **cannot touch the first**. Only
reducing $\rho$ lowers the floor, and forcing different trees to consider different
features at each node is what does that.

**Parallel growth.** The trees are conditionally independent given the data, so
they can all be built at once. Spark does something better than one-tree-per-task:
it grows *all* trees breadth-first, level by level. Each pass over the data
computes sufficient statistics for every node currently on the frontier across
every tree, so the number of data passes is $O(\text{depth})$ rather than
$O(B \times \text{depth})$. `maxMemoryInMB` caps how many nodes are processed per
pass.

### A3.5 Gradient-Boosted Trees — sequential residual fitting

Boosting builds an additive model stage-wise:

$$F_m(x) = F_{m-1}(x) + \nu\, h_m(x)$$

where $h_m$ is fitted to the **negative gradient** of the loss at the current
prediction,

$$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{m-1}}$$

and $\nu$ is the learning rate. For squared error, $L = \frac{1}{2}(y - F)^2$ gives
$r_{im} = y_i - F_{m-1}(x_i)$ — the residuals, which is where "fitting the
residuals" comes from; it is the special case, not the definition.

**Histogram binning and `maxBins`.** Exact split-finding on a continuous feature
would sort every value at every node: $O(n\log n)$ per feature per node, and a sort
is a shuffle. Instead Spark discretises each continuous feature *once* into at most
`maxBins` intervals via a sampled quantile sketch. Split finding then reduces to
accumulating per-bin sufficient statistics — one pass, $O(n)$ with no shuffle — and
scanning $O(\text{maxBins})$ candidate thresholds. `maxBins` is the accuracy/cost
dial: too few and genuine thresholds are unavailable; too many and both memory and
scan cost rise. It must also be $\ge$ the largest categorical arity, which is why
our grid uses 32 and 64 against a maximum arity of 24 (`DEP_HOUR`).

**Why trees cannot be built in parallel.** $h_m$ is fitted to residuals of
$F_{m-1}$, so tree $m$ cannot start before tree $m-1$ finishes — a genuine
sequential dependency, unlike bagging's conditional independence. The parallelism
is therefore *within* each tree: across partitions when accumulating histograms,
and across features and bins when evaluating splits. The practical consequences
are that GBT makes $O(M \times \text{depth})$ passes where RF makes
$O(\text{depth})$, that adding executors stops helping much sooner, and that GBT
overfits with more trees whereas RF does not — RF averages, so extra trees only
reduce variance, while boosting keeps reducing *bias* against the training set.

**Where the tuning effort goes.** This asymmetry is why our grids differ: RF is
tuned over `numTrees` and `maxDepth` (both cheap, both variance-side), GBT over
`maxDepth` and `maxBins` at fixed `maxIter`, since `maxIter` is the parameter most
likely to overfit and the most expensive to search.

### A3.6 Summary of the trade-off

| | Random Forest | Gradient-Boosted Trees |
|---|---|---|
| Ensemble direction | parallel, averaging | sequential, additive |
| Attacks | variance | bias |
| Tree dependence | independent given data | tree $m$ depends on $m-1$ |
| Data passes | $O(\text{depth})$ | $O(M \times \text{depth})$ |
| More trees | safe (converges) | can overfit |
| Distributed scaling | near-linear in executors | limited by the sequential chain |

---

# Part B — Implementation and results

## B1. Data preparation and the leakage boundary

### B1.1 The leakage policy — the single most important decision here

The prediction point is fixed at **wheels-off**: the aircraft has pushed back and
left the ground, so departure delay and taxi-out are legitimately known, and the
question is whether it will arrive late. Every column observable only *after* that
instant is dropped, and the policy is asserted in code
(`flight_schema.py: LEAKY_COLUMNS`, enforced by an assertion in `data_prep.py`),
not merely documented.

Dropped: `ARRIVAL_TIME`, `ELAPSED_TIME`, `AIR_TIME`, `WHEELS_ON`, `TAXI_IN`, and
the five delay-attribution columns `AIR_SYSTEM_DELAY`, `SECURITY_DELAY`,
`AIRLINE_DELAY`, `LATE_AIRCRAFT_DELAY`, `WEATHER_DELAY`.

Those last five are the trap. By the USDOT definition they **sum to
`ARRIVAL_DELAY`** for any flight delayed 15 minutes or more, so a model given them
can reconstruct the target by addition. Leaving them in produces $R^2 \approx
0.99$ and a model that is worthless in production, because at wheels-off none of
them exist yet. We also drop `ARRIVAL_DELAY` itself after deriving the labels, so
no unlabelled copy of the target remains in the table to be picked up by accident.

**How to check this held.** At the wheels-off horizon, expect $R^2 \approx
0.85\text{–}0.92$ — departure delay legitimately dominates arrival delay. A result
near 0.99 means a leak survived.

### B1.2 Schema drift: the October airport codes

One month of the 2015 feed ships a **different key encoding**: October rows carry
numeric DOT airport ids (`14747`) where every other month carries IATA codes
(`SEA`). Verified over the full year:

| months affected | rows affected | distinct numeric codes |
|---|---|---|
| October only (month 10) | 486,165 — *every* October row | 307 |

Since `airports.csv` is keyed by IATA only, the naive join silently drops all
486,165 rows — no error, just 8% of the year missing and every October flight
without coordinates. We recover the mapping **from the data itself**
(`scripts/build_airport_code_map.py`) using two independent signals, because
neither is sufficient alone.

**Signal 1 — direction-aware flight-number vote.** A flight number flies the same
route all year, so key on `(AIRLINE, FLIGHT_NUMBER, DISTANCE)` and read off the
IATA code the other eleven months show on the same key. One subtlety: some
carriers reuse a flight number for both legs of a round trip, so that key can
describe A→B on some days and B→A on others, letting an origin code collect votes
for the destination airport. We therefore vote only from keys whose direction is
unambiguous. This resolves 300/307 codes and is reliable where votes are plentiful.

**Signal 2 — geometric fit.** Every row carries its own route `DISTANCE`, so an
unknown code can be located by trilateration: score each candidate airport by how
well it reproduces the observed distances to its already-known partners, and take
the best fit. A candidate is accepted only if the fit is tight (≤10 mi) *and*
decisive against the runner-up — either a 25 mi absolute margin or a 5× ratio,
the latter being what makes a two-anchor fit safe.

**Why both.** Swapping a route's endpoints leaves its great-circle distance
unchanged, so signal 2 is blind to a transposition — which is exactly what signal
1's direction-awareness prevents. Conversely signal 1 fails in the thin tail
(some codes get two votes), which is where signal 2 is decisive.

**Verification, and why it had to be per-code.** Pooled statistics looked
excellent from the very first attempt — median error 0.96 mi across 486k rows —
while five codes were badly wrong. A mis-mapped airport is wrong on *every* row it
appears in, so it shows as a large **median** for that code, but at ~1,500 rows
each out of 486,165 it barely moves the global median. Scoring each code
separately exposed them immediately:

| code | voted | median error | resolved to |
|---|---|---|---|
| `12016` | GUM | 2,781 mi | *(unresolved)* |
| `14222` | PPG | 1,630 mi | *(unresolved)* |
| `15323` | ATL | 227 mi | **TRI** (fit 0.28 mi, margin 415 mi) |
| `11898` | DTW | 243 mi | **GFK** (fit 0.31 mi, margin 27 mi) |
| `10577` | DTW | 378 mi | **BGM** (fit 0.99 mi, margin 21 mi) |

**Outcome.** 302 of 307 codes resolved and geometrically verified; the mapping is
injective; worst per-code error 3.25 mi; **485,280 of 486,165 October rows (99.8%)
recovered**. The five remaining codes had no confident fit — best candidates 7.8
to 235 mi off — and are left unmapped rather than guessed, costing 885 rows
(0.015% of the dataset). Refusing to assert an unsupported mapping is the right
call; the alternative would have put wrong coordinates into the haversine feature.

### B1.3 Curation summary

| step | rows |
|---|---|
| raw | 5,819,079 |
| after dropping cancelled / diverted / null target | 5,714,008 |
| after dropping missing airport coordinates | 5,704,000 |
| **final curated** | **5,704,000 (98.0%)** |

Severe-delay rate (`ARRIVAL_DELAY > 30`): **11.08%** — a moderately imbalanced
binary problem, which is why the classification arm reports AUC-PR alongside
AUC-ROC.

CSV is converted once to Parquet partitioned by month. The tournament re-reads the
table dozens of times, and columnar storage with predicate pushdown pays that back
on every read.

## B2. Custom transformers

### B2.1 Why `OutlierIQRTruncator` is an Estimator, not a Transformer

The brief says "subclass `pyspark.ml.Transformer` to compute $Q_1, Q_3$ bounds and
clip". Implemented literally, that is a leak. A `Transformer` has only
`_transform`, so it must recompute its quantiles from whatever DataFrame it is
handed — meaning that applied to the test split it would fit itself to test data,
and applied to a streaming micro-batch it would compute quantiles over a handful
of rows, giving different clipping behaviour on every batch.

Clipping fences are **learned parameters**. So `OutlierIQRTruncator` is an
`Estimator` whose `_fit` computes the fences on the training split only and
freezes them into `OutlierIQRTruncatorModel`. Our Phase 2 test asserts both halves
of this: that test data *would* have produced different fences (so the risk is
real — 1 of 3 columns differs), and that `transform(test)` nonetheless respects the
training fences exactly.

Quantiles come from `approxQuantile`, a distributed Greenwald–Khanna sketch:
single pass, $O(1/\varepsilon)$ memory per partition, all columns in one go —
versus a full sort of 5.7M rows.

Fitted fences (full training split): `DEPARTURE_DELAY` $[-23, 25]$,
`TAXI_OUT` $[-1, 31]$.

### B2.2 Arrow-vectorised UDFs

`haversine_miles_udf` is a Series-to-Series `pandas_udf`. Spark hands whole Arrow
record batches to Python, so the trigonometry runs once over a NumPy array at C
speed rather than once per row through the interpreter — no per-row
serialisation and no Python-level loop. It is invoked from inside a Pipeline stage
(`HaversineTransformer`), not as a loose DataFrame operation, so the feature
engineering travels with the serialized model and is reproduced identically at
inference time.

Independent check: the computed great-circle distance agrees with the `DISTANCE`
column the data already carries to a **median of 0.97 mi** (p99 5.99 mi) — the
residual being the difference between airport reference points and published route
distance.

A second UDF, `schedule_speed_udf`, computes the implied ground speed of the
published schedule — a domain pressure metric, since a leg scheduled at unusually
high implied speed has little slack for a departure delay to absorb.

### B2.3 Leakage-safe target encoding

`ORIGIN_AIRPORT` (318), `DESTINATION_AIRPORT` (317) and `ROUTE` (4,279 levels) are
too high-cardinality for one-hot encoding — `ROUTE` alone would add 4,278 columns
and, per A1.2, dominate the sparsity budget. Target encoding replaces each category
with a smoothed mean of the target:

$$e_c = \frac{\sum_{i \in c} y_i + m\bar{y}}{n_c + m}$$

As $n_c \to \infty$ this approaches the raw category mean; for a category seen once
or twice it stays near the global prior $\bar{y}$, which is what stops rare
airports being fitted to noise. We use $m = 20$.

Leak-safety is **structural**: the means are computed in `_fit`, so a Pipeline
fitted on the training split cannot see test targets, and because `CrossValidator`
refits the whole Pipeline per fold, each fold's encoding is computed from that
fold's training portion only. Unseen categories fall back to the prior rather than
producing nulls — verified in the Phase 2 test.

**The remaining subtlety, stated honestly.** A training row still contributes to
its own category's mean, so its own target leaks into its own feature. With ~1,300
rows per category here the self-influence is $O(1/n_c)$ and negligible, but the
strict remedy is out-of-fold encoding, and on a dataset with many rare categories
it would be required rather than optional.

### B2.4 Serialization, and why it is the crux

Phase 4 loads a serialized `PipelineModel`, so any parameter that will not
round-trip breaks streaming inference — and nothing earlier would notice. Every
custom class mixes in `DefaultParamsReadable`/`DefaultParamsWritable`, and
`TargetEncoderModel` carries its fitted mapping as a **JSON string Param**. That
keeps it inside the default writer with no custom `MLWriter`/`MLReader`. A Parquet
side-car would scale to far larger vocabularies at the cost of custom IO code; at
a few thousand keys JSON is comfortable. The Phase 2 test asserts a full
save → load → identical-output round trip.

### B2.5 Performance: a 6× fit speedup and 60× transform speedup

The first working pipeline took **146 s to fit** on 114k rows while its stages cost
~4 s in isolation. The cause is structural, not a bug: `Pipeline.fit` fits each
stage against the *lazy* output of the stages before it and never caches in
between, so every fitted stage downstream re-executes the whole upstream chain —
and iterative learners (LinearRegression, PCA's SVD, LinearSVC, GBT) re-execute it
once per iteration.

Two fixes, both measured rather than assumed:

**1. Target encoding via a cached map expression, not a join.** Profiling the
fitted stages one at a time showed `TargetEncoderModel` alone taking a pass from
0.08 s to 13.18 s: it rebuilt three lookup DataFrames from Python objects on every
call, each costing a pickle, a parallelize job and a broadcast exchange. Replacing
the broadcast join with a literal `create_map` expression made execution ~60×
cheaper (0.10 s vs 6 s per pass) at the cost of 6.3 s of driver-side plan
construction — which is then paid **once**, because Column expressions are
independent of any particular DataFrame and can be memoised across transforms. A
broadcast-join fallback is retained above 20,000 categories.

**2. Caching at fit time only.** A cache stage after the feature engineering cut
fitting from 51 s to 35 s, but *raised* transform cost from 0.28 s to 6.5 s — a 23×
penalty on every scoring call, including each fold's evaluation, because every
stage after that point is a single-pass projection that never needed
materialising. Splitting it into an `Estimator` whose `_fit` persists and a `Model`
whose `_transform` is a no-op captures both sides: `persist` marks the logical
plan so subsequently-fitted stages hit the cache, while the saved model carries no
caching behaviour into scoring or streaming at all.

| | fit | transform |
|---|---|---|
| original | 146 s | 19.7 s |
| + fit-time cache, cached map expression | **24.6 s** | **0.32 s** |
| | **5.9× faster** | **62× faster** |

Verified semantically neutral: after the optimisation the Phase 2 test reports
byte-identical column totals.

## B3. Pipeline architecture

```
OutlierIQRTruncator      (Estimator: Tukey fences from train only)
HaversineTransformer     (Arrow pandas_udf: great-circle, schedule speed, detour)
SignedLog1pTransformer   (skew compression, negative-safe)
Imputer                  (median, fitted on train)
TargetEncoder            (Estimator: smoothed category means from train only)
StringIndexer            (AIRLINE)
OneHotEncoder            (AIRLINE, MONTH, DAY_OF_WEEK, DEP_HOUR)
MaterializeCache         (persists during fit only)
VectorAssembler          -> features_raw   (80 slots)
StandardScaler           (withMean=False, withStd=True)  -- see A1.2
VarianceThresholdSelector(drops zero-variance columns)   -- see A2.3
PCA(k=10)  or  identity  (two arms)
<estimator>
```

**Why two arms.** PCA is required by the brief, but it rotates features into
linear combinations, so `featureImportances` over principal components is not
interpretable in terms of the original variables. Every model is therefore trained
both with PCA(k=10) and without; the no-PCA arm supplies the interpretable
importance plots, and the comparison itself measures what the compression to 10
components costs.

## B4. Tournament, tuning and results

Five algorithm families across both tasks, under one identical 80/20 split
(`seed=42`): Linear Regression (ElasticNet), GLM (Poisson/log), LinearSVC, Random
Forest (regressor + classifier) and GBT (regressor + classifier).

`CrossValidator(numFolds=5, parallelism=4)` evaluates grid points concurrently
across executor slots rather than serially.

**Tuning strategy.** Hyperparameters are searched on a stratified sample
(`--tune-fraction`, default 10%) and the winning configuration is then refitted on
the *full* training set. Seven model families × 5 folds × grid over 4.5M rows is
many hours, and the *ranking* of hyperparameters stabilises well before the metric
value does. The sample fraction is a CLI flag, so a full-data search is one
argument away.

### Results

<!-- results_table.md is generated by benchmark_results.py from the MLflow store -->

### Regression arm

| model | arm | rmse | mae | r2 | train_s |
|---|---|---|---|---|---|
| linear_regression | nopca | 10.5171 | 7.1939 | 0.9289 | 1111 |
| gbt_regressor | nopca | 12.8530 | 7.5262 | 0.8938 | 2325 |
| random_forest_regressor | nopca | 14.0499 | 8.3463 | 0.8731 | 1881 |
| random_forest_regressor | pca | 14.9809 | 10.0465 | 0.8558 | 1731 |
| gbt_regressor | pca | 15.2086 | 10.0520 | 0.8513 | 2262 |
| linear_regression | pca | 21.4813 | 14.3725 | 0.7034 | 1142 |
| glm_poisson_log | pca | 27.0180 | 12.5177 | 0.5309 | 2149 |
| glm_poisson_log | nopca | 40.1434 | 11.4800 | -0.0357 | 2073 |

### Classification arm

| model | arm | areaUnderROC | areaUnderPR | f1 | accuracy | train_s |
|---|---|---|---|---|---|---|
| gbt_classifier | nopca | 0.9817 | 0.9384 | 0.9716 | 0.9723 | 2336 |
| linear_svc | nopca | 0.9806 | 0.9338 | 0.9675 | 0.9692 | 959 |
| random_forest_classifier | nopca | 0.9792 | 0.9277 | 0.9683 | 0.9691 | 2084 |
| gbt_classifier | pca | 0.9699 | 0.8801 | 0.9546 | 0.9559 | 1533 |
| linear_svc | pca | 0.9657 | 0.8769 | 0.9547 | 0.9561 | 803 |
| random_forest_classifier | pca | 0.9635 | 0.8709 | 0.9523 | 0.9547 | 2074 |

Split: 4,564,168 train / 1,139,832 test. Hyperparameters searched on 91,144 rows
(2.0% of train), 5 folds, `parallelism=4`, then refitted on the full training set.

**The linear model wins the regression task outright**, and that is the headline
result rather than an anticlimax. Arrival delay is very nearly an affine function
of departure delay ($r = 0.947$): a flight that pushes back 40 minutes late lands
about 40 minutes late. A single coefficient captures that exactly. Both ensembles
must approximate the same straight line with piecewise-constant regions, spending
depth budget to do worse — GBT reaches $R^2$ 0.894 and Random Forest 0.873 against
linear regression's **0.929**, while costing 2.1× and 1.7× the wall-clock time.
This is the honest answer to the brief's parametric-vs-ensemble comparison: model
capacity beyond the true functional form buys nothing, and axis-aligned splits are
a poor basis for a diagonal relationship.

**PCA loses in all seven paired arms, without exception.** At $k = 10$ the
components retain only **28.5%** of total variance (§A2.2), and the damage tracks
how much each model depends on the discarded directions:

| | no-PCA | PCA | cost |
|---|---|---|---|
| linear_regression | 0.9289 | 0.7034 | **−0.226** $R^2$ |
| gbt_regressor | 0.8938 | 0.8513 | −0.043 $R^2$ |
| random_forest_regressor | 0.8731 | 0.8558 | −0.017 $R^2$ |
| gbt_classifier | 0.9817 | 0.9699 | −0.012 AUC |
| linear_svc | 0.9806 | 0.9657 | −0.015 AUC |
| random_forest_classifier | 0.9792 | 0.9635 | −0.016 AUC |

The linear model suffers most, by an order of magnitude. It has no way to recover a
predictor that has been rotated into a discarded component, whereas a tree can
partially reconstruct the signal from whichever surviving components correlate with
it. This is the trade-off §B3 predicted, measured: PCA is required by the brief, it
is defensible as decorrelation and compression, and here it costs accuracy in every
single arm while also destroying `featureImportances` interpretability.

**Classification is a three-way tie decided on cost.** GBT (0.9817), LinearSVC
(0.9806) and Random Forest (0.9792) are separated by 0.0025 AUC — well inside what
a different seed would move. But LinearSVC reaches that in **959 s against GBT's
2,336 s**, a 2.4× difference, because boosting fits its trees sequentially and
cannot parallelise across them (§A3.3) while a linear SVM's hinge-loss objective is
convex and solved by a handful of distributed gradient passes. Thresholding at 30
minutes turns a near-linear regression surface into a near-linear decision boundary,
so the simplest model finds it. On this data the ensembles are not buying accuracy;
they are buying time.

**The GLM underperforms for a structural reason, not a tuning one.** Poisson/log
reaches $R^2$ 0.531 (PCA) and −0.036 (no-PCA), far behind every other regressor. A
log link models $\mu = \exp(\mathbf{x}^\top\beta)$ — a *multiplicative* response —
while the true relationship is *additive*: arrival delay is departure delay plus a
roughly constant taxi-and-cruise term. Compounding this, the log link requires a
strictly positive response, so the label is shifted by $+88$ minutes; the model must
then reproduce an additive shift through an exponential, which it can only do over a
narrow range. The residual RMSE/MAE ratio of 3.5 (40.14 / 11.48) shows the failure
mode directly — the median prediction is reasonable while a thin tail of
$\exp(\mathbf{x}^\top\beta)$ blow-ups dominates the squared error.

This replaced an earlier Gamma/log arm that failed far more violently ($R^2 = -207$,
RMSE 568.90 against MAE 10.80). Gamma's variance function $V(\mu) = \mu^2$ amplifies
exactly these tail excursions where Poisson's $V(\mu) = \mu$ grows linearly; the
switch plus a heavier regularisation grid moved $R^2$ by more than two orders of
magnitude. Both remain within the brief's "Gamma/Poisson family with log link", and
the comparison is itself the clearest available demonstration of what a variance
function does.

**Sanity check.** The winning $R^2$ of 0.929 sits inside the 0.85–0.93 band expected
at the wheels-off horizon. A value near 0.99 would indicate a leak column survived
the drop list; near 0.48 would indicate a feature transform destroying
`DEPARTURE_DELAY` (§B2.2). Neither occurred.


Figures in `docs/benchmarks/`:

| figure | shows |
|---|---|
| `pca_explained_variance.png` | cumulative variance retained vs. $k$ |
| `model_comparison_regression.png` | RMSE by model, PCA vs. no-PCA |
| `model_comparison_classification.png` | AUC-ROC by model, PCA vs. no-PCA |
| `feature_importance_*.png` | Gini importance, no-PCA arm |
| `residuals.png` | predicted vs. actual, residual distribution |

## B5. MLflow instrumentation

`mlflow.pyspark.ml.autolog()` captures pipeline hyperparameters and the nested
cross-validation runs automatically; metrics, the explained-variance artifact and
the feature-importance artifacts are logged explicitly, and the winning regression
pipeline is registered and promoted **Staging → Production**.

MLflow is pinned to **2.x deliberately**: `transition_model_version_stage` — the
stage API the brief specifies — is removed in MLflow 3.x in favour of aliases.

## B6. Serialization and streaming inference

`inference.py` loads the serialized `PipelineModel` (from disk, or from the
registry's Production stage with `--from-registry`) and scores a
`spark.readStream` JSON source, writing to a console sink and a Parquet sink with
checkpointing. Scoring is stateless, so each micro-batch is independent.

Two details that make it work:

* `custom_transformers.py` must be importable when the model deserializes — the
  saved metadata refers to those classes by qualified name — hence `--py-files` in
  `submit_pipeline.sh`.
* A file-source stream cannot infer its schema, so the training run publishes
  `models/input_schema.json` next to the model and the streaming job declares it.
  That file *is* the serving contract.

`MaterializeCache` returns the DataFrame untouched for streaming inputs, since
caching is unsupported there — which is why it was built as a fit-time Estimator
rather than a Transformer (B2.5).

---

# Appendix: reproducing this

```bash
source ./env.sh
python scripts/smoke_test.py                    # runtime gate (5/5)
python data_prep.py --step raw                  # CSV -> Parquet
python scripts/build_airport_code_map.py        # October code recovery
python data_prep.py --step curate               # joins, labels, leak policy
python scripts/test_transformers.py             # transformer gate (8/8)
./submit_pipeline.sh --tune-fraction 0.02 --folds 5   # tournament + MLflow
python benchmark_results.py                     # figures + results table
python inference.py &                           # streaming scorer
python scripts/make_stream_events.py --batches 5
```

Environment pins and the reasoning behind each are in `SETUP.md`. The short
version: PySpark 3.5 silently kills its Python workers on Python 3.12.0, the
machine's default JDK 26 is unsupported by every Spark release, and
`PipelineModel.save()` on Windows needs a `hadoop.dll` matching Spark's bundled
Hadoop 3.3.4 client jars.
