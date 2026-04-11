"""
============================================================
  rmath Data Science Project: Student Performance Analysis
  Modules: constants, scalar, stats, Vector, vector, Array
============================================================
"""

import random, time
import rmath
from rmath import scalar, vector, stats, constants as C, Vector, Array

SEP = "=" * 60
def hdr(n, t): print(f"\n{SEP}\n[{n:02d}] {t}\n{SEP}")

# ── Synthetic data ────────────────────────────────────────
def make_data(n=400, seed=7):
    random.seed(seed)
    rows, y_r, y_c = [], [], []
    for _ in range(n):
        sh = random.uniform(1, 10)
        sl = random.uniform(4, 9)
        ps = random.uniform(40, 100)
        at = random.uniform(60, 100)
        score = 5*sh + 2*sl + 0.3*ps + 0.2*at + random.gauss(0, 4)
        score = max(0.0, min(100.0, score))
        rows.append([sh, sl, ps, at])
        y_r.append(score); y_c.append(1.0 if score>=60 else 0.0)
    return rows, y_r, y_c

# ════════════════════════════════════════════════════════
# 01  CONSTANTS
# ════════════════════════════════════════════════════════
hdr(1, "Mathematical Constants  (rmath.constants)")
print(f"  PI          = {C.PI:.15f}")
print(f"  TAU         = {C.TAU:.15f}")
print(f"  E           = {C.E:.15f}")
print(f"  PHI         = {C.PHI:.15f}  (golden ratio)")
print(f"  SQRT_2      = {C.SQRT_2:.15f}")
print(f"  LN_2        = {C.LN_2:.15f}")
print(f"  LOG2_E      = {C.LOG2_E:.15f}")
print(f"  EPSILON_F64 = {C.EPSILON_F64:.2e}")
print(f"  EPSILON_F32 = {C.EPSILON_F32:.2e}")
print(f"  MAX_F64     = {C.MAX_F64:.4e}")
print(f"  INF / NAN   = {C.INF}  /  {C.NAN}")

# ════════════════════════════════════════════════════════
# 02  SCALAR  (all groups)
# ════════════════════════════════════════════════════════
hdr(2, "Scalar Operations  (rmath.scalar)")

print("  -- Arithmetic --")
print(f"  add(7,3)={scalar.add(7,3)}  sub(10,4)={scalar.sub(10,4)}")
print(f"  mul(6,7)={scalar.mul(6,7)}  div(22,7)={scalar.div(22,7):.6f}")
print(f"  remainder(10,3)={scalar.remainder(10,3)}  fmod(-7,3)={scalar.fmod(-7,3)}")

print("  -- Rounding --")
print(f"  ceil(2.1)={scalar.ceil(2.1)}  floor(2.9)={scalar.floor(2.9)}")
print(f"  round(2.5)={scalar.round(2.5)}  round_half_even(2.5)={scalar.round_half_even(2.5)}")
print(f"  trunc(-3.9)={scalar.trunc(-3.9)}  sign(-5)={scalar.sign(-5)}")
print(f"  clamp(15,0,10)={scalar.clamp(15,0,10)}  lerp(0,100,0.4)={scalar.lerp(0,100,0.4)}")

print("  -- Roots/Powers --")
print(f"  sqrt(144)={scalar.sqrt(144)}  cbrt(-27)={scalar.cbrt(-27)}")
print(f"  root(16,4)={scalar.root(16,4)}  root(-125,3)={scalar.root(-125,3)}")
print(f"  pow(2,10)={scalar.pow(2,10)}  abs(-9)={scalar.abs(-9)}")
print(f"  inv_sqrt(4)={scalar.inv_sqrt(4)}")

print("  -- Exp/Log --")
print(f"  exp(1)={scalar.exp(1):.8f}  exp2(8)={scalar.exp2(8)}")
print(f"  expm1(1e-10)={scalar.expm1(1e-10):.4e}  log(E)={scalar.log(C.E):.1f}")
print(f"  log(1024,2)={scalar.log(1024,2):.1f}  log2(256)={scalar.log2(256):.1f}")
print(f"  log10(1000)={scalar.log10(1000):.1f}  log1p(1e-12)={scalar.log1p(1e-12):.4e}")
print(f"  logsumexp2(3,4)={scalar.logsumexp2(3,4):.6f}")

print("  -- Trigonometry --")
print(f"  sin(PI/2)={scalar.sin(C.PI/2):.1f}  cos(PI)={scalar.cos(C.PI):.1f}")
print(f"  tan(PI/4)={scalar.tan(C.PI/4):.10f}  atan2(1,1)={scalar.atan2(1,1):.8f}")
print(f"  asin(1)={scalar.asin(1):.8f}  acos(0)={scalar.acos(0):.8f}")
print(f"  degrees(PI)={scalar.degrees(C.PI):.1f}  radians(90)={scalar.radians(90):.8f}")

print("  -- Hyperbolic --")
print(f"  sinh(1)={scalar.sinh(1):.6f}  cosh(0)={scalar.cosh(0):.1f}")
print(f"  tanh(100)={scalar.tanh(100):.1f}  asinh(1)={scalar.asinh(1):.6f}")
print(f"  acosh(2)={scalar.acosh(2):.6f}  atanh(0.5)={scalar.atanh(0.5):.6f}")

print("  -- Geometry --")
print(f"  hypot(3,4)={scalar.hypot(3,4):.1f}  hypot_3d(2,3,6)={scalar.hypot_3d(2,3,6):.1f}")
print(f"  fma(3,4,5)={scalar.fma(3,4,5)}  copysign(5,-2)={scalar.copysign(5,-2)}")
print(f"  frexp(0.5)={scalar.frexp(0.5)}  ulp(1.0)={scalar.ulp(1.0):.2e}")
print(f"  nextafter(1.0,2.0)={scalar.nextafter(1.0,2.0):.20f}")

print("  -- Predicates --")
print(f"  isfinite(INF)={scalar.isfinite(C.INF)}  isinf(INF)={scalar.isinf(C.INF)}")
print(f"  isnan(NAN)={scalar.isnan(C.NAN)}  is_integer(3.0)={scalar.is_integer(3.0)}")
print(f"  isclose(0.1+0.2,0.3)={scalar.isclose(0.1+0.2, 0.3)}")

print("  -- Integer --")
print(f"  factorial(12)={scalar.factorial(12):,}")
print(f"  gcd(252,105)={scalar.gcd(252,105)}  lcm(12,15)={scalar.lcm(12,15)}")
print(f"  is_prime(97)={scalar.is_prime(97)}  is_power_of_two(1024)={scalar.is_power_of_two(1024)}")
print(f"  next_power_of_two(100)={scalar.next_power_of_two(100)}")

# ════════════════════════════════════════════════════════
# 03  STATS
# ════════════════════════════════════════════════════════
hdr(3, "Statistics  (rmath.stats)")
d = [23.5,18.0,31.2,27.8,19.5,33.1,25.0,28.4,21.7,30.0,
     26.5,22.3,29.8,24.1,20.9,32.0,17.5,28.0,25.5,31.8]
d2= [22.1,17.5,30.8,26.5,19.0,32.5,24.2,27.9,21.0,29.5,
     25.8,21.9,29.2,23.8,20.5,31.5,17.0,27.5,25.0,31.2]
print(f"  n={len(d)}  data={d[:4]}…")
print(f"  mean={stats.mean(d):.4f}  median={stats.median(d):.4f}  mode={stats.mode(d)}")
print(f"  variance={stats.variance(d):.4f}  std_dev={stats.std_dev(d):.4f}")
print(f"  geometric_mean={stats.geometric_mean(d):.4f}")
print(f"  harmonic_mean={stats.harmonic_mean(d):.4f}")
print(f"  MAD={stats.median_abs_dev(d):.4f}")
print(f"  skewness={stats.skewness(d):.4f}  kurtosis={stats.kurtosis(d):.4f}")
q=stats.quantiles(d,4)
print(f"  quartiles: Q1={q[0]:.2f}  Q2={q[1]:.2f}  Q3={q[2]:.2f}")
z=stats.z_scores(d)
print(f"  z_scores (first 4)={[round(v,3) for v in z[:4]]}")
print(f"  covariance(d,d2)={stats.covariance(d,d2):.4f}")
print(f"  correlation(d,d2)={stats.correlation(d,d2):.6f}")

# ════════════════════════════════════════════════════════
# 04  VECTOR CLASS
# ════════════════════════════════════════════════════════
hdr(4, "Vector class  (rmath.Vector)")
v1=Vector([1.,2.,3.,4.,5.])
v2=Vector([5.,4.,3.,2.,1.])
print(f"  v1={v1.to_list()}  v2={v2.to_list()}")
print(f"  v1+10 = {(v1+10.).to_list()}")
print(f"  v1+v2 = {(v1+v2).to_list()}")
print(f"  v1*v2 = {(v1*v2).to_list()}")
print(f"  v1-v2 = {(v1-v2).to_list()}")
print(f"  v1/2  = {(v1/2.).to_list()}")
print(f"  -v1   = {(-v1).to_list()}")
print(f"  v1@v2 = {v1@v2}  (dot)")
print(f"  add_vec   = {v1.add_vec(v2).to_list()}")
print(f"  mul_vec   = {v1.mul_vec(v2).to_list()}")
print(f"  dot       = {v1.dot(v2)}")

vr=Vector.range(1,11)
print(f"\n  range(1,11)          = {vr.to_list()}")
vl=Vector.linspace(0,C.PI,5)
print(f"  linspace(0,PI,5)     = {[round(x,4) for x in vl.to_list()]}")
print(f"  sum_range(100)       = {Vector.sum_range(100.):.0f}")
print(f"  sum_range(1,1000001) = {Vector.sum_range(1.,1_000_001.):.0f}")

sc=Vector([72.,85.,91.,63.,78.,88.,55.,94.,71.,82.])
print(f"\n  scores={sc.to_list()}")
print(f"  sum={sc.sum()}  prod={sc.prod():.4e}")
print(f"  mean={sc.mean():.2f}  min={sc.min():.1f}  max={sc.max():.1f}")
print(f"  variance={sc.variance():.4f}  std_dev={sc.std_dev():.4f}")

print(f"\n  Math chains:")
a=Vector.linspace(0,2*C.PI,360)
wave=a.sin().mul_scalar(10.)
print(f"  10*sin(0..2PI): mean={wave.mean():.6f}  std={wave.std_dev():.4f}")
chained=Vector.range(1,101).add_scalar(-50.).div_scalar(25.).tanh()
print(f"  ((1..100)-50)/25 → tanh: min={chained.min():.4f} max={chained.max():.4f}")

print(f"\n  Predicates:")
vm=Vector([1.,float('inf'),float('nan'),-2.,0.])
print(f"  isnan={vm.isnan()}  isinf={vm.isinf()}")
print(f"  is_prime [2..11]={Vector.range(2,12).is_prime()}")

print(f"\n  pow_scalar(2)={v1.pow_scalar(2.).to_list()}")
print(f"  clamp(2,4)={v1.clamp(2.,4.).to_list()}")
print(f"  is_empty={v1.is_empty()}  len={len(v1)}")

# ════════════════════════════════════════════════════════
# 05  VECTOR FUNCTIONAL API
# ════════════════════════════════════════════════════════
hdr(5, "Vector functional API  (rmath.vector.*)")
a=[1.,4.,9.,16.,25.]
b=[5.,4.,3.,2.,1.]
print(f"  sqrt({a})={vector.sqrt(a)}")
print(f"  add_scalar(+10)={vector.add_scalar(a,10.)}")
print(f"  add_vec={vector.add_vec(a,b)}")
print(f"  sub_vec={vector.sub_vec(a,b)}")
print(f"  mul_vec={vector.mul_vec(a,b)}")
print(f"  div_vec={vector.div_vec(a,b)}")
print(f"  dot={vector.dot(a,b)}")
print(f"  sum={vector.sum(a)}  mean={vector.mean(a)}")
print(f"  variance={vector.variance(a):.4f}  std_dev={vector.std_dev(a):.4f}")
print(f"  min={vector.min(a)}  max={vector.max(a)}")

# ════════════════════════════════════════════════════════
# 06  ARRAY  (all features)
# ════════════════════════════════════════════════════════
hdr(6, "2-D Array  (rmath.Array)")

print("  -- Constructors --")
print(f"  zeros(2,3) sum={Array.zeros(2,3).sum()}")
print(f"  ones(3,3) sum={Array.ones(3,3).sum()}")
I4=Array.eye(4); print(f"  eye(4) trace={I4.trace()}")
D=Array.diag([2.,3.,4.]); print(f"  diag([2,3,4]) det={D.det():.1f}")
F=Array.full(2,3,C.PI); print(f"  full(2,3,PI) mean={F.mean():.8f}")
Fl=Array.from_flat(list(range(1,13)),3,4)
print(f"  from_flat(1..12,3×4)={Fl.row(0)}…")

print("\n  -- Operators --")
M=Array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]])
N=Array([[2.,0.,1.],[1.,3.,1.],[0.,1.,2.]])
print(f"  M+1  row0={( M+1.).row(0)}")
print(f"  M*2  row0={(M*2.).row(0)}")
print(f"  M+N  row0={(M+N).row(0)}")
print(f"  M*N  row0={(M*N).row(0)}")
print(f"  -M   row0={(-M).row(0)}")
MN=M@N; print(f"  M@N  row0={[round(x,1) for x in MN.row(0)]}")
print(f"  reshape(3×4→4×3): {Fl.reshape(4,3).shape}")

print("\n  -- Linear Algebra --")
A3=Array([[2.,1.,-1.],[-3.,-1.,2.],[-2.,1.,2.]])
b3=[8.,-11.,-3.]
print(f"  det(A3)={A3.det():.2f}")
print(f"  trace(A3)={A3.trace():.2f}")
x_sol=A3.solve(b3)
print(f"  solve Ax=b → x={[round(v,6) for v in x_sol]}  (expect [2,3,-1])")
Ai=A3.inv()
Ich=A3@Ai
print(f"  A@inv(A) diagonal≈[1,1,1]: {[round(Ich.get(i,i),6) for i in range(3)]}")
print(f"  norm_fro={A3.norm_fro():.6f}  norm_l1={A3.norm_l1():.4f}  norm_inf={A3.norm_inf():.4f}")

print("\n  -- Structure --")
sq=Array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]])
print(f"  triu row0={sq.triu().row(0)}  tril row2={sq.tril().row(2)}")
print(f"  diag_extract={sq.diag_extract()}")

print("\n  -- Element-wise math --")
Em=Array([[0.,-1.,1.],[4.,9.,16.]])
print(f"  abs={Em.abs().row(0)}")
print(f"  sqrt  row1={Em.sqrt().row(1)}")
print(f"  ceil  row0={Em.ceil().row(0)}")

print("\n  -- ML Activations --")
L=Array([[-2.,0.,1.,3.],[1.,-1.,2.,-0.5]])
print(f"  sigmoid row0={[round(x,4) for x in L.sigmoid().row(0)]}")
print(f"  relu    row0={L.relu().row(0)}")
print(f"  leaky_relu(0.1) row0={[round(x,4) for x in L.leaky_relu(0.1).row(0)]}")
sm=L.softmax()
print(f"  softmax row0={[round(x,4) for x in sm.row(0)]}  sum={sum(sm.row(0)):.8f}")

print("\n  -- Predicates --")
Pn=Array([[1.,float('nan')],[float('inf'),2.]])
print(f"  any_nan={Pn.any_nan()}  all_finite={Pn.all_finite()}")

# ════════════════════════════════════════════════════════
# 07  FULL ML PIPELINE  (EDA → Preprocess → OLS → LogReg)
# ════════════════════════════════════════════════════════
hdr(7, "Full ML Pipeline  (400 students, 4 features)")

data_list, y_reg, y_cls = make_data(400)
n = len(data_list)
feat_names = ["study_hrs","sleep_hrs","prev_score","attendance"]
X  = Array(data_list)
yr_v = Vector(y_reg)
yc_v = Vector(y_cls)

print(f"  Samples={n}  Features={len(feat_names)}")
print(f"  Pass rate={sum(y_cls)/n*100:.1f}%")
print(f"  score: mean={yr_v.mean():.2f}  std={yr_v.std_dev():.2f}"
      f"  min={yr_v.min():.1f}  max={yr_v.max():.1f}")

print("\n  Feature stats (raw):")
for j,fn in enumerate(feat_names):
    col=X.col(j)
    print(f"    {fn:12s}: mean={stats.mean(col):7.2f}  "
          f"std={stats.std_dev(col):.2f}  "
          f"min={vector.min(col):.1f}  max={vector.max(col):.1f}")

print("\n  Per-axis Array ops:")
print(f"    col_sums[:3]  = {[round(v,2) for v in X.col_sums()[:3]]}")
print(f"    col_means[:3] = {[round(v,2) for v in X.col_means()[:3]]}")
print(f"    row_means[:3] = {[round(v,2) for v in X.row_means()[:3]]}")
print(f"    col_min[:3]   = {[round(v,2) for v in X.col_min()[:3]]}")
print(f"    col_max[:3]   = {[round(v,2) for v in X.col_max()[:3]]}")
print(f"    any_nan={X.any_nan()}  all_finite={X.all_finite()}")

# ── Preprocessing ──────────────────────────────────────
print("\n  MinMax scaling (col_normalize):")
Xmm = X.col_normalize()
print(f"    col_min={[round(v,4) for v in Xmm.col_min()]}")
print(f"    col_max={[round(v,4) for v in Xmm.col_max()]}")

print("\n  Standard scaling (col_z_normalize):")
Xsc = X.col_z_normalize()
print(f"    col_means≈0: {[round(v,8) for v in Xsc.col_means()]}")
print(f"    col_stds ≈1: {[round(v,6) for v in Xsc.col_std_dev()]}")

# ── Correlation Matrix ──────────────────────────────────
print("\n  Correlation matrix (4×4):")
cor = Xsc.correlation_matrix()
for i,fn in enumerate(feat_names):
    row=[round(cor.get(i,j),3) for j in range(4)]
    print(f"    {fn:12s}: {row}")

# ── OLS Regression ─────────────────────────────────────
print("\n  Linear Regression (OLS normal equations):")
Xb_data = [[1.]+Xsc.row(i) for i in range(n)]
Xb = Array(Xb_data)                       # 400×5 (bias + 4 features)
yc_data = [[v] for v in y_reg]
yc = Array(yc_data)                        # 400×1

XtX = Xb.transpose() @ Xb                 # 5×5
Xty = Xb.transpose() @ yc                 # 5×1
beta = XtX.solve(Xty.col(0))             # list of 5

labs = ["intercept"]+feat_names
for lb,bv in zip(labs,beta):
    print(f"    {lb:12s}= {bv:+.4f}")

bc = Array([[v] for v in beta])
yp_col = Xb @ bc
yp = yp_col.col(0)

res = vector.sub_vec(y_reg, yp)
ss_res = vector.dot(res, res)
ym = stats.mean(y_reg)
ss_tot = sum((v-ym)**2 for v in y_reg)
r2 = 1.0 - ss_res/ss_tot
res_v = Vector(res)
mae = res_v.abs().mean()
rmse = scalar.sqrt(ss_res/n)
print(f"    R²={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

# ── Logistic Regression ─────────────────────────────────
print("\n  Logistic Regression (gradient descent, 80 epochs):")
w = [0.0]*5; lr=0.1
ycc = Array([[v] for v in y_cls])
t0 = time.perf_counter()
for ep in range(80):
    wc = Array([[v] for v in w])
    yh = (Xb @ wc).sigmoid()
    res_c = yh.sub_array(ycc)
    g = [(v/n) for v in (Xb.transpose()@res_c).col(0)]
    w = [wi-lr*gi for wi,gi in zip(w,g)]
    if ep%20==0 or ep==79:
        loss_v = [y_cls[i]*scalar.log(max(yh.col(0)[i],C.EPSILON_F64))
                 +(1-y_cls[i])*scalar.log(max(1-yh.col(0)[i],C.EPSILON_F64))
                  for i in range(n)]
        print(f"    ep={ep:3d}  loss={-stats.mean(loss_v):.6f}")

print(f"    Trained in {time.perf_counter()-t0:.3f}s")
wc_f = Array([[v] for v in w])
probs = (Xb@wc_f).sigmoid().col(0)
preds = [1.0 if p>=0.5 else 0.0 for p in probs]
acc = sum(1 for a,b in zip(preds,y_cls) if a==b)/n
tp=sum(1 for a,b in zip(preds,y_cls) if a==1 and b==1)
fp=sum(1 for a,b in zip(preds,y_cls) if a==1 and b==0)
fn=sum(1 for a,b in zip(preds,y_cls) if a==0 and b==1)
pr=tp/(tp+fp) if tp+fp>0 else 0
re=tp/(tp+fn) if tp+fn>0 else 0
f1=2*pr*re/(pr+re) if pr+re>0 else 0
print(f"    Accuracy={acc*100:.2f}%  Precision={pr:.4f}  Recall={re:.4f}  F1={f1:.4f}")

# ════════════════════════════════════════════════════════
# 08  SIGNAL PROCESSING
# ════════════════════════════════════════════════════════
hdr(8, "Signal Processing  (Vector)")
t_v = Vector.linspace(0., 2*C.PI, 1000)
sig = t_v.sin() + (t_v*3.).sin().mul_scalar(0.5) + (t_v*5.).sin().mul_scalar(0.25)
print(f"  composite = sin(t) + 0.5*sin(3t) + 0.25*sin(5t)")
print(f"  n={sig.len()}  mean={sig.mean():.8f}  std={sig.std_dev():.6f}")
print(f"  min={sig.min():.6f}  max={sig.max():.6f}")
energy = (sig*sig).sum()
print(f"  energy={energy:.4f}")
hann = t_v.sin().pow_scalar(2.)
wind = sig*hann
print(f"  windowed (Hann): std={wind.std_dev():.6f}  energy={(wind*wind).sum():.4f}")

# ════════════════════════════════════════════════════════
# 09  FINANCIAL SIMULATION
# ════════════════════════════════════════════════════════
hdr(9, "Financial Simulation  (scalar + Vector + stats)")
random.seed(42); T=252
assets = {"Tech":(.0009,.020),"Bond":(.0001,.003),"Crypto":(.0012,.040)}
wts    = [0.5, 0.3, 0.2]
all_rets = {}
for (nm,(mu,sg)) in assets.items():
    daily=[]
    for _ in range(T):
        u1,u2=random.random(),random.random()
        z = scalar.sqrt(-2.*scalar.log(u1+1e-300))*scalar.cos(2*C.PI*u2)
        daily.append(mu + sg*z)
    all_rets[nm]=daily
    v=Vector(daily)
    ar=v.mean()*252; av=v.std_dev()*scalar.sqrt(252.)
    sh=ar/av if av>0 else 0
    print(f"  {nm:8s}: ann_ret={ar*100:+6.2f}%  vol={av*100:.2f}%  Sharpe={sh:.3f}")

port=[sum(w*all_rets[nm][d] for nm,w in zip(assets,wts)) for d in range(T)]
pv=Vector(port)
par=pv.mean()*252; pav=pv.std_dev()*scalar.sqrt(252.); psh=par/pav
log_r=Vector([scalar.log(1.+r) for r in port])
total=scalar.exp(log_r.sum())-1.
print(f"  Portfolio: ann={par*100:+.2f}%  vol={pav*100:.2f}%  "
      f"Sharpe={psh:.3f}  total={total*100:+.2f}%")
print(f"  skew={stats.skewness(port):.4f}  kurt={stats.kurtosis(port):.4f}")

# ════════════════════════════════════════════════════════
# 10  ARRAY CUMULATIVE & MISC
# ════════════════════════════════════════════════════════
hdr(10, "Cumulative, Norms, Softmax, Number Theory")
G=Array([[1.,2.,3.,4.],[5.,6.,7.,8.]])
print(f"  cumsum row0={G.cumsum().row(0)}")
print(f"  cumprod row0={[round(v) for v in G.cumprod().row(0)]}")

print(f"\n  Array normalizations on G:")
print(f"    normalize:     min={G.normalize().min():.4f}  max={G.normalize().max():.4f}")
print(f"    z_normalize:   mean={G.z_normalize().mean():.8f}  std={G.z_normalize().std_dev():.6f}")
print(f"    col_normalize min={G.col_normalize().col_min()}")
print(f"    col_z_normalize means={[round(v,8) for v in G.col_z_normalize().col_means()]}")

print(f"\n  Primes ≤ 50 via Vector.is_prime:")
cand=Vector.range(2.,51.)
primes=[int(v) for v,m in zip(cand.to_list(),cand.is_prime()) if m]
print(f"    {primes}")

print(f"\n  sum_range benchmarks:")
print(f"    sum(1..1,000,000)   = {Vector.sum_range(1.,1_000_001.):.0f}")
print(f"    sum(1..100,000,000) = {Vector.sum_range(1.,100_000_001.):.0f}")

# ════════════════════════════════════════════════════════
# DONE
# ════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  PROJECT COMPLETE — all rmath modules exercised")
print(f"  constants · scalar · stats · Vector · vector · Array")
print(f"{'='*60}\n")
