# Scientific model

## Scope

LangevinSANS represents a dilute ensemble of identical, spherical, uniformly
magnetized particles with fixed magnetic-moment magnitude. The particles do
not interact, have no anisotropy, and are in thermal equilibrium with a heat
bath at temperature `T`. A prescribed, spatially uniform external magnetic
flux density points along the laboratory `z` axis.

These assumptions define the standard classical Langevin model. They do not
describe blocked particles, hysteresis, relaxation times, particle-size
distributions, interparticle interactions, internal spin textures, or
anisotropy. In particular, the analytical curve is reversible and
single-valued.

## Equilibrium distribution

For a spherical particle of radius `R`, saturation magnetization `M_s`, and
volume

```math
V=\frac{4\pi R^3}{3},
```

the fixed moment magnitude is

```math
\mu=M_sV.
```

The Zeeman energy in the prescribed external induction
`B_ext = B_ext e_z` is

```math
E_Z=-\boldsymbol\mu\cdot\mathbf B_{\mathrm{ext}}.
```

Introducing

```math
\xi=\frac{\mu B_{\mathrm{ext}}}{k_{\mathrm B}T},
```

the orientational probability density is

```math
p(\theta,\phi\mid\xi)
=\frac{\xi}{4\pi\sinh\xi}\exp(\xi\cos\theta),
```

with the continuous zero-field limit `p = 1/(4 pi)`. The implementation uses
the exact SI value `k_B = 1.380649e-23 J K^-1`.

## First and second moments

The Langevin function is

```math
L(\xi)=\coth\xi-\frac{1}{\xi}.
```

For the field direction chosen above,

```math
\langle\mathbf m\rangle=(0,0,L(\xi)),
\qquad
\frac{M_z}{M_s}=L(\xi).
```

The dimensional curve returned by `LangevinResult.magnetization_a_per_m` is
the magnetization of the magnetic particle phase, `M_s L(xi)`. A bulk sample
average would additionally contain the magnetic volume fraction `phi`.

The raw second-moment tensor used by the SANS backend is

```math
\Gamma_{\alpha\beta}
=\langle m_\alpha m_\beta\rangle.
```

Its nonzero components have the closed form

```math
\Gamma_{xx}=\Gamma_{yy}=\frac{L(\xi)}{\xi},
\qquad
\Gamma_{zz}=1-2\frac{L(\xi)}{\xi}.
```

At zero field, `Gamma = I/3`. For every field,

```math
\operatorname{tr}\boldsymbol\Gamma=1.
```

The tensor is even in field, whereas `M_z` is odd. Consequently, this ideal
model predicts identical magnetic SANS at `+B_ext` and `-B_ext`. The centered
covariance is available separately as
`Gamma - <m><m>^T`; it must not replace the raw moment tensor in the cross
sections below.

Series expansions are used around `xi = 0` to avoid cancellation in both
`L(xi)` and `L(xi)/xi`.

## Magnetic SANS

The implementation contains only the magnetic particle contribution. Nuclear
coherent scattering, nuclear spin-incoherent background, structure factors,
finite beam polarization, polarization efficiency, and leakage corrections
are outside the model.

For spherical particles with magnetic volume fraction `phi`,

```math
C(q)=\phi\frac{16\pi^2R^6M_s^2b_H^2}{V}
\left[\frac{j_1(qR)}{qR}\right]^2,
\qquad b_H=2.91\times10^8\,\mathrm{A^{-1}m^{-1}}.
```

The reported differential cross sections are in `cm^-1`.

### Magnetic unpolarized cross section

For `B_ext` perpendicular to the incident neutron beam,

```math
\frac{d\Sigma_{M,\perp}}{d\Omega}=C(q)
\left(\Gamma_{xx}+\Gamma_{yy}\cos^2\theta
+\Gamma_{zz}\sin^2\theta
-2\Gamma_{yz}\sin\theta\cos\theta\right).
```

The detector convention is
`q = (0, q_y, q_z) = q(0, sin(theta), cos(theta))`: `q_z` is horizontal and
`q_y` vertical.

For `B_ext` parallel to the incident beam,

```math
\frac{d\Sigma_{M,\parallel}}{d\Omega}=C(q)
\left(\Gamma_{xx}\sin^2\theta
+\Gamma_{yy}\cos^2\theta+\Gamma_{zz}
-2\Gamma_{xy}\sin\theta\cos\theta\right).
```

Here `q = (q_x, q_y, 0) = q(cos(theta), sin(theta), 0)`: `q_x` is horizontal
and `q_y` vertical.

### Polarization-averaged spin-flip cross section

The polarization-independent average of the two spin-flip channels is

```math
\frac{d\Sigma_{\mathrm{sf}}}{d\Omega}
=\frac12\left(\frac{d\Sigma^{+-}}{d\Omega}
+\frac{d\Sigma^{-+}}{d\Omega}\right).
```

The chiral term cancels in this average. With field and polarization along the
laboratory `z` axis, the perpendicular expression is

```math
\frac{d\Sigma_{\mathrm{sf},\perp}}{d\Omega}=C(q)
\left(\Gamma_{xx}+\Gamma_{yy}\cos^4\theta
+\Gamma_{zz}\sin^2\theta\cos^2\theta
-2\Gamma_{yz}\sin\theta\cos^3\theta\right),
```

and the parallel expression is

```math
\frac{d\Sigma_{\mathrm{sf},\parallel}}{d\Omega}=C(q)
\left(\Gamma_{xx}\sin^2\theta
+\Gamma_{yy}\cos^2\theta
-2\Gamma_{xy}\sin\theta\cos\theta\right).
```

The analytical azimuthal moments are

```math
\langle A_{M,\perp}\rangle
=\Gamma_{xx}+\frac12(\Gamma_{yy}+\Gamma_{zz}),
\qquad
\langle A_{M,\parallel}\rangle
=\frac12(\Gamma_{xx}+\Gamma_{yy})+\Gamma_{zz},
```

and

```math
\langle A_{\mathrm{sf},\perp}\rangle
=\Gamma_{xx}+\frac38\Gamma_{yy}+\frac18\Gamma_{zz},
\qquad
\langle A_{\mathrm{sf},\parallel}\rangle
=\frac12(\Gamma_{xx}+\Gamma_{yy}).
```

The same factors are used for the azimuthal `I(q)`, field-dependent intensity,
and spherical-particle `p(r)` and `c(r)=p(r)/r^2` curves.

## Field convention and magnetostatics

`B_ext` is the externally prescribed quantity that enters the single-particle
Zeeman energy. The model does not derive it from a sample magnetization and
does not assume the generally non-universal relation
`B = mu_0 (H + M)` as a starting point. Demagnetizing fields, magnetostatic
interactions, and a self-consistent internal field are outside this first
equilibrium reference model.

## Planned extensions

A browser interface can expose particle parameters, temperature, field range,
the magnetization and `Gamma_kk` curves, a field selector, and the same SANS
maps and real/reciprocal-space curves as SWSANS. A stochastic model with noise
and anisotropy should be implemented as a separate backend so that its
dynamics and equilibrium assumptions remain explicit.
