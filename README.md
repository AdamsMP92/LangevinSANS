# LangevinSANS

Analytical equilibrium magnetization and magnetic small-angle neutron
scattering (SANS) for dilute ensembles of identical, noninteracting spherical
macrospins.

The first release is a compact reference implementation of the standard
Langevin model. It evaluates the magnetization and the complete second-moment
tensor analytically as functions of the externally applied magnetic flux
density, then uses the same detector conventions as SWSANS for magnetic
unpolarized and polarization-averaged spin-flip SANS.

[Open the browser app](https://adamsmp92.github.io/LangevinSANS/) ·
[View the source](https://github.com/AdamsMP92/LangevinSANS)

## What is included

- a numerically stable Langevin function;
- the exact normalized equilibrium orientational distribution;
- analytical first moments and
  `Gamma_kl = <m_k m_l>` for a field along the laboratory `z` axis;
- reversible field curves for reduced and dimensional magnetization;
- two-dimensional magnetic SANS detector maps in perpendicular and parallel
  geometry;
- azimuthally averaged `I(q)`, `p(r)`, and `c(r) = p(r) / r^2`;
- magnetic unpolarized and polarization-averaged spin-flip cross sections;
- field scans of the azimuthally averaged SANS intensity;
- a Shiny/Shinylive interface that runs entirely in the browser;
- a reproducible Pixi environment and a focused scientific test suite.

The standard Langevin model is an equilibrium model. It therefore predicts a
single-valued, reversible magnetization curve—not coercivity, remanence, or
hysteresis.

## Quick start

```bash
pixi run test
pixi run app
```

The second command serves the local app at `http://127.0.0.1:8000`. The model
selector currently exposes the exact analytical solution. Its dispatch layer
is prepared for a separate future simulation backend without changing the
analytical implementation.

```python
from langevinsans import (
    FieldGrid,
    ParticleProperties,
    SANSConfig,
    langevin_curve,
    sans_map,
)

particle = ParticleProperties(
    radius_nm=5.0,
    saturation_magnetization_a_per_m=3.1e5,
    temperature_k=300.0,
)
result = langevin_curve(particle, FieldGrid(field_max_t=1.0, n_field_points=201))

# SANS at the largest positive field.
qx, qy, intensity = sans_map(
    result.gamma[-1],
    particle,
    SANSConfig(volume_fraction=0.01),
    geometry="perpendicular",
    cross_section="spin_flip",
)
```

The model, units, coordinate conventions, and analytical formulas are given
in [docs/model.md](docs/model.md).

## Development

```bash
pixi run format
pixi run check
pixi run export-site
```

Pixi supports the locked macOS ARM64 and Linux 64-bit environments. The Python
package itself requires Python 3.12 and NumPy.

## Scope and roadmap

This version assumes identical fixed-magnitude moments, no anisotropy, no
interparticle interaction, and thermal equilibrium in a prescribed uniform
external field. The browser interface follows the SWSANS layout and adds an
exact ODF view and an azimuthally averaged SANS field scan. A separate future
model may add stochastic dynamics, thermal noise, and particle anisotropy;
those effects will not be folded silently into the analytical Langevin
reference model.

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff). If the software
is archived with Zenodo, cite the exact released version and its version DOI.

## License

MIT
