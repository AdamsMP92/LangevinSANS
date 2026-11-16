"""Browser-native Shiny interface for LangevinSANS."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from shiny import App, Inputs, Outputs, Session, reactive, render, req, ui

from langevinsans import (
    FieldGrid,
    InputValidationError,
    LangevinResult,
    ParticleProperties,
    SANSConfig,
    direction_cosine_density,
    langevin_curve,
    pair_distance_curves,
    sans_field_curve,
    sans_intensity_curve,
    sans_map,
)

APP_DIR = Path(__file__).resolve().parent

MODEL_CHOICES = {
    "analytical": "Analytical Langevin · exact equilibrium",
}


def _field_point_choices(result: LangevinResult) -> dict[str, str]:
    total = result.field_t.size
    return {
        str(index): (
            f"{index + 1:03d}/{total:03d} · B = {field:+.4f} T · "
            f"ξ = {result.xi[index]:+.3f} · m_z = "
            f"{result.reduced_magnetization[index]:+.4f}"
        )
        for index, field in enumerate(result.field_t)
    }


def _numeric_field(
    input_id: str,
    label: str,
    value: float,
    *,
    step: float,
    note: str,
    minimum: float | None = None,
) -> ui.Tag:
    return ui.div(
        ui.input_numeric(
            input_id,
            label,
            value,
            min=minimum,
            step=step,
            update_on="change",
        ),
        ui.tags.small(note, class_="field-note"),
        class_="field-block",
    )


def _metric_card(label: str, value: str, detail: str, accent: str) -> ui.Tag:
    return ui.div(
        ui.div(
            ui.tags.span(label, class_="metric-label"),
            ui.tags.span(class_=f"metric-dot metric-dot-{accent}"),
            class_="metric-heading",
        ),
        ui.tags.strong(value, class_="metric-value"),
        ui.tags.p(detail, class_="metric-detail"),
        class_=f"metric-card metric-card-{accent}",
    )


def _equation(latex: str, caption: str) -> ui.Tag:
    return ui.div(
        ui.HTML(rf"\[{latex}\]"),
        ui.tags.p(caption),
        class_="equation-block",
    )


def _evaluate_model(
    model_name: str,
    particle: ParticleProperties,
    field_grid: FieldGrid,
) -> LangevinResult:
    """Dispatch a model without coupling the UI to its implementation."""

    if model_name == "analytical":
        return langevin_curve(particle, field_grid)
    raise InputValidationError(f"Unknown evaluation model: {model_name}")


controls = ui.div(
    ui.div(
        ui.tags.span("EQUILIBRIUM MODEL", class_="eyebrow"),
        ui.h2("Model inputs"),
        ui.tags.p(
            "Evaluate the exact Langevin equilibrium distribution and its "
            "field-dependent first and second moments.",
            class_="section-intro",
        ),
        class_="panel-heading",
    ),
    ui.div(
        ui.input_select(
            "model",
            "Evaluation model",
            MODEL_CHOICES,
            selected="analytical",
        ),
        ui.tags.p(
            "The selector is intentionally model-aware. A stochastic simulation "
            "backend can be added later without changing the analytical path.",
            class_="model-note",
        ),
        class_="field-grid field-grid-one",
    ),
    ui.div(
        _numeric_field(
            "radius_nm",
            "Particle radius R [nm]",
            5.0,
            step=0.5,
            minimum=0.1,
            note="Spherical particle volume V = 4πR³/3",
        ),
        _numeric_field(
            "magnetization",
            "Saturation magnetization Mₛ [kA/m]",
            310.0,
            step=10.0,
            minimum=0.1,
            note="Magnetic moment μ = MₛV",
        ),
        _numeric_field(
            "temperature",
            "Temperature T [K]",
            300.0,
            step=5.0,
            minimum=0.1,
            note="Thermal scale kBT",
        ),
        class_="field-grid field-grid-one",
    ),
    ui.div(class_="panel-divider"),
    ui.tags.span("FIELD GRID", class_="eyebrow"),
    ui.div(
        _numeric_field(
            "field_max",
            "Maximum external induction [T]",
            1.0,
            step=0.1,
            minimum=0.001,
            note="Symmetric grid from −Bmax to +Bmax",
        ),
        _numeric_field(
            "field_points",
            "Field points",
            201,
            step=2,
            minimum=5,
            note="Odd values include the exact zero-field state",
        ),
        class_="field-grid field-grid-one",
    ),
    ui.input_action_button(
        "run",
        "Evaluate analytical model",
        class_="run-button",
    ),
    ui.output_ui("calculation_message"),
    class_="control-card",
)

results = ui.div(
    ui.output_ui("metric_cards"),
    ui.div(
        ui.div(
            ui.div(
                ui.tags.span("MAGNETIZATION", class_="eyebrow"),
                ui.h3("Reversible equilibrium curve"),
                class_="plot-heading",
            ),
            ui.output_plot("magnetization_plot", height="390px"),
            class_="plot-card",
        ),
        ui.div(
            ui.div(
                ui.tags.span("SECOND MOMENTS", class_="eyebrow"),
                ui.h3("Correlation tensor & anisotropy"),
                class_="plot-heading",
            ),
            ui.output_plot("moments_plot", height="390px"),
            class_="plot-card",
        ),
        class_="plot-grid",
    ),
    ui.div(
        ui.div(
            ui.tags.span("MAGNETIC SANS", class_="eyebrow"),
            ui.h2("Detector response"),
            ui.tags.p(
                "Evaluate any analytical field point for a dilute ensemble of "
                "uniformly magnetized spherical particles.",
                class_="section-intro",
            ),
            class_="panel-heading sans-heading",
        ),
        ui.div(
            ui.input_radio_buttons(
                "cross_section",
                "Magnetic cross section",
                {
                    "unpolarized": "Unpolarized",
                    "spin_flip": "Spin-flip average",
                },
                selected="unpolarized",
                inline=True,
            ),
            ui.tags.p(
                "Choose the unpolarized magnetic signal or "
                "½(Σ⁺⁻ + Σ⁻⁺), for which the chiral difference cancels.",
                class_="cross-section-note",
            ),
            class_="cross-section-switch",
        ),
        ui.div(
            ui.input_select(
                "sans_state",
                "Calculated field point",
                {"auto": "Positive maximum field"},
                selected="auto",
            ),
            ui.input_select(
                "geometry",
                "Scattering geometry",
                {
                    "perpendicular": "B ⟂ k₀",
                    "parallel": "B ∥ k₀",
                },
                selected="perpendicular",
            ),
            ui.input_select(
                "reciprocal_scale",
                "Reciprocal-space scaling",
                {"log": "Logarithmic", "linear": "Linear"},
                selected="log",
            ),
            _numeric_field(
                "volume_fraction",
                "Magnetic volume fraction φ",
                0.01,
                step=0.005,
                minimum=0.000001,
                note="Absolute intensity scales linearly with φ",
            ),
            _numeric_field(
                "q_max",
                "Detector qmax [nm⁻¹]",
                1.0,
                step=0.1,
                minimum=0.01,
                note="Symmetric detector extent",
            ),
            _numeric_field(
                "probe_q",
                "Field-scan q [nm⁻¹]",
                0.1,
                step=0.01,
                minimum=0.0,
                note="Azimuthally averaged intensity versus field",
            ),
            class_="sans-controls",
        ),
        ui.div(
            ui.div(
                ui.output_plot("sans_map_plot", height="450px"),
                class_="plot-card plot-card-flat",
            ),
            ui.div(
                ui.output_plot("sans_curves_plot", height="450px"),
                class_="plot-card plot-card-flat",
            ),
            class_="plot-grid sans-plot-grid",
        ),
        class_="sans-card",
    ),
    ui.div(
        ui.div(
            ui.tags.span("EQUILIBRIUM STRUCTURE", class_="eyebrow"),
            ui.h3("Exact ODF & SANS field scan"),
            ui.output_plot("odf_field_plot", height="380px"),
            class_="diagnostic-card",
        ),
        ui.div(
            ui.tags.span("REFERENCE STATES", class_="eyebrow"),
            ui.h3("Analytical state summary"),
            ui.output_ui("state_table"),
            ui.download_button(
                "download_csv",
                "Download complete field curve",
                class_="download-button",
            ),
            class_="diagnostic-card",
        ),
        class_="diagnostic-grid",
    ),
    class_="results-column",
)

simulator_panel = ui.div(controls, results, class_="simulator-grid")

documentation_panel = ui.div(
    ui.div(
        ui.tags.span("MODEL & DOCUMENTATION", class_="eyebrow"),
        ui.h2("From the exact Langevin ODF to magnetic SANS"),
        ui.tags.p(
            "The analytical backend evaluates the normalized equilibrium "
            "orientation distribution, its raw second moments, and the same "
            "magnetic SANS detector conventions used by SWSANS.",
            class_="docs-lead",
        ),
        class_="docs-hero",
    ),
    ui.div(
        ui.tags.section(
            ui.tags.span("01", class_="docs-index"),
            ui.h3("Scope and assumptions"),
            ui.tags.ul(
                ui.tags.li("Identical spherical particles with fixed moment μ = MₛV."),
                ui.tags.li("Thermal equilibrium in a prescribed external induction."),
                ui.tags.li("No anisotropy, interactions, blocking, or hysteresis."),
                ui.tags.li("Applied field along the laboratory z axis."),
            ),
            class_="docs-section",
        ),
        ui.tags.section(
            ui.tags.span("02", class_="docs-index"),
            ui.h3("Exact orientational distribution"),
            _equation(
                r"p(\theta,\phi\mid\xi)=\frac{\xi}{4\pi\sinh\xi}"
                r"\exp(\xi\cos\theta),\qquad "
                r"\xi=\frac{\mu B_{\mathrm{ext}}}{k_{\mathrm B}T}",
                "The zero-field limit is the uniform distribution 1/(4π).",
            ),
            ui.tags.p(
                "The ODF plot displays its normalized marginal in "
                "u = cos θ for the selected field point."
            ),
            class_="docs-section",
        ),
        ui.tags.section(
            ui.tags.span("03", class_="docs-index"),
            ui.h3("Exact first and second moments"),
            _equation(
                r"L(\xi)=\coth\xi-\frac{1}{\xi},\qquad "
                r"\langle\mathbf m\rangle=(0,0,L(\xi))",
                "The magnetization curve is single-valued and reversible.",
            ),
            _equation(
                r"\Gamma_{xx}=\Gamma_{yy}=\frac{L(\xi)}{\xi},\qquad "
                r"\Gamma_{zz}=1-2\frac{L(\xi)}{\xi}",
                "Gamma contains raw moments <mαmβ>; tr Gamma = 1.",
            ),
            class_="docs-section",
        ),
        ui.tags.section(
            ui.tags.span("04", class_="docs-index"),
            ui.h3("Raw moments in SANS"),
            _equation(
                r"\left\langle\widetilde M_\alpha(\mathbf q)"
                r"\widetilde M_\beta^*(\mathbf q)\right\rangle"
                r"\propto |F(q)|^2\Gamma_{\alpha\beta}",
                "The centered covariance is not substituted for Gamma.",
            ),
            ui.tags.p(
                "A fluctuation-only or saturation-subtracted observable would "
                "require a separate, explicitly defined subtraction."
            ),
            class_="docs-section",
        ),
        ui.tags.section(
            ui.tags.span("05", class_="docs-index"),
            ui.h3("Perpendicular magnetic SANS"),
            _equation(
                r"\frac{d\Sigma_{M,\perp}}{d\Omega}=C(q)"
                r"\left[\Gamma_{xx}+\Gamma_{yy}\cos^2\theta+"
                r"\Gamma_{zz}\sin^2\theta-2\Gamma_{yz}"
                r"\sin\theta\cos\theta\right]",
                "The detector displays qz horizontally and qy vertically.",
            ),
            class_="docs-section",
        ),
        ui.tags.section(
            ui.tags.span("06", class_="docs-index"),
            ui.h3("Parallel magnetic SANS"),
            _equation(
                r"\frac{d\Sigma_{M,\parallel}}{d\Omega}=C(q)"
                r"\left[\Gamma_{xx}\sin^2\theta+\Gamma_{yy}\cos^2\theta+"
                r"\Gamma_{zz}-2\Gamma_{xy}\sin\theta\cos\theta\right]",
                "The detector displays qx horizontally and qy vertically.",
            ),
            class_="docs-section",
        ),
        ui.tags.section(
            ui.tags.span("07", class_="docs-index"),
            ui.h3("Polarization-averaged spin-flip SANS"),
            _equation(
                r"\frac{d\Sigma_{\mathrm{sf}}}{d\Omega}"
                r"=\frac12\left(\frac{d\Sigma^{+-}}{d\Omega}+"
                r"\frac{d\Sigma^{-+}}{d\Omega}\right)",
                "Averaging removes the chiral term.",
            ),
            ui.tags.p(
                "Only the magnetic polarization-independent spin-flip term is "
                "included. Nuclear background, finite polarization, and leakage "
                "corrections are outside this model."
            ),
            class_="docs-section",
        ),
        ui.tags.section(
            ui.tags.span("08", class_="docs-index"),
            ui.h3("Future simulation backend"),
            ui.tags.p(
                "The model selector and result contract are prepared for a "
                "separate stochastic backend with thermal noise and anisotropy. "
                "Such a model will retain its own assumptions, diagnostics, and "
                "numerical controls rather than altering the exact analytical path."
            ),
            class_="docs-section",
        ),
        ui.tags.section(
            ui.tags.span("09", class_="docs-index"),
            ui.h3("External-field convention"),
            ui.tags.p(
                "Bext is the prescribed external induction entering the Zeeman "
                "energy. The model does not derive it from the sample "
                "magnetization and does not assume B = μ₀(H + M) as a universal "
                "starting identity."
            ),
            class_="docs-section docs-section-wide",
        ),
    ),
    ui.div(
        ui.tags.span("SOURCE", class_="eyebrow"),
        ui.h3("Open scientific implementation"),
        ui.tags.p(
            "The analytical formulas, SANS conventions, tests, and browser "
            "interface are versioned together in the LangevinSANS repository."
        ),
        ui.tags.a(
            "View source code",
            href="https://github.com/AdamsMP92/LangevinSANS",
            target="_blank",
            rel="noopener noreferrer",
            class_="source-link",
        ),
        ui.tags.a(
            "Compare with SWSANS",
            href="https://adamsmp92.github.io/SWSANS_python/",
            target="_blank",
            rel="noopener noreferrer",
            class_="source-link source-link-secondary",
        ),
        class_="source-card",
    ),
    class_="documentation-panel",
)

app_ui = ui.page_bootstrap(
    ui.head_content(
        ui.tags.meta(
            name="description",
            content=(
                "Interactive analytical Langevin magnetization and magnetic SANS "
                "calculator."
            ),
        ),
        ui.tags.link(rel="icon", type="image/svg+xml", href="magnet.svg"),
        ui.tags.link(rel="preconnect", href="https://fonts.googleapis.com"),
        ui.tags.link(
            rel="preconnect",
            href="https://fonts.gstatic.com",
            crossorigin="anonymous",
        ),
        ui.tags.link(
            rel="stylesheet",
            href=(
                "https://fonts.googleapis.com/css2?"
                "family=DM+Sans:wght@400;500;600;700&"
                "family=Newsreader:opsz,wght@6..72,600&display=swap"
            ),
        ),
        ui.tags.script(
            id="MathJax-script",
            src="https://cdn.jsdelivr.net/npm/mathjax@4.1.3/tex-chtml.js",
            defer=True,
        ),
    ),
    ui.include_css(APP_DIR / "www" / "styles.css"),
    ui.div(
        ui.div(
            ui.tags.header(
                ui.div(
                    ui.div(
                        ui.img(
                            src="magnet.svg",
                            alt="Red horseshoe magnet",
                            class_="hero-mark",
                        ),
                        ui.div(
                            ui.tags.span(
                                "MICHAEL P. ADAMS · SCIENTIFIC SOFTWARE",
                                class_="hero-kicker",
                            ),
                            ui.h1("LangevinSANS"),
                            ui.tags.p(
                                "Connect the exact equilibrium Langevin "
                                "distribution, correlation moments, and magnetic "
                                "SANS detector response."
                            ),
                        ),
                        class_="hero-copy",
                    ),
                    ui.div(
                        ui.tags.span(class_="runtime-dot"),
                        ui.div(
                            ui.tags.strong("Browser-native Python"),
                            ui.tags.small("Shinylive · no calculation server"),
                        ),
                        class_="runtime-badge",
                    ),
                    class_="hero-inner",
                ),
                class_="app-hero",
            ),
            ui.div(
                ui.navset_card_underline(
                    ui.nav_panel("Calculator", simulator_panel, value="calculator"),
                    ui.nav_panel(
                        "Model & documentation",
                        documentation_panel,
                        value="documentation",
                    ),
                    id="main_navigation",
                    selected="calculator",
                ),
                class_="main-nav-wrap",
            ),
            ui.tags.footer(
                ui.tags.span("LangevinSANS"),
                ui.tags.span("·"),
                ui.tags.span("Michael P. Adams"),
                ui.tags.span("·"),
                ui.tags.a(
                    "Source code",
                    href="https://github.com/AdamsMP92/LangevinSANS",
                    target="_blank",
                    rel="noopener noreferrer",
                ),
                class_="app-footer",
            ),
            class_="app-shell",
        ),
        class_="container-fluid app-page",
    ),
    title="LangevinSANS",
    lang="en",
)


def _style_axes(axis: plt.Axes) -> None:
    axis.set_facecolor("#ffffff")
    axis.grid(True, color="#e8eaf3", linewidth=0.8)
    axis.tick_params(colors="#667085", labelsize=9)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#cfd4e3")


def server(input: Inputs, output: Outputs, session: Session) -> None:
    def _collect_particle() -> ParticleProperties:
        return ParticleProperties(
            radius_nm=float(input.radius_nm()),
            saturation_magnetization_a_per_m=float(input.magnetization()) * 1.0e3,
            temperature_k=float(input.temperature()),
        )

    def _collect_field_grid() -> FieldGrid:
        return FieldGrid(
            field_max_t=float(input.field_max()),
            n_field_points=int(input.field_points()),
        )

    def _collect_sans_config() -> SANSConfig:
        return SANSConfig(
            volume_fraction=float(input.volume_fraction()),
            q_max_per_nm=float(input.q_max()),
        )

    @reactive.calc
    @reactive.event(input.run, ignore_none=False)
    def calculation() -> tuple[LangevinResult | None, str | None]:
        try:
            result = _evaluate_model(
                str(input.model()),
                _collect_particle(),
                _collect_field_grid(),
            )
            return result, None
        except (InputValidationError, ArithmeticError, ValueError) as exc:
            return None, str(exc)
        except Exception as exc:
            return None, f"Model evaluation failed: {exc}"

    def _result() -> LangevinResult:
        result, _ = calculation()
        req(result is not None)
        assert result is not None
        return result

    @reactive.effect
    def _update_sans_field_choices() -> None:
        result = _result()
        ui.update_select(
            "sans_state",
            choices=_field_point_choices(result),
            selected=str(result.field_t.size - 1),
        )

    @reactive.calc
    def selected_index() -> int:
        result = _result()
        raw_index = str(input.sans_state())
        try:
            index = int(raw_index)
        except ValueError:
            index = result.field_t.size - 1
        return min(max(index, 0), result.field_t.size - 1)

    @output
    @render.ui
    def calculation_message() -> ui.Tag:
        result, error = calculation()
        if error is not None:
            return ui.div(error, class_="message message-error")
        req(result is not None)
        return ui.div(
            "Exact closed-form equilibrium evaluation complete; no numerical "
            "relaxation or orientation sampling was required.",
            class_="message message-success",
        )

    @output
    @render.ui
    def metric_cards() -> ui.Tag:
        result = _result()
        trace_error = float(
            np.max(np.abs(np.trace(result.gamma, axis1=-2, axis2=-1) - 1.0))
        )
        return ui.div(
            _metric_card(
                "PARTICLE MOMENT",
                f"{result.particle.magnetic_moment_a_m2:.3e}",
                "μ = MₛV [A m²]",
                "violet",
            ),
            _metric_card(
                "ξ AT +BMAX",
                f"{result.xi[-1]:.3f}",
                "μBmax / (kBT)",
                "cyan",
            ),
            _metric_card(
                "REDUCED M(+BMAX)",
                f"{result.reduced_magnetization[-1]:.4f}",
                f"M = {result.magnetization_a_per_m[-1] / 1.0e3:.2f} kA/m",
                "gold",
            ),
            _metric_card(
                "TRACE ERROR",
                f"{trace_error:.1e}",
                "max |tr Γ − 1|",
                "green",
            ),
            class_="metric-grid",
        )

    @output
    @render.plot
    def magnetization_plot():
        result = _result()
        index = selected_index()
        figure, axis = plt.subplots(figsize=(7.2, 4.0))
        axis.plot(
            result.field_t,
            result.reduced_magnetization,
            color="#6656d9",
            linewidth=2.5,
            label=r"$M_z/M_s=L(\xi)$",
        )
        axis.scatter(
            [result.field_t[index]],
            [result.reduced_magnetization[index]],
            color="#d99632",
            edgecolor="white",
            linewidth=1.0,
            s=70,
            zorder=4,
            label="selected SANS state",
        )
        axis.axhline(0.0, color="#aeb5c4", linewidth=0.8)
        axis.axvline(0.0, color="#dfe3ed", linewidth=0.8)
        axis.set_xlabel(r"External induction $B_{\mathrm{ext}}$ [T]")
        axis.set_ylabel("Reduced magnetization")
        axis.set_ylim(-1.05, 1.05)
        axis.legend(frameon=False, fontsize=9)
        _style_axes(axis)
        figure.tight_layout()
        return figure

    @output
    @render.plot
    def moments_plot():
        result = _result()
        index = selected_index()
        figure, axes = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True)
        colors = ("#6656d9", "#17a7b5", "#d99632")
        labels = (r"$\Gamma_{xx}$", r"$\Gamma_{yy}$", r"$\Gamma_{zz}$")
        for component, (color, label) in enumerate(zip(colors, labels, strict=True)):
            axes[0].plot(
                result.field_t,
                result.gamma[:, component, component],
                color=color,
                linewidth=2.0,
                label=label,
            )
        axes[0].axvline(result.field_t[index], color="#cfd4e3", linewidth=0.9)
        axes[0].set_ylabel("Raw second moment")
        axes[0].set_ylim(-0.03, 1.03)
        axes[0].legend(frameon=False, fontsize=9, ncol=3)
        axes[1].plot(
            result.field_t,
            result.anisotropy_parameter_perpendicular,
            color="#d95670",
            linewidth=2.0,
        )
        axes[1].axhline(1.0, color="#aeb5c4", linewidth=0.8, linestyle="--")
        axes[1].axvline(result.field_t[index], color="#cfd4e3", linewidth=0.9)
        axes[1].set_xlabel(r"External induction $B_{\mathrm{ext}}$ [T]")
        axes[1].set_ylabel(r"$A_\perp$")
        for axis in axes:
            _style_axes(axis)
        figure.tight_layout()
        return figure

    @output
    @render.plot
    def sans_map_plot():
        result = _result()
        index = selected_index()
        config = _collect_sans_config()
        geometry = str(input.geometry())
        scale = str(input.reciprocal_scale())
        cross_section = str(input.cross_section())
        x_axis, y_axis, intensity = sans_map(
            result.gamma[index],
            result.particle,
            config,
            geometry,
            cross_section,
        )
        positive = intensity[intensity > 0.0]
        maximum = float(np.max(positive)) if positive.size else 1.0
        minimum = max(
            float(np.min(positive)) if positive.size else 1.0e-8,
            maximum * 1.0e-4,
        )
        is_logarithmic = scale == "log"
        displayed_intensity = (
            np.maximum(intensity, minimum) if is_logarithmic else intensity
        )
        norm = (
            LogNorm(vmin=minimum, vmax=maximum)
            if is_logarithmic
            else Normalize(vmin=0.0, vmax=maximum)
        )
        figure, axis = plt.subplots(figsize=(5.5, 4.7))
        figure.subplots_adjust(left=0.17, right=0.84, bottom=0.15, top=0.88)
        image = axis.imshow(
            displayed_intensity,
            origin="lower",
            extent=[x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]],
            cmap="magma",
            norm=norm,
            interpolation="bilinear",
        )
        if not positive.size:
            axis.text(
                0.5,
                0.5,
                "Zero magnetic signal",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="white",
                fontsize=10,
            )
        if geometry == "perpendicular":
            axis.set_xlabel(r"$q_z$ [nm$^{-1}$]")
            axis.set_ylabel(r"$q_y$ [nm$^{-1}$]", labelpad=8)
            geometry_title = r"$B\perp k_0$"
        else:
            axis.set_xlabel(r"$q_x$ [nm$^{-1}$]")
            axis.set_ylabel(r"$q_y$ [nm$^{-1}$]", labelpad=8)
            geometry_title = r"$B\parallel k_0$"
        channel_title = (
            "Spin-flip average" if cross_section == "spin_flip" else "Unpolarized"
        )
        axis.set_title(
            f"{channel_title} · point {index + 1}/{result.field_t.size} · "
            f"B = {result.field_t[index]:+.4f} T · {geometry_title}",
            fontsize=10.5,
        )
        axis.set_aspect("equal")
        colorbar = figure.colorbar(image, ax=axis, fraction=0.047, pad=0.04)
        colorbar_label = (
            r"$d\Sigma_{\mathrm{sf}}/d\Omega$ [cm$^{-1}$]"
            if cross_section == "spin_flip"
            else r"$d\Sigma_M/d\Omega$ [cm$^{-1}$]"
        )
        colorbar.set_label(colorbar_label)
        return figure

    @output
    @render.plot
    def sans_curves_plot():
        result = _result()
        index = selected_index()
        config = _collect_sans_config()
        geometry = str(input.geometry())
        scale = str(input.reciprocal_scale())
        cross_section = str(input.cross_section())
        spacing = "linear" if scale == "linear" else "log"
        q, intensity = sans_intensity_curve(
            result.gamma[index],
            result.particle,
            config,
            geometry,
            spacing=spacing,
            cross_section=cross_section,
        )
        r, pair, correlation = pair_distance_curves(
            result.gamma[index],
            result.particle,
            config,
            geometry,
            cross_section,
        )
        figure, axes = plt.subplots(3, 1, figsize=(6.2, 5.0))
        axes[0].plot(q, intensity, color="#6656d9", linewidth=2.0)
        intensity_label = (
            r"$I_{\mathrm{sf}}(q)$ [cm$^{-1}$]"
            if cross_section == "spin_flip"
            else r"$I_M(q)$ [cm$^{-1}$]"
        )
        axes[0].set_ylabel(intensity_label)
        axes[1].plot(r, pair, color="#17a7b5", linewidth=2.0)
        axes[1].set_ylabel(r"$p(r)$ [nm$^{-2}$]")
        axes[2].plot(r, correlation, color="#d99632", linewidth=2.0)
        axes[2].set_ylabel(r"$c(r)$ [nm$^{-4}$]")
        axes[2].set_xlabel("r [nm]")
        axes[0].set_xlabel(r"q [nm$^{-1}$]")
        for axis in axes:
            _style_axes(axis)
        for axis in axes[1:]:
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
        if scale == "log":
            axes[0].set_xscale("log")
            if np.any(intensity > 0.0):
                axes[0].set_yscale("log")
            else:
                axes[0].set_ylim(-0.05, 1.0)
                axes[0].set_yticks([0.0])
        else:
            axes[0].set_xlim(0.0, config.q_max_per_nm)
        figure.tight_layout()
        return figure

    @output
    @render.plot
    def odf_field_plot():
        result = _result()
        index = selected_index()
        config = _collect_sans_config()
        geometry = str(input.geometry())
        cross_section = str(input.cross_section())
        probe_q = float(input.probe_q())

        direction_cosine = np.linspace(-1.0, 1.0, 401)
        density = direction_cosine_density(direction_cosine, float(result.xi[index]))
        field, intensity = sans_field_curve(
            result.field_t,
            result.gamma,
            result.particle,
            config,
            geometry,
            probe_q,
            cross_section,
        )

        figure, axes = plt.subplots(2, 1, figsize=(6.6, 4.2))
        axes[0].plot(direction_cosine, density, color="#6656d9", linewidth=2.0)
        axes[0].fill_between(
            direction_cosine,
            density,
            color="#eceafd",
            alpha=0.9,
        )
        axes[0].axvline(
            result.reduced_magnetization[index],
            color="#d99632",
            linestyle="--",
            linewidth=1.2,
            label=r"$\langle\cos\theta\rangle$",
        )
        axes[0].set_xlabel(r"$u=\cos\theta$")
        axes[0].set_ylabel(r"$p(u\mid\xi)$")
        axes[0].set_title(
            f"Exact ODF marginal · ξ = {result.xi[index]:+.3f}",
            fontsize=10,
        )
        axes[0].legend(frameon=False, fontsize=8)

        axes[1].plot(field, intensity, color="#17a7b5", linewidth=2.0)
        axes[1].scatter(
            [field[index]],
            [intensity[index]],
            color="#d99632",
            edgecolor="white",
            linewidth=0.8,
            s=45,
            zorder=4,
        )
        axes[1].set_xlabel(r"External induction $B_{\mathrm{ext}}$ [T]")
        axes[1].set_ylabel(r"$I(q,B)$ [cm$^{-1}$]")
        axes[1].set_title(
            f"Azimuthal SANS field scan · q = {probe_q:.3f} nm⁻¹", fontsize=10
        )
        axes[1].ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
        for axis in axes:
            _style_axes(axis)
        figure.tight_layout()
        return figure

    @output
    @render.ui
    def state_table() -> ui.Tag:
        result = _result()
        centre = result.field_t.size // 2
        rows = (
            ("−Bmax", 0),
            ("Zero field", centre),
            ("+Bmax", result.field_t.size - 1),
        )
        return ui.tags.table(
            ui.tags.thead(
                ui.tags.tr(
                    ui.tags.th("State"),
                    ui.tags.th("B [T]"),
                    ui.tags.th("ξ"),
                    ui.tags.th("m_z"),
                    ui.tags.th("Γxx"),
                    ui.tags.th("Γzz"),
                )
            ),
            ui.tags.tbody(
                *[
                    ui.tags.tr(
                        ui.tags.td(name),
                        ui.tags.td(f"{result.field_t[index]:.4f}"),
                        ui.tags.td(f"{result.xi[index]:.3f}"),
                        ui.tags.td(f"{result.reduced_magnetization[index]:.4f}"),
                        ui.tags.td(f"{result.gamma[index, 0, 0]:.4f}"),
                        ui.tags.td(f"{result.gamma[index, 2, 2]:.4f}"),
                    )
                    for name, index in rows
                ]
            ),
            class_="state-table",
        )

    @render.download_button(
        filename="langevinsans_field_curve.csv",
        media_type="text/csv",
    )
    def download_csv():
        result = _result()
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "field_T",
                "xi",
                "mean_mx",
                "mean_my",
                "mean_mz",
                "magnetization_A_per_m",
                "Gamma_xx",
                "Gamma_xy",
                "Gamma_xz",
                "Gamma_yy",
                "Gamma_yz",
                "Gamma_zz",
                "A_perpendicular",
            ]
        )
        for index, field in enumerate(result.field_t):
            gamma = result.gamma[index]
            writer.writerow(
                [
                    field,
                    result.xi[index],
                    *result.mean_m[index],
                    result.magnetization_a_per_m[index],
                    gamma[0, 0],
                    gamma[0, 1],
                    gamma[0, 2],
                    gamma[1, 1],
                    gamma[1, 2],
                    gamma[2, 2],
                    result.anisotropy_parameter_perpendicular[index],
                ]
            )
        yield buffer.getvalue()


app = App(app_ui, server, static_assets=APP_DIR / "www")
