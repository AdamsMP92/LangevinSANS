import runpy
from pathlib import Path

from langevinsans import FieldGrid, ParticleProperties


def _app_namespace() -> dict[str, object]:
    app_path = Path(__file__).parents[1] / "src" / "app.py"
    return runpy.run_path(str(app_path))


def test_shiny_app_constructs() -> None:
    namespace = _app_namespace()
    assert namespace["app"] is not None


def test_model_dispatcher_returns_analytical_result() -> None:
    namespace = _app_namespace()
    evaluate = namespace["_evaluate_model"]
    result = evaluate("analytical", ParticleProperties(), FieldGrid(n_field_points=5))
    assert result.field_t.size == 5
