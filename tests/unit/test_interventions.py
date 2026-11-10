import pytest
from precise_attribution.core.models import Intervention
from precise_attribution.interventions.policies import apply_interventions


def test_bypass_disables_only_target_module():
    active = apply_interventions(("A", "B"), {"A", "B"}, (Intervention("A", "bypass"),))
    assert active == {"B"}


def test_unsupported_semantics_fail_explicitly():
    with pytest.raises(NotImplementedError, match="null_output"):
        apply_interventions(("A",), {"A"}, (Intervention("A", "null_output"),))
