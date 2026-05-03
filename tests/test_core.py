from lumina.backend import BackendCapabilities, FeasibilityStatus, classify_action
from lumina.kv import KVModelSpec, KVPolicyAction, action_grid


def test_action_grid_builds_cartesian_space() -> None:
    actions = action_grid(retention_values=[0.5, 1.0], bit_widths=[8, 16])
    assert len(actions) == 4


def test_backend_blocks_quantized_retention_when_not_composable() -> None:
    backend = BackendCapabilities(
        name="test",
        quantized_kv_bits=(8,),
        supports_retention=True,
        supports_quantized_retention=False,
    )
    result = classify_action(KVPolicyAction(retention=0.5, kv_bits=8), backend)
    assert result.status == FeasibilityStatus.BACKEND_INFEASIBLE


def test_kv_memory_scales_with_retention_and_precision() -> None:
    spec = KVModelSpec(layers=16, kv_heads=8, head_dim=64)
    full = spec.full_cache_mib(1024)
    half_q8 = spec.action_cache_mib(KVPolicyAction(retention=0.5, kv_bits=8), 1024)
    assert half_q8 == full * 0.25
