from nmoe.moe import _blockscaled_backward_profiles


def test_blockscaled_backward_profiles_match_live_profile_by_default() -> None:
    assert _blockscaled_backward_profiles("nvfp4", "off") == ("nvfp4", "nvfp4", "nvfp4", "nvfp4")
    assert _blockscaled_backward_profiles("fp8", "off") == ("fp8", "fp8", "fp8", "fp8")


def test_blockscaled_backward_profiles_follow_explicit_forward_ablation() -> None:
    assert _blockscaled_backward_profiles("nvfp4", "stage1_fp8") == ("fp8", "fp8", "fp8", "nvfp4")
    assert _blockscaled_backward_profiles("nvfp4", "w2_fp8") == ("nvfp4", "nvfp4", "nvfp4", "fp8")
