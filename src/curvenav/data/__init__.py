"""Source-agnostic CurveNav data contracts with lazy training imports."""

__all__ = [
    "PolicyLoaderBundle",
    "PreparedPolicyBatch",
    "PreparedPolicyDataset",
    "build_policy_loader",
    "build_policy_overfit_loader",
    "build_policy_training_loader",
    "build_policy_validation_loader",
    "policy_dataset_contract",
    "read_policy_manifest",
    "unpack_policy_batch",
]


def __getattr__(name: str):
    if name in {"PreparedPolicyBatch", "unpack_policy_batch"}:
        from .batch import PreparedPolicyBatch, unpack_policy_batch

        return {
            "PreparedPolicyBatch": PreparedPolicyBatch,
            "unpack_policy_batch": unpack_policy_batch,
        }[name]
    if name in {
        "PolicyLoaderBundle",
        "build_policy_loader",
        "build_policy_overfit_loader",
        "build_policy_training_loader",
        "build_policy_validation_loader",
    }:
        from .loader import (
            PolicyLoaderBundle,
            build_policy_loader,
            build_policy_overfit_loader,
            build_policy_training_loader,
            build_policy_validation_loader,
        )

        return {
            "PolicyLoaderBundle": PolicyLoaderBundle,
            "build_policy_loader": build_policy_loader,
            "build_policy_overfit_loader": build_policy_overfit_loader,
            "build_policy_training_loader": build_policy_training_loader,
            "build_policy_validation_loader": build_policy_validation_loader,
        }[name]
    if name in {
        "PreparedPolicyDataset",
        "policy_dataset_contract",
        "read_policy_manifest",
    }:
        from .prepared import (
            PreparedPolicyDataset,
            policy_dataset_contract,
            read_policy_manifest,
        )

        return {
            "PreparedPolicyDataset": PreparedPolicyDataset,
            "policy_dataset_contract": policy_dataset_contract,
            "read_policy_manifest": read_policy_manifest,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
