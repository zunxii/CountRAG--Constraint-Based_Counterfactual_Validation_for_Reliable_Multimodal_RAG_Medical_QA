from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT_PATH = REPO_ROOT / "configs" / "evaluation_contract.yaml"


def load_eval_contract(contract_path: Union[str, Path] = DEFAULT_CONTRACT_PATH) -> Dict[str, Any]:
    path = Path(contract_path)

    if not path.exists():
        raise FileNotFoundError(f"Evaluation contract not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        contract = yaml.safe_load(f)

    required = ["paths", "models", "retrieval", "counterfactual", "constraints", "metrics"]
    missing = [k for k in required if k not in contract]
    if missing:
        raise ValueError(f"Evaluation contract missing keys: {missing}")

    contract["_contract_path"] = str(path)
    return contract