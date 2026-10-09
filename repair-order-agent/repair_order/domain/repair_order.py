from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OrderState(str, Enum):
    COLLECTING = "collecting"
    CONFIRMING_REPAIR = "confirming_repair"
    CONFIRMING_ADDRESS = "confirming_address"
    CONFIRMING_ORDER = "confirming"
    # Backward-compatible alias for callers/tests that used the old name.
    CONFIRMING = "confirming"
    SAVING = "saving"
    COMPLETED = "completed"


@dataclass
class RepairOrderDraft:
    customer_name: str | None = None
    customer_name_needs_title: bool = False
    customer_phone: str | None = None
    customer_address: str | None = None
    repair_description: str | None = None
    repair_confirmed: bool = False
    address_confirmed: bool = False
    confirmed: bool = False
    state: OrderState = OrderState.COLLECTING
    awaiting_field: str | None = None
    session_language: str | None = None  # yue | cmn | en
    language_source: str | None = None  # explicit | detected
    language_confidence: float = 0.0
    language_asked: bool = False

    def missing_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.customer_name or self.customer_name_needs_title:
            missing.append("customer_name")
        if not self.customer_phone:
            missing.append("customer_phone")
        if not self.customer_address:
            missing.append("customer_address")
        if not self.repair_description:
            missing.append("repair_description")
        return missing

    def is_complete(self) -> bool:
        return len(self.missing_fields()) == 0
