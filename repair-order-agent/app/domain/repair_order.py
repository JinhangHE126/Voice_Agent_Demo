"""Compatibility shim: prefer `repair_order` package for new imports."""

from repair_order.domain.repair_order import OrderState, RepairOrderDraft

__all__ = ["OrderState", "RepairOrderDraft"]
