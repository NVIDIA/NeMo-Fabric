# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility imports for the shared NeMo Fabric codec contract."""

from nemo_fabric_evidence.codec import (
    ContractValidationError as ContractValidationError,
)
from nemo_fabric_evidence.codec import JsonValue as JsonValue
from nemo_fabric_evidence.codec import decode_dataclass as decode_dataclass
from nemo_fabric_evidence.codec import decode_field as decode_field
from nemo_fabric_evidence.codec import encode_dataclass as encode_dataclass
from nemo_fabric_evidence.codec import json_mapping as json_mapping
from nemo_fabric_evidence.codec import json_value as json_value
from nemo_fabric_evidence.codec import validate_dataclass as validate_dataclass
