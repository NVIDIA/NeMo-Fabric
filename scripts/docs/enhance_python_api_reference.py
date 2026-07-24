# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Add exact Python SDK contracts to the lazydocs-generated reference."""

from __future__ import annotations

import argparse
import importlib
import inspect
import re
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

MODULE_NAMES = (
    "nemo_fabric.client",
    "nemo_fabric.runtime",
    "nemo_fabric.models",
    "nemo_fabric.types",
    "nemo_fabric.errors",
)
CLASS_HEADING = re.compile(r"^## <kbd>class</kbd> `([^`]+)`$")
MEMBER_HEADING = re.compile(r"^### <kbd>(?:method|classmethod)</kbd> `([^`]+)`$")


def _annotation_text(annotation: Any) -> str:
    if annotation is inspect.Signature.empty:
        return ""
    if isinstance(annotation, str):
        if (
            len(annotation) >= 2
            and annotation[0] == annotation[-1]
            and annotation[0] in {'"', "'"}
        ):
            return annotation[1:-1]
        return annotation
    return inspect.formatannotation(annotation)


def _parameter_text(parameter: inspect.Parameter) -> str:
    prefix = ""
    if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
        prefix = "*"
    elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
        prefix = "**"

    rendered = prefix + parameter.name
    if parameter.annotation is not inspect.Parameter.empty:
        rendered += f": {_annotation_text(parameter.annotation)}"
    if parameter.default is not inspect.Parameter.empty:
        rendered += f" = {parameter.default!r}"
    return rendered


def _signature_tokens(
    parameters: list[inspect.Parameter],
) -> list[str]:
    tokens: list[str] = []
    positional_only = [
        index
        for index, parameter in enumerate(parameters)
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY
    ]
    last_positional_only = positional_only[-1] if positional_only else None
    has_var_positional = False
    added_keyword_separator = False

    for index, parameter in enumerate(parameters):
        if (
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            and not has_var_positional
            and not added_keyword_separator
        ):
            tokens.append("*")
            added_keyword_separator = True

        tokens.append(_parameter_text(parameter))
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            has_var_positional = True
        if index == last_positional_only:
            tokens.append("/")

    return tokens


def _render_signature(name: str, member: Any) -> str:
    signature = inspect.signature(member)
    parameters = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.name not in {"self", "cls"}
    ]
    tokens = _signature_tokens(parameters)
    prefix = "async " if inspect.iscoroutinefunction(member) else ""
    returns = ""
    if signature.return_annotation is not inspect.Signature.empty:
        returns = f" -> {_annotation_text(signature.return_annotation)}"

    compact = f"{prefix}{name}({', '.join(tokens)}){returns}"
    if len(compact) <= 88:
        return compact

    lines = [f"{prefix}{name}("]
    lines.extend(f"    {token}," for token in tokens)
    lines.append(f"){returns}")
    return "\n".join(lines)


def _replace_signature_fences(text: str, module: Any) -> str:
    lines = text.splitlines()
    class_name: str | None = None
    pending_member: str | None = None
    output: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        class_match = CLASS_HEADING.match(line)
        if class_match:
            class_name = class_match.group(1)
            pending_member = None

        member_match = MEMBER_HEADING.match(line)
        if member_match:
            pending_member = member_match.group(1)

        if line == "```python" and class_name and pending_member:
            closing = index + 1
            while closing < len(lines) and lines[closing] != "```":
                closing += 1
            if closing == len(lines):
                raise ValueError(
                    f"unterminated signature fence for {class_name}.{pending_member}"
                )

            exported_class = getattr(module, class_name)
            member = getattr(exported_class, pending_member)
            output.extend(
                [
                    "```text",
                    _render_signature(pending_member, member),
                    "```",
                ]
            )
            pending_member = None
            index = closing + 1
            continue

        output.append(line)
        index += 1

    return "\n".join(output) + "\n"


def _inline_code(value: str) -> str:
    escaped = " ".join(value.splitlines()).replace("|", r"\|")
    return f"`{escaped}`"


def _default_factory_text(factory: Any) -> str:
    name = getattr(factory, "__name__", type(factory).__name__)
    if name == "<lambda>":
        return "<generated>"
    return f"{name}()"


def _model_fields_table(model: type[BaseModel]) -> str:
    if not model.model_fields:
        return ""

    annotations = inspect.get_annotations(model, eval_str=False)
    lines = [
        "### Fields",
        "",
        "The model defines the following fields:",
        "",
        "| Field | Type | Required | Default | Constraints | Description |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, field in model.model_fields.items():
        annotation = _annotation_text(annotations.get(name, field.annotation))
        required = "Yes" if field.is_required() else "No"
        if field.is_required():
            default = "—"
        elif field.default_factory is not None:
            default = _inline_code(_default_factory_text(field.default_factory))
        else:
            default = _inline_code(repr(field.default))
        constraints = (
            _inline_code(", ".join(repr(item) for item in field.metadata))
            if field.metadata
            else "—"
        )
        description = " ".join((field.description or "—").split()).replace("|", r"\|")
        lines.append(
            f"| {_inline_code(name)} | {_inline_code(annotation)} | "
            f"{required} | {default} | {constraints} | {description} |"
        )
    return "\n".join(lines)


def _typed_fields_table(exported_class: type[Any]) -> str:
    annotations = inspect.get_annotations(exported_class, eval_str=False)
    if not annotations:
        return ""

    lines = [
        "### Fields",
        "",
        "The mapping exposes the following typed fields:",
        "",
        "| Field | Type |",
        "| --- | --- |",
    ]
    for name, annotation in annotations.items():
        lines.append(
            f"| {_inline_code(name)} | {_inline_code(_annotation_text(annotation))} |"
        )
    return "\n".join(lines)


def _inheritance_section(exported_class: type[Any]) -> str:
    bases = [base.__name__ for base in exported_class.__bases__ if base is not object]
    if not bases:
        return ""

    label = "Direct base" if len(bases) == 1 else "Direct bases"
    rendered = ", ".join(_inline_code(base) for base in bases)
    return "\n".join(
        [
            "### Inheritance",
            "",
            f"{label}: {rendered}.",
        ]
    )


def _enum_values_table(exported_class: type[Enum]) -> str:
    lines = [
        "### Values",
        "",
        "The enum defines the following values:",
        "",
        "| Name | Value |",
        "| --- | --- |",
    ]
    for member in exported_class:
        lines.append(
            f"| {_inline_code(member.name)} | {_inline_code(str(member.value))} |"
        )
    return "\n".join(lines)


def _class_supplement(exported_class: type[Any]) -> str:
    sections: list[str] = []
    if exported_class.__module__ == "nemo_fabric.errors" or issubclass(
        exported_class, Enum
    ):
        sections.append(_inheritance_section(exported_class))
    if issubclass(exported_class, BaseModel):
        sections.append(_model_fields_table(exported_class))
    elif exported_class.__module__ == "nemo_fabric.types":
        sections.append(_typed_fields_table(exported_class))
    if issubclass(exported_class, Enum):
        sections.append(_enum_values_table(exported_class))
    return "\n\n".join(section for section in sections if section)


def _insert_class_supplements(text: str, module: Any) -> str:
    lines = text.splitlines()
    headings: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        match = CLASS_HEADING.match(line)
        if match:
            headings.append((index, match.group(1)))

    for heading_index, (start, class_name) in reversed(list(enumerate(headings))):
        end = (
            headings[heading_index + 1][0]
            if heading_index + 1 < len(headings)
            else len(lines)
        )
        first_member = next(
            (
                index
                for index in range(start + 1, end)
                if lines[index].startswith("### <kbd>")
            ),
            end,
        )
        insert_at = first_member
        separators = [
            index for index in range(start + 1, first_member) if lines[index] == "---"
        ]
        if separators:
            insert_at = separators[-1]

        supplement = _class_supplement(getattr(module, class_name))
        if supplement:
            lines[insert_at:insert_at] = ["", *supplement.splitlines(), ""]

    return "\n".join(lines) + "\n"


def enhance_reference(output_dir: Path) -> None:
    for module_name in MODULE_NAMES:
        module = importlib.import_module(module_name)
        page = output_dir / f"{module_name}.md"
        text = page.read_text(encoding="utf-8")
        text = _replace_signature_fences(text, module)
        text = _insert_class_supplements(text, module)
        page.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    enhance_reference(args.output_dir)


if __name__ == "__main__":
    main()
