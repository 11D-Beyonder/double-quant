from __future__ import annotations

import argparse
import re
import textwrap
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FONT_ASCII_CANDIDATES = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Supplemental/Andale Mono.ttf",
]

FONT_CJK_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]

TABLE_CHARS = "━─┃│┏┓┗┛┳┻┣┫╋┌┐└┘├┤┬┴┼╭╮╰╯"


@dataclass(frozen=True, slots=True)
class TerminalRenderStyle:
    width: int = 2500
    max_height: int = 4200
    font_size: int = 22
    line_height: int = 42
    top: int = 104
    left: int = 58
    max_output_lines: int = 72
    command_wrap_columns: int = 128
    output_wrap_columns: int = 132
    table_clip_columns: int = 170
    background: str = "#f3f4f6"
    window_background: str = "#ffffff"
    border: str = "#cbd5e1"
    text: str = "#111827"
    command_text: str = "#166534"
    muted: str = "#6b7280"
    chrome: str = "#d1d5db"


def render_results_terminal_image(
    *,
    results_md: Path,
    output: Path,
    command: str = "",
    style: TerminalRenderStyle | None = None,
) -> None:
    text = results_md.read_text(encoding="utf-8")
    command_text = extract_command(text, command)
    output_text = extract_output(text)
    render_terminal_image(
        output,
        command=command_text,
        output=output_text,
        style=style,
    )


def render_terminal_image(
    path: Path,
    *,
    command: str,
    output: str,
    style: TerminalRenderStyle | None = None,
) -> None:
    style = style or TerminalRenderStyle()
    ascii_font = _load_font(style.font_size, FONT_ASCII_CANDIDATES)
    cjk_font = _load_font(style.font_size, FONT_CJK_CANDIDATES)
    cell_width = max(12, int(ascii_font.getlength("M")))
    display_lines = _build_display_lines(command, output, style)

    height = min(
        style.max_height,
        style.top + style.line_height * len(display_lines) + 70,
    )
    image = Image.new("RGB", (style.width, height), style.background)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (24, 24, style.width - 24, height - 24),
        radius=26,
        fill=style.window_background,
        outline=style.border,
        width=2,
    )
    for i in range(3):
        draw.ellipse((58 + i * 34, 52, 78 + i * 34, 72), fill=style.chrome)

    y = style.top
    for line, color in display_lines:
        _draw_terminal_text(
            draw,
            (style.left, y),
            line,
            color,
            ascii_font=ascii_font,
            cjk_font=cjk_font,
            cell_width=cell_width,
        )
        y += style.line_height
        if y > height - 58:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def extract_command(results_text: str, fallback: str = "") -> str:
    command = _extract_first_fence_after_heading(results_text, ("运行命令", "测试命令"))
    if command:
        return command
    match = re.search(r"uv run pytest [^\n]+", results_text)
    if match:
        return match.group(0).strip()
    return fallback


def extract_output(results_text: str) -> str:
    output = _extract_first_fence_after_heading(
        results_text,
        ("运行结果", "运行输出", "程序输出", "实测结果"),
    )
    if output:
        return output
    lines: list[str] = []
    for line in results_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("```"):
            continue
        if stripped.startswith("|"):
            lines.append(stripped)
        elif any(
            token in stripped
            for token in ("通过", "输出", "生成", "结论", "覆盖", "压缩", "风险", "接口")
        ):
            lines.append(stripped)
    return "\n".join(lines[:40]).strip()


def _build_display_lines(
    command: str,
    output: str,
    style: TerminalRenderStyle,
) -> list[tuple[str, str]]:
    prompt = "double-quant % "
    display_lines: list[tuple[str, str]] = []
    command_lines = _wrap_line(prompt + command, style.command_wrap_columns)
    for index, line in enumerate(command_lines):
        prefix = "" if index == 0 else " " * len(prompt)
        display_lines.append((prefix + line, style.command_text))
    display_lines.append(("", style.text))

    output_lines = output.splitlines() if output else ["输出详见 results.md。"]
    for raw in output_lines[: style.max_output_lines]:
        if _is_terminal_table_line(raw):
            display_lines.append((raw[: style.table_clip_columns], style.text))
        else:
            for line in _wrap_line(raw, style.output_wrap_columns):
                display_lines.append((line, style.text))
    if len(output_lines) > style.max_output_lines:
        display_lines.append(("...（输出已按 results.md 摘要截断）", style.muted))
    return display_lines


def _extract_first_fence_after_heading(
    markdown: str,
    heading_names: tuple[str, ...],
) -> str:
    lines = markdown.splitlines()
    for idx, line in enumerate(lines):
        if line.strip("# ").strip() in heading_names:
            for start in range(idx + 1, len(lines)):
                if lines[start].strip().startswith("```"):
                    collected: list[str] = []
                    for end in range(start + 1, len(lines)):
                        if lines[end].strip().startswith("```"):
                            return "\n".join(collected).strip()
                        collected.append(lines[end])
                    return "\n".join(collected).strip()
            collected = []
            for start in range(idx + 1, len(lines)):
                stripped = lines[start].strip()
                if stripped.startswith("#"):
                    break
                if stripped:
                    collected.append(stripped)
            return "\n".join(collected).strip()
    return ""


def _wrap_line(line: str, width: int) -> list[str]:
    if not line:
        return [""]
    wrapped = textwrap.wrap(
        line,
        width=width,
        break_long_words=False,
        replace_whitespace=False,
        drop_whitespace=False,
    )
    return wrapped or [line]


def _is_terminal_table_line(line: str) -> bool:
    return any(char in line for char in TABLE_CHARS)


def _load_font(
    size: int,
    candidates: list[str],
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _terminal_char_columns(char: str) -> int:
    if char == "\t":
        return 4
    if unicodedata.east_asian_width(char) in {"W", "F"}:
        return 2
    return 1


def _font_for_char(
    char: str,
    *,
    ascii_font: ImageFont.ImageFont,
    cjk_font: ImageFont.ImageFont,
) -> ImageFont.ImageFont:
    if ord(char) < 128 or _is_terminal_table_line(char):
        return ascii_font
    return cjk_font


def _draw_terminal_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fill: str,
    *,
    ascii_font: ImageFont.ImageFont,
    cjk_font: ImageFont.ImageFont,
    cell_width: int,
) -> None:
    x, y = xy
    for char in text:
        if char == "\t":
            x += cell_width * _terminal_char_columns(char)
            continue
        font = _font_for_char(char, ascii_font=ascii_font, cjk_font=cjk_font)
        draw.text((x, y), char, font=font, fill=fill)
        x += cell_width * _terminal_char_columns(char)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render pytest terminal output to PNG.")
    parser.add_argument("--results-md", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--command", default="")
    args = parser.parse_args()
    render_results_terminal_image(
        results_md=args.results_md,
        output=args.output,
        command=args.command,
    )


if __name__ == "__main__":
    main()
