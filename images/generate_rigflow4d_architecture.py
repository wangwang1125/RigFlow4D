from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import textwrap
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


WIDTH = 2600
HEIGHT = 1650
OUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUT_DIR / "rigflow4d_network_architecture.png"
SVG_PATH = OUT_DIR / "rigflow4d_network_architecture.svg"


PALETTE = {
    "ink": "#1f2937",
    "muted": "#667085",
    "grid": "#d0d5dd",
    "paper": "#fbfcfe",
    "blue_fill": "#eaf2ff",
    "blue": "#2563eb",
    "cyan_fill": "#e6f8fb",
    "cyan": "#0891b2",
    "green_fill": "#eaf7ec",
    "green": "#15803d",
    "orange_fill": "#fff3e6",
    "orange": "#ea7600",
    "violet_fill": "#f0edff",
    "violet": "#6d5bd0",
    "rose_fill": "#fff0f3",
    "rose": "#d92d5b",
    "gray_fill": "#f2f4f7",
    "gray": "#475467",
}


@dataclass
class Box:
    key: str
    x: int
    y: int
    w: int
    h: int
    title: str
    lines: list[str] = field(default_factory=list)
    fill: str = PALETTE["gray_fill"]
    stroke: str = PALETTE["gray"]
    dashed: bool = False

    @property
    def left(self) -> tuple[int, int]:
        return self.x, self.y + self.h // 2

    @property
    def right(self) -> tuple[int, int]:
        return self.x + self.w, self.y + self.h // 2

    @property
    def top(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y

    @property
    def bottom(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h


@dataclass
class Arrow:
    start: tuple[int, int]
    end: tuple[int, int]
    color: str = PALETTE["ink"]
    dashed: bool = False
    label: str | None = None
    label_offset: tuple[int, int] = (0, -12)


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def wrap_lines(lines: Iterable[str], width: int) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=width) or [""])
    return wrapped


def draw_dashed_line(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill: str, width: int) -> None:
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dash = 14
    gap = 9
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    pos = 0.0
    while pos < length:
        end_pos = min(pos + dash, length)
        draw.line(
            (x1 + dx * pos, y1 + dy * pos, x1 + dx * end_pos, y1 + dy * end_pos),
            fill=hex_to_rgb(fill),
            width=width,
        )
        pos += dash + gap


def draw_arrowhead(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str) -> None:
    x1, y1 = start
    x2, y2 = end
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 16
    points = [
        (x2, y2),
        (x2 - size * math.cos(angle - 0.42), y2 - size * math.sin(angle - 0.42)),
        (x2 - size * math.cos(angle + 0.42), y2 - size * math.sin(angle + 0.42)),
    ]
    draw.polygon(points, fill=hex_to_rgb(color))


def draw_png(boxes: list[Box], arrows: list[Arrow]) -> None:
    img = Image.new("RGB", (WIDTH, HEIGHT), hex_to_rgb(PALETTE["paper"]))
    draw = ImageDraw.Draw(img)
    title_font = load_font(48, bold=True)
    section_font = load_font(28, bold=True)
    box_title_font = load_font(25, bold=True)
    body_font = load_font(20)
    small_font = load_font(17)

    draw.text((70, 42), "RigFlow4D Network Architecture", fill=hex_to_rgb(PALETTE["ink"]), font=title_font)
    draw.text(
        (72, 101),
        "Camera-optional visual encoding + topology-conditioned kinematic latent flow for arbitrary rig motion",
        fill=hex_to_rgb(PALETTE["muted"]),
        font=body_font,
    )

    panels = [
        (45, 145, 2510, 560, "A. Visual-to-Rig Motion Pipeline"),
        (45, 745, 2510, 790, "B. Topology-Conditioned Stage 1 TG-VAE and Latent Flow Refinement"),
    ]
    for x, y, w, h, label in panels:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=18, outline=hex_to_rgb("#c7d7ee"), width=3, fill=hex_to_rgb("#ffffff"))
        draw.text((x + 26, y + 18), label, fill=hex_to_rgb(PALETTE["ink"]), font=section_font)

    for arrow in arrows:
        if arrow.dashed:
            draw_dashed_line(draw, arrow.start, arrow.end, arrow.color, width=4)
        else:
            draw.line((*arrow.start, *arrow.end), fill=hex_to_rgb(arrow.color), width=4)
        draw_arrowhead(draw, arrow.start, arrow.end, arrow.color)
        if arrow.label:
            lx = (arrow.start[0] + arrow.end[0]) // 2 + arrow.label_offset[0]
            ly = (arrow.start[1] + arrow.end[1]) // 2 + arrow.label_offset[1]
            draw.text((lx, ly), arrow.label, fill=hex_to_rgb(arrow.color), font=small_font)

    for box in boxes:
        outline = hex_to_rgb(box.stroke)
        fill = hex_to_rgb(box.fill)
        if box.dashed:
            draw.rounded_rectangle((box.x, box.y, box.x + box.w, box.y + box.h), radius=12, fill=fill, outline=outline, width=1)
            for xx in range(box.x, box.x + box.w, 22):
                draw.line((xx, box.y, min(xx + 12, box.x + box.w), box.y), fill=outline, width=3)
                draw.line((xx, box.y + box.h, min(xx + 12, box.x + box.w), box.y + box.h), fill=outline, width=3)
            for yy in range(box.y, box.y + box.h, 22):
                draw.line((box.x, yy, box.x, min(yy + 12, box.y + box.h)), fill=outline, width=3)
                draw.line((box.x + box.w, yy, box.x + box.w, min(yy + 12, box.y + box.h)), fill=outline, width=3)
        else:
            draw.rounded_rectangle((box.x, box.y, box.x + box.w, box.y + box.h), radius=12, fill=fill, outline=outline, width=3)
        draw.text((box.x + 18, box.y + 15), box.title, fill=outline, font=box_title_font)
        y = box.y + 50
        max_chars = max(20, box.w // 12)
        for line in wrap_lines(box.lines, max_chars):
            draw.text((box.x + 18, y), line, fill=hex_to_rgb(PALETTE["ink"]), font=body_font)
            y += 27

    legend_y = HEIGHT - 82
    legend = [
        ("solid arrow", PALETTE["ink"], "inference data flow"),
        ("dashed arrow", PALETTE["orange"], "training supervision / optional branch"),
        ("blue", PALETTE["blue"], "visual stream"),
        ("green", PALETTE["green"], "rig topology stream"),
        ("violet", PALETTE["violet"], "latent motion stream"),
    ]
    x = 70
    for name, color, text in legend:
        draw.rounded_rectangle((x, legend_y, x + 28, legend_y + 18), radius=4, fill=hex_to_rgb(color))
        draw.text((x + 36, legend_y - 3), f"{name}: {text}", fill=hex_to_rgb(PALETTE["muted"]), font=small_font)
        x += 470 if name == "solid arrow" else 410

    img.save(PNG_PATH, dpi=(300, 300))


def svg_text(x: int, y: int, text: str, size: int, weight: str = "400", fill: str = PALETTE["ink"]) -> str:
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}">{safe}</text>'


def svg_box(box: Box) -> str:
    dash = ' stroke-dasharray="14 9"' if box.dashed else ""
    lines = [
        f'<rect x="{box.x}" y="{box.y}" width="{box.w}" height="{box.h}" rx="12" fill="{box.fill}" stroke="{box.stroke}" stroke-width="3"{dash}/>',
        svg_text(box.x + 18, box.y + 34, box.title, 25, "700", box.stroke),
    ]
    y = box.y + 64
    for line in wrap_lines(box.lines, max(20, box.w // 12)):
        lines.append(svg_text(box.x + 18, y, line, 20))
        y += 27
    return "\n".join(lines)


def svg_arrow(arrow: Arrow, idx: int) -> str:
    dash = ' stroke-dasharray="14 9"' if arrow.dashed else ""
    label = ""
    if arrow.label:
        lx = (arrow.start[0] + arrow.end[0]) // 2 + arrow.label_offset[0]
        ly = (arrow.start[1] + arrow.end[1]) // 2 + arrow.label_offset[1]
        label = svg_text(lx, ly, arrow.label, 17, "600", arrow.color)
    return (
        f'<line x1="{arrow.start[0]}" y1="{arrow.start[1]}" x2="{arrow.end[0]}" y2="{arrow.end[1]}" '
        f'stroke="{arrow.color}" stroke-width="4" marker-end="url(#arrow{idx})"{dash}/>\n{label}'
    )


def draw_svg(boxes: list[Box], arrows: list[Arrow]) -> None:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<defs>",
    ]
    used_colors = sorted({arrow.color for arrow in arrows})
    for idx, color in enumerate(used_colors):
        parts.append(
            f'<marker id="arrow{idx}" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">'
            f'<path d="M0,0 L0,6 L9,3 z" fill="{color}"/></marker>'
        )
    color_to_marker = {color: idx for idx, color in enumerate(used_colors)}
    parts.extend(["</defs>", f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{PALETTE["paper"]}"/>'])
    parts.append(svg_text(70, 82, "RigFlow4D Network Architecture", 48, "700"))
    parts.append(
        svg_text(
            72,
            125,
            "Camera-optional visual encoding + topology-conditioned kinematic latent flow for arbitrary rig motion",
            20,
            "400",
            PALETTE["muted"],
        )
    )
    for x, y, w, h, label in [
        (45, 145, 2510, 560, "A. Visual-to-Rig Motion Pipeline"),
        (45, 745, 2510, 790, "B. Topology-Conditioned Stage 1 TG-VAE and Latent Flow Refinement"),
    ]:
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="#ffffff" stroke="#c7d7ee" stroke-width="3"/>')
        parts.append(svg_text(x + 26, y + 50, label, 28, "700"))
    for arrow in arrows:
        parts.append(svg_arrow(arrow, color_to_marker[arrow.color]))
    for box in boxes:
        parts.append(svg_box(box))
    legend_y = HEIGHT - 82
    x = 70
    for name, color, text in [
        ("solid arrow", PALETTE["ink"], "inference data flow"),
        ("dashed arrow", PALETTE["orange"], "training supervision / optional branch"),
        ("blue", PALETTE["blue"], "visual stream"),
        ("green", PALETTE["green"], "rig topology stream"),
        ("violet", PALETTE["violet"], "latent motion stream"),
    ]:
        parts.append(f'<rect x="{x}" y="{legend_y}" width="28" height="18" rx="4" fill="{color}"/>')
        parts.append(svg_text(x + 36, legend_y + 14, f"{name}: {text}", 17, "400", PALETTE["muted"]))
        x += 470 if name == "solid arrow" else 410
    parts.append("</svg>")
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


def build_diagram() -> tuple[list[Box], list[Arrow]]:
    boxes = [
        Box("visual_in", 80, 220, 300, 135, "Image / Video Input", ["single or multi-view", "B x V x T x H x W x 3"], PALETTE["blue_fill"], PALETTE["blue"]),
        Box("cam", 80, 385, 300, 125, "Camera Hint", ["optional K, R, t", "calibrated / weak / unknown"], PALETTE["cyan_fill"], PALETTE["cyan"], dashed=True),
        Box("rig", 80, 530, 300, 145, "Target Rig Graph", ["J joints, parents E", "rest offsets, masks", "chain coordinates"], PALETTE["green_fill"], PALETTE["green"]),
        Box("motion_label", 80, 800, 300, 130, "Motion Labels", ["positions, root traj", "local rot6D", "training only"], PALETTE["orange_fill"], PALETTE["orange"], dashed=True),
        Box("dinov3", 455, 225, 315, 125, "DINOv3 Dense Encoder", ["frozen / LoRA optional", "patch + global tokens"], PALETTE["blue_fill"], PALETTE["blue"]),
        Box("vt", 835, 220, 355, 135, "View-Time Relation Transformer", ["camera-optional fusion", "token dropout + masks", "Z_v: B x T x P x d"], PALETTE["blue_fill"], PALETTE["blue"]),
        Box("topo", 455, 515, 355, 150, "Topology Encoder", ["topology token per joint", "offset, length, depth", "root flag, chain coordinate"], PALETTE["green_fill"], PALETTE["green"]),
        Box("query", 875, 510, 320, 155, "Rig Query Builder", ["Q_r: B x J x d", "joint mask aware", "no fixed joint index needed"], PALETTE["green_fill"], PALETTE["green"]),
        Box("peer", 1275, 320, 360, 170, "Skeleton-Peer Decoder", ["visual tokens x rig queries", "cross-view/time attention", "direct rig-native pose seed"], PALETTE["cyan_fill"], PALETTE["cyan"]),
        Box("p0", 1710, 320, 300, 170, "Initial Motion Seed P0", ["root traj", "root-relative joints", "local rot6D, contact"], PALETTE["violet_fill"], PALETTE["violet"]),
        Box("flow", 1710, 535, 340, 140, "Latent Flow Refinement", ["conditioned flow matching", "z_t, t, c -> velocity", "multi-hypothesis Delta P"], PALETTE["violet_fill"], PALETTE["violet"]),
        Box("out", 2165, 370, 330, 175, "Rig-Native 4D Output", ["absolute joints P*", "root trajectory", "local rotations", "contact / uncertainty"], PALETTE["rose_fill"], PALETTE["rose"]),
        Box("split", 445, 850, 305, 150, "Motion Factorization", ["P_rel = P - root", "root delta tokens", "local rot6D"], PALETTE["orange_fill"], PALETTE["orange"]),
        Box("topo2", 80, 1040, 300, 145, "Topology Tokens", ["parents + rest offsets", "bone length + depth", "chain coordinate"], PALETTE["green_fill"], PALETTE["green"]),
        Box("enc", 830, 840, 360, 210, "TG-VAE Encoder x L", ["input projection", "time embedding", "topology embedding", "Temporal SA -> Graph Mixer -> Spatial SA"], PALETTE["violet_fill"], PALETTE["violet"]),
        Box("pool", 1260, 870, 265, 155, "Latent Posterior", ["masked pool", "mu / logvar", "sample z or use mu"], PALETTE["violet_fill"], PALETTE["violet"]),
        Box("dec", 1595, 840, 380, 210, "Topology-Conditioned Decoder x L", ["z + time queries", "topology embedding", "Temporal SA -> Graph Mixer -> Spatial SA"], PALETTE["violet_fill"], PALETTE["violet"]),
        Box("compose", 2040, 850, 300, 195, "Motion Composition", ["predict P_rel_hat", "root_hat from root delta", "rot6D_hat", "P_hat = P_rel_hat + root_hat"], PALETTE["violet_fill"], PALETTE["violet"]),
        Box("vae_out", 2380, 870, 145, 175, "Stage 1 Output", ["P_rel", "root", "P_abs", "rot6D"], PALETTE["rose_fill"], PALETTE["rose"]),
        Box("cond", 640, 1220, 340, 145, "Condition Encoder", ["visual tokens", "rig topology", "pose seed P0"], PALETTE["cyan_fill"], PALETTE["cyan"]),
        Box("lfm", 1065, 1215, 360, 160, "Latent Flow Matcher", ["sample z_t between noise and z", "condition c, time t", "learn velocity field v_theta"], PALETTE["violet_fill"], PALETTE["violet"]),
        Box("refined", 1515, 1220, 340, 145, "Refined Latent / Motion", ["z* -> VAE decoder", "Delta P refinement", "motion hypotheses"], PALETTE["rose_fill"], PALETTE["rose"]),
        Box("losses", 420, 1420, 1735, 85, "Training Objectives", ["root-relative reconstruction | root position / velocity | rot6D | temporal velocity / acceleration | bone length | flow matching"], PALETTE["orange_fill"], PALETTE["orange"], dashed=True),
    ]
    b = {box.key: box for box in boxes}
    arrows = [
        Arrow(b["visual_in"].right, b["dinov3"].left, PALETTE["blue"]),
        Arrow(b["dinov3"].right, b["vt"].left, PALETTE["blue"]),
        Arrow(b["cam"].right, (835, 288), PALETTE["cyan"], dashed=True, label="optional camera relation"),
        Arrow(b["rig"].right, b["topo"].left, PALETTE["green"]),
        Arrow(b["topo"].right, b["query"].left, PALETTE["green"]),
        Arrow(b["vt"].right, b["peer"].left, PALETTE["blue"], label="fused visual tokens"),
        Arrow(b["query"].right, (1275, 405), PALETTE["green"], label="rig queries"),
        Arrow(b["peer"].right, b["p0"].left, PALETTE["cyan"]),
        Arrow(b["p0"].right, b["out"].left, PALETTE["rose"]),
        Arrow(b["p0"].bottom, b["flow"].top, PALETTE["violet"]),
        Arrow(b["flow"].right, (2165, 488), PALETTE["violet"]),
        Arrow(b["motion_label"].right, b["split"].left, PALETTE["orange"], dashed=True),
        Arrow(b["split"].right, b["enc"].left, PALETTE["orange"]),
        Arrow(b["topo2"].right, (550, 970), PALETTE["green"]),
        Arrow(b["enc"].right, b["pool"].left, PALETTE["violet"]),
        Arrow(b["pool"].right, b["dec"].left, PALETTE["violet"]),
        Arrow(b["topo2"].right, (1595, 980), PALETTE["green"], dashed=True),
        Arrow(b["dec"].right, b["compose"].left, PALETTE["violet"]),
        Arrow(b["compose"].right, b["vae_out"].left, PALETTE["rose"]),
        Arrow(b["cond"].right, b["lfm"].left, PALETTE["cyan"]),
        Arrow(b["pool"].bottom, (1245, 1215), PALETTE["violet"], dashed=True, label="target latent z"),
        Arrow(b["lfm"].right, b["refined"].left, PALETTE["violet"]),
        Arrow(b["refined"].top, (2190, 1045), PALETTE["rose"], dashed=True),
        Arrow((900, 1420), (1010, 1050), PALETTE["orange"], dashed=True),
        Arrow((1380, 1420), (1730, 1050), PALETTE["orange"], dashed=True),
        Arrow((1830, 1420), (2190, 1045), PALETTE["orange"], dashed=True),
    ]
    return boxes, arrows


def main() -> None:
    boxes, arrows = build_diagram()
    draw_png(boxes, arrows)
    draw_svg(boxes, arrows)
    print(PNG_PATH)
    print(SVG_PATH)


if __name__ == "__main__":
    main()
