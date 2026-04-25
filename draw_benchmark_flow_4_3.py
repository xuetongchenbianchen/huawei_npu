#!/usr/bin/env python3
from PIL import Image, ImageDraw, ImageFont


W, H = 1200, 900
LATIN_FONT_PATH = "/usr/share/fonts/google-droid-fonts/DroidSans.ttf"
CJK_FONT_PATH = "/usr/share/fonts/google-droid-fonts/DroidSansFallback.ttf"


def latin_font(size):
    return ImageFont.truetype(LATIN_FONT_PATH, size=size)


def cjk_font(size):
    return ImageFont.truetype(CJK_FONT_PATH, size=size)


TEXT_LATIN = latin_font(22)
TEXT_CJK = cjk_font(22)
SMALL_LATIN = latin_font(18)
SMALL_CJK = cjk_font(18)

BOX_FILL = (247, 250, 252)
BOX_STROKE = (51, 65, 85)
EDGE = (71, 85, 105)
DECISION_FILL = (255, 247, 237)
DECISION_STROKE = (194, 65, 12)
WHITE = (255, 255, 255)


def pick_font(ch, latin, cjk):
    return latin if ord(ch) < 128 else cjk


def line_size(draw, line, latin, cjk):
    width = 0
    height = 0
    for ch in line:
        fnt = pick_font(ch, latin, cjk)
        bbox = draw.textbbox((0, 0), ch, font=fnt)
        width += bbox[2] - bbox[0]
        height = max(height, bbox[3] - bbox[1])
    return width, height


def draw_mixed_line(draw, xy, line, latin, cjk, fill):
    x, y = xy
    for ch in line:
        fnt = pick_font(ch, latin, cjk)
        draw.text((x, y), ch, font=fnt, fill=fill)
        bbox = draw.textbbox((0, 0), ch, font=fnt)
        x += bbox[2] - bbox[0]


def text_size(draw, text, latin, cjk):
    lines = text.split("\n")
    widths = []
    heights = []
    for line in lines:
        width, height = line_size(draw, line, latin, cjk)
        widths.append(width)
        heights.append(height)
    return max(widths), sum(heights) + (len(lines) - 1) * 6


def centered_text(draw, box, text, latin=TEXT_LATIN, cjk=TEXT_CJK, fill=(0, 0, 0)):
    x1, y1, x2, y2 = box
    tw, th = text_size(draw, text, latin, cjk)
    y = y1 + ((y2 - y1) - th) / 2
    for line in text.split("\n"):
        lw, lh = line_size(draw, line, latin, cjk)
        x = x1 + ((x2 - x1) - lw) / 2
        draw_mixed_line(draw, (x, y), line, latin, cjk, fill)
        y += lh + 6


def rounded_box(draw, box, text, fill=BOX_FILL):
    draw.rounded_rectangle(box, radius=18, fill=fill, outline=BOX_STROKE, width=3)
    centered_text(draw, box, text)


def diamond(draw, box, text):
    x1, y1, x2, y2 = box
    pts = [((x1 + x2) / 2, y1), (x2, (y1 + y2) / 2), ((x1 + x2) / 2, y2), (x1, (y1 + y2) / 2)]
    draw.polygon(pts, fill=DECISION_FILL, outline=DECISION_STROKE)
    draw.line(pts + [pts[0]], fill=DECISION_STROKE, width=3)
    centered_text(draw, box, text)


def arrow(draw, start, end, label=None, label_pos=None):
    draw.line([start, end], fill=EDGE, width=3)
    sx, sy = start
    ex, ey = end
    if abs(ex - sx) >= abs(ey - sy):
        sign = 1 if ex >= sx else -1
        head = [(ex, ey), (ex - sign * 13, ey - 7), (ex - sign * 13, ey + 7)]
    else:
        sign = 1 if ey >= sy else -1
        head = [(ex, ey), (ex - 7, ey - sign * 13), (ex + 7, ey - sign * 13)]
    draw.polygon(head, fill=EDGE)
    if label:
        lx, ly = label_pos or ((sx + ex) / 2, (sy + ey) / 2)
        tw, th = line_size(draw, label, SMALL_LATIN, SMALL_CJK)
        draw_mixed_line(draw, (lx - tw / 2, ly - th / 2), label, SMALL_LATIN, SMALL_CJK, (0, 0, 0))


def poly_arrow(draw, points, label=None, label_pos=None):
    for a, b in zip(points, points[1:]):
        draw.line([a, b], fill=EDGE, width=3)
    arrow(draw, points[-2], points[-1], label=label, label_pos=label_pos)


def main():
    img = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(img)

    boxes = {
        "start": (55, 70, 315, 145),
        "gen": (375, 70, 655, 145),
        "materialize": (715, 70, 1065, 145),
        "warmup": (145, 270, 405, 345),
        "repeat": (470, 270, 730, 345),
        "sync": (795, 270, 1055, 345),
        "stats": (420, 455, 780, 535),
        "precision": (430, 620, 770, 725),
        "check": (120, 735, 440, 810),
        "skip": (790, 735, 1060, 810),
        "record": (465, 735, 755, 810),
        "json": (465, 825, 755, 890),
    }

    rounded_box(draw, boxes["start"], "遍历每个算子 op\n和每个 case")
    rounded_box(draw, boxes["gen"], "CPU 后端生成共享输入\ngenerate_input()")
    rounded_box(draw, boxes["materialize"], "CPU / NPU 输入物化\nmaterialize_input()")
    rounded_box(draw, boxes["warmup"], "执行 warmup\n预热不计入统计")
    rounded_box(draw, boxes["repeat"], "正式计时 repeat 次\n记录每次运行耗时")
    rounded_box(draw, boxes["sync"], "计时前后同步\nmaybe_sync()")
    rounded_box(draw, boxes["stats"], "生成性能统计\nmean / median / min / max / stdev")
    diamond(draw, boxes["precision"], "是否开启\n--check-precision?")
    rounded_box(draw, boxes["check"], "重新计算 CPU / NPU 输出\nvalidate_outputs()")
    rounded_box(draw, boxes["skip"], "跳过精度校验")
    rounded_box(draw, boxes["record"], "记录当前 case 结果\n耗时 / 工作量 / 参数 / 精度")
    rounded_box(draw, boxes["json"], "汇总所有结果\n可选写入 JSON")

    arrow(draw, (315, 107), (375, 107))
    arrow(draw, (655, 107), (715, 107))
    poly_arrow(draw, [(890, 145), (890, 205), (275, 205), (275, 270)])
    arrow(draw, (405, 307), (470, 307))
    arrow(draw, (730, 307), (795, 307))
    poly_arrow(draw, [(925, 345), (925, 400), (600, 400), (600, 455)])
    arrow(draw, (600, 535), (600, 620))
    poly_arrow(draw, [(430, 672), (280, 672), (280, 735)], label="是", label_pos=(330, 646))
    poly_arrow(draw, [(770, 672), (925, 672), (925, 735)], label="否", label_pos=(872, 646))
    arrow(draw, (440, 772), (465, 772))
    arrow(draw, (790, 772), (755, 772))
    arrow(draw, (610, 810), (610, 825))

    img.save("benchmark_flow_4_3.png")


if __name__ == "__main__":
    main()
