from flask import Flask, request, render_template, send_file, jsonify
from PIL import Image, ImageOps
from io import BytesIO
from dotenv import load_dotenv
import os
import cv2
import numpy as np

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload limit
load_dotenv()


def validate_env():
    return []


@app.route("/")
def index():
    missing = validate_env()
    return render_template("index.html", missing=missing)


def process_single_image(input_image_bytes):
    from rembg import remove

    if not input_image_bytes:
        raise ValueError("empty_image")

    # Resize large images first to reduce memory use on Render
    original = Image.open(BytesIO(input_image_bytes)).convert("RGB")

    if original.width > 1000 or original.height > 1000:
        original.thumbnail((1000, 1000))

    buffer = BytesIO()
    original.save(buffer, format="PNG")
    buffer.seek(0)

    try:
        removed_bytes = remove(buffer.getvalue())
        img = Image.open(BytesIO(removed_bytes)).convert("RGBA")

        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        final_img = background

    except Exception:
        # fallback if rembg fails
        final_img = original

    return final_img


def detect_face_pil(pil_img):
    img_np = np.array(pil_img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )

    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.03,
            minNeighbors=3,
            minSize=(30, 30),
        )

    if len(faces) == 0:
        profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
        profile_cascade = cv2.CascadeClassifier(profile_path)
        faces = profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

    if len(faces) == 0:
        return None

    faces = sorted(faces, key=lambda f: f[1])
    top_face = faces[0]

    _, face_y, _, _ = top_face
    if face_y > img_np.shape[0] * 0.55:
        return None

    return top_face


def auto_crop_passport(pil_img, out_w, out_h, crop_mode="auto"):
    img = pil_img.convert("RGB")
    img_w, img_h = img.size
    target_ratio = out_w / out_h

    face = detect_face_pil(img)

    if face is not None:
        _, fy, _, _ = face
        if fy > img_h * 0.5:
            face = None

    if face is not None and crop_mode in ("auto", "face"):
        x, y, w, h = face

        y1 = max(0, int(y - h * 0.45))
        y2 = min(img_h, int(y + h * 1.75))

        if crop_mode == "face":
            y2 = min(y2, int(y + h * 1.8))

        min_height = int(h * 1.6)
        if y2 - y1 < min_height:
            y2 = min(img_h, y1 + min_height)

        face_cx = x + w / 2
        desired_w = min(int(w * 2.2), img_w)

        crop_x1 = max(0, int(face_cx - desired_w / 2))
        crop_x2 = min(img_w, int(face_cx + desired_w / 2))

        if crop_x2 - crop_x1 < desired_w:
            if crop_x1 == 0:
                crop_x2 = min(img_w, crop_x1 + desired_w)
            elif crop_x2 == img_w:
                crop_x1 = max(0, img_w - desired_w)
            else:
                crop_x1 = max(0, int(face_cx - desired_w / 2))
                crop_x2 = min(img_w, int(face_cx + desired_w / 2))

        cropped = img.crop((crop_x1, y1, crop_x2, y2))

    else:
        y2 = int(img_h * (0.58 if crop_mode == "center" else 0.62))
        y2 = min(img_h, y2)
        cropped = img.crop((0, 0, img_w, y2))

    crop_w, crop_h = cropped.size
    current_ratio = crop_w / crop_h

    if current_ratio > target_ratio:
        new_w = int(target_ratio * crop_h)
        left = max(0, (crop_w - new_w) // 2)
        cropped = cropped.crop((left, 0, left + new_w, crop_h))
    else:
        new_h = int(crop_w / target_ratio)
        top = max(0, (crop_h - new_h) // 2)
        cropped = cropped.crop((0, top, crop_w, top + new_h))

    if cropped.mode != "RGB":
        cropped = cropped.convert("RGB")

    white_bg = Image.new("RGB", cropped.size, (255, 255, 255))
    white_bg.paste(cropped)

    return white_bg.resize((out_w, out_h), Image.LANCZOS)


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (255, 255, 255)
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


@app.route("/process", methods=["POST"])
def process():
    try:
        passport_width = int(request.form.get("width", 413))
        passport_height = int(request.form.get("height", 531))
        border = int(request.form.get("border", 2))
        spacing = int(request.form.get("spacing", 10))
        copies = int(request.form.get("copies", 8))
    except ValueError:
        return jsonify({"error": "Width, height, border, spacing and copies must be numbers."}), 400

    crop_mode = request.form.get("crop_mode", "auto")
    bgcolor = request.form.get("bgcolor", "#ffffff")
    bg_rgb = hex_to_rgb(bgcolor)
    add_border = request.form.get("add_border") == "on"

    margin_x = 30
    margin_y = 30
    a4_w, a4_h = 2480, 3508

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    if not file or not file.filename:
        return jsonify({"error": "No image uploaded."}), 400

    try:
        img_bytes = file.read()
        img = process_single_image(img_bytes)
        img = auto_crop_passport(img, passport_width, passport_height, crop_mode=crop_mode)

        if add_border:
            img = ImageOps.expand(img, border=border, fill="black")

    except ValueError as e:
        return jsonify({"error": "processing_failed", "details": str(e)}), 500
    except Exception as e:
        return jsonify({"error": "unexpected_error", "details": str(e)}), 500

    paste_w, paste_h = img.size
    page = Image.new("RGB", (a4_w, a4_h), bg_rgb)
    x, y = margin_x, margin_y

    for _ in range(copies):
        if x + paste_w > a4_w - margin_x:
            x = margin_x
            y += paste_h + spacing
        if y + paste_h > a4_h - margin_y:
            break
        page.paste(img, (x, y))
        x += paste_w + spacing

    output = BytesIO()
    page.save(output, format="PDF", dpi=(300, 300))
    output.seek(0)

    return send_file(
        output,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="passport-sheet.pdf",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)