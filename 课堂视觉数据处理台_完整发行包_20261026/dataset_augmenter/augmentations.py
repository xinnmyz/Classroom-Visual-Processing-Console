"""Region-aware synthetic occlusion and privacy transforms.

The caller must provide an annotation, detector box, or manual ROI.  These
functions never infer identity and return metadata describing every synthetic
operation so it cannot be mistaken for real occlusion.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

Box = tuple[int, int, int, int]


def clip_box(box: Sequence[float], image_size: tuple[int, int]) -> Box:
    if len(box) < 4:
        raise ValueError("box must contain x1, y1, x2, y2")
    width, height = image_size
    values = [float(value) for value in box[:4]]
    left = max(0, min(width - 1, int(round(min(values[0], values[2])))))
    top = max(0, min(height - 1, int(round(min(values[1], values[3])))))
    right = max(left + 1, min(width, int(round(max(values[0], values[2])))))
    bottom = max(top + 1, min(height, int(round(max(values[1], values[3])))))
    return left, top, right, bottom


def normalized_box(box: Sequence[float], image_size: tuple[int, int]) -> Box:
    values = [float(value) for value in box[:4]]
    width, height = image_size
    if all(0.0 <= value <= 1.0 for value in values):
        values = [values[0] * width, values[1] * height, values[2] * width, values[3] * height]
    return clip_box(values, image_size)


def box_area(box: Sequence[float]) -> int:
    return max(0, int(round(float(box[2]) - float(box[0]))) * int(round(float(box[3]) - float(box[1]))))


def _metadata(box: Box, image_size: tuple[int, int], method: str, **extra: Any) -> dict[str, Any]:
    width, height = image_size
    return {
        "augmentation": method,
        "region_box": "|".join(str(value) for value in box),
        "region_area_pixels": box_area(box),
        "region_area_fraction": box_area(box) / max(1, width * height),
        **extra,
    }


def apply_region_occlusion(
    image: Image.Image,
    box: Sequence[float],
    method: str = "black",
    color: tuple[int, int, int] = (0, 0, 0),
    blur_radius: float = 12.0,
    pixel_size: int = 12,
) -> tuple[Image.Image, dict[str, Any]]:
    """Apply black, blur, or mosaic occlusion to one provided region."""
    output = image.convert("RGB").copy()
    clipped = clip_box(box, output.size)
    left, top, right, bottom = clipped
    region = output.crop(clipped)
    canonical = str(method).strip().lower()
    if canonical in {"black", "solid", "遮挡", "黑色"}:
        output.paste(Image.new("RGB", region.size, tuple(int(value) for value in color)), (left, top))
        canonical = "black"
    elif canonical in {"blur", "模糊", "face_blur"}:
        output.paste(region.filter(ImageFilter.GaussianBlur(radius=max(0.1, float(blur_radius)))), (left, top))
        canonical = "blur"
    elif canonical in {"pixelate", "mosaic", "马赛克"}:
        scale = max(1, int(pixel_size))
        small = region.resize((max(1, region.width // scale), max(1, region.height // scale)), Image.Resampling.BILINEAR)
        output.paste(small.resize(region.size, Image.Resampling.NEAREST), (left, top))
        canonical = "pixelate"
    else:
        raise ValueError(f"unsupported region occlusion method: {method}")
    return output, _metadata(clipped, output.size, f"region_{canonical}", source_region="provided_box")


def simulate_foreground_occlusion(
    image: Image.Image,
    target_box: Sequence[float],
    foreground_box: Sequence[float],
    foreground_image: Image.Image | None = None,
    blend: float = 1.0,
) -> tuple[Image.Image, dict[str, Any]]:
    """Place a supplied foreground crop over a target person.

    Without ``foreground_image`` a neutral synthetic patch is used and the
    metadata marks it as synthetic; it is not presented as a real student.
    """
    output = image.convert("RGB").copy()
    target = clip_box(target_box, output.size)
    source_image = foreground_image.convert("RGB") if foreground_image is not None else output
    source = clip_box(foreground_box, source_image.size)
    target_region = output.crop(target)
    if foreground_image is None:
        patch = Image.new("RGB", target_region.size, (64, 64, 64))
        source_kind = "synthetic_neutral_foreground"
    else:
        patch = source_image.crop(source).resize(target_region.size, Image.Resampling.BILINEAR)
        source_kind = "provided_foreground_image"
    alpha = max(0.0, min(1.0, float(blend)))
    if alpha < 1.0:
        patch = Image.blend(target_region, patch, alpha)
    output.paste(patch, (target[0], target[1]))
    return output, _metadata(
        target,
        output.size,
        "foreground_person_occlusion",
        source_box="|".join(str(value) for value in source),
        source_kind=source_kind,
        blend=alpha,
        occluder_is_real=bool(foreground_image is not None),
    )


def _keypoint_xy(keypoint: Any, image_size: tuple[int, int]) -> tuple[float, float] | None:
    if isinstance(keypoint, Mapping):
        x, y = keypoint.get("x", keypoint.get("u")), keypoint.get("y", keypoint.get("v"))
    elif isinstance(keypoint, Sequence) and len(keypoint) >= 2:
        x, y = keypoint[0], keypoint[1]
    else:
        return None
    try:
        x, y = float(x), float(y)
    except (TypeError, ValueError):
        return None
    width, height = image_size
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        x, y = x * width, y * height
    return x, y


def apply_keypoint_occlusion(
    image: Image.Image,
    keypoints: Iterable[Any],
    radius: float = 10.0,
    fraction: float = 1.0,
    method: str = "black",
    rng: np.random.Generator | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    """Mask a reproducible subset of supplied keypoints and report counts."""
    points = [point for point in (_keypoint_xy(item, image.size) for item in keypoints) if point is not None]
    if not points:
        return image.convert("RGB").copy(), {"augmentation": "keypoint_occlusion", "occluded_keypoint_count": 0, "keypoint_count": 0, "occluded_keypoint_fraction": 0.0}
    fraction = max(0.0, min(1.0, float(fraction)))
    count = max(1, int(math.ceil(len(points) * fraction))) if fraction > 0 else 0
    indices = list(range(count)) if rng is None else rng.choice(len(points), size=count, replace=False).tolist()
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    radius = max(1.0, float(radius))
    canonical = str(method).strip().lower()
    for index in indices:
        x, y = points[index]
        if canonical in {"blur", "模糊"}:
            box = clip_box((x - radius, y - radius, x + radius, y + radius), output.size)
            output.paste(output.crop(box).filter(ImageFilter.GaussianBlur(max(1.0, radius / 2))), (box[0], box[1]))
        else:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(0, 0, 0))
    return output, {"augmentation": "keypoint_occlusion", "occluded_keypoint_count": len(indices), "keypoint_count": len(points), "occluded_keypoint_fraction": len(indices) / len(points), "keypoint_radius": radius, "keypoint_method": canonical or "black"}


def anonymize_regions(
    image: Image.Image,
    boxes: Iterable[Sequence[float]],
    method: str = "blur",
    blur_radius: float = 14.0,
    pixel_size: int = 14,
) -> tuple[Image.Image, dict[str, Any]]:
    """Apply face/person anonymization to multiple provided boxes."""
    output = image.convert("RGB").copy()
    clipped_boxes: list[Box] = []
    canonical = str(method).strip().lower()
    for raw_box in boxes:
        clipped = clip_box(raw_box, output.size)
        output, _ = apply_region_occlusion(output, clipped, method=canonical, blur_radius=blur_radius, pixel_size=pixel_size)
        clipped_boxes.append(clipped)
    return output, {
        "augmentation": "privacy_anonymization",
        "privacy_method": canonical,
        "privacy_region_count": len(clipped_boxes),
        "privacy_boxes": ";".join("|".join(str(value) for value in box) for box in clipped_boxes),
    }


def anonymize_faces(image: Image.Image, face_boxes: Iterable[Sequence[float]], method: str = "blur", **kwargs: Any) -> tuple[Image.Image, dict[str, Any]]:
    """Semantic alias for face anonymization; boxes come from a detector/annotation."""
    output, metadata = anonymize_regions(image, face_boxes, method=method, **kwargs)
    metadata["privacy_target"] = "face"
    return output, metadata


def anonymize_persons(image: Image.Image, person_boxes: Iterable[Sequence[float]], method: str = "blur", **kwargs: Any) -> tuple[Image.Image, dict[str, Any]]:
    """Semantic alias for full-person anonymization."""
    output, metadata = anonymize_regions(image, person_boxes, method=method, **kwargs)
    metadata["privacy_target"] = "person"
    return output, metadata


__all__ = [
    "Box",
    "clip_box",
    "normalized_box",
    "box_area",
    "apply_region_occlusion",
    "simulate_foreground_occlusion",
    "apply_keypoint_occlusion",
    "anonymize_regions",
    "anonymize_faces",
    "anonymize_persons",
]
