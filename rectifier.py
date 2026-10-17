from pathlib import Path
import argparse

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_text_curve(image):
    """
    Находит примерную центральную линию изогнутого текста.

    Возвращает:
        polynomial - коэффициенты y = ax^2 + bx + c
        x_min, x_max - диапазон текста
        mask - маска найденных темных элементов
        points - найденные центры компонентов
        accepted - компоненты, использованные для кривой
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Black Hat хорошо выделяет тёмные буквы
    # на медленно меняющемся сером фоне.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (17, 17),
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel,
    )

    _, mask = cv2.threshold(
        blackhat,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    points = []

    for i in range(1, count):
        x, y, w, h, area = stats[i]

        # Отбрасываем слишком мелкий шум
        # и огромные элементы фона.
        if not (20 <= area <= 450):
            continue

        if not (3 <= w <= 35):
            continue

        if not (7 <= h <= 40):
            continue

        cx, cy = centroids[i]

        points.append(
            [
                cx,
                cy,
                area,
                w,
                h,
            ]
        )

    if len(points) < 5:
        raise RuntimeError(
            f"Не удалось найти достаточно букв. "
            f"Найдено компонентов: {len(points)}"
        )

    points = np.array(points, dtype=np.float64)

    accepted = np.ones(
        len(points),
        dtype=bool,
    )

    # Несколько проходов:
    # строим параболу и убираем точки,
    # которые слишком далеко от неё.
    for _ in range(4):

        polynomial = np.polyfit(
            points[accepted, 0],
            points[accepted, 1],
            2,
        )

        predicted_y = np.polyval(
            polynomial,
            points[:, 0],
        )

        residuals = np.abs(
            points[:, 1] - predicted_y
        )

        accepted_residuals = residuals[accepted]

        median = np.median(accepted_residuals)

        mad = np.median(
            np.abs(accepted_residuals - median)
        )

        threshold = max(
            5.0,
            median + 3.5 * 1.4826 * (mad + 1e-6),
        )

        new_accepted = residuals < threshold

        if new_accepted.sum() < 5:
            break

        if np.array_equal(
            new_accepted,
            accepted,
        ):
            break

        accepted = new_accepted

    polynomial = np.polyfit(
        points[accepted, 0],
        points[accepted, 1],
        2,
    )

    accepted_points = points[accepted]

    # Немного захватываем область слева/справа от букв.
    margin = 10

    x_min = int(
        np.min(
            accepted_points[:, 0]
            - accepted_points[:, 3] / 2
        )
    ) - margin

    x_max = int(
        np.max(
            accepted_points[:, 0]
            + accepted_points[:, 3] / 2
        )
    ) + margin

    x_min = max(0, x_min)
    x_max = min(image.shape[1] - 1, x_max)

    return (
        polynomial,
        x_min,
        x_max,
        mask,
        points,
        accepted,
    )


def rectify_image(image, output_height=64):
    """
    Выпрямляет текст вдоль найденной кривой.

    В отличие от простого вертикального сдвига колонок,
    здесь учитывается локальный угол кривой.
    """

    (
        polynomial,
        x_min,
        x_max,
        mask,
        points,
        accepted,
    ) = find_text_curve(image)

    a, b, c = polynomial

    # Частая сетка вдоль исходной кривой.
    sample_count = max(
        300,
        (x_max - x_min) * 4,
    )

    x_samples = np.linspace(
        x_min,
        x_max,
        sample_count,
    )

    y_samples = np.polyval(
        polynomial,
        x_samples,
    )

    # Производная параболы:
    #
    # y = ax² + bx + c
    #
    # y' = 2ax + b
    slopes = (
        2 * a * x_samples + b
    )

    # Считаем длину дуги.
    arc_factor = np.sqrt(
        1.0 + slopes ** 2
    )

    dx = np.diff(x_samples)

    cumulative_length = np.concatenate(
        [
            [0],
            np.cumsum(
                (
                    arc_factor[:-1]
                    + arc_factor[1:]
                )
                / 2
                * dx
            ),
        ]
    )

    arc_length = cumulative_length[-1]

    output_width = max(
        120,
        int(round(arc_length)),
    )

    # Теперь идём не по X,
    # а равномерно по длине самой дуги.
    output_positions = np.linspace(
        0,
        arc_length,
        output_width,
    )

    curve_x = np.interp(
        output_positions,
        cumulative_length,
        x_samples,
    )

    curve_y = np.polyval(
        polynomial,
        curve_x,
    )

    slopes = (
        2 * a * curve_x + b
    )

    length = np.sqrt(
        1.0 + slopes ** 2
    )

    # Нормаль к кривой.
    normal_x = -slopes / length
    normal_y = 1.0 / length

    vertical_offsets = (
        np.arange(
            output_height,
            dtype=np.float32,
        )
        - (output_height - 1) / 2
    )

    map_x = (
        curve_x[None, :]
        + vertical_offsets[:, None]
        * normal_x[None, :]
    )

    map_y = (
        curve_y[None, :]
        + vertical_offsets[:, None]
        * normal_y[None, :]
    )

    rectified = cv2.remap(
        image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    debug_data = {
        "polynomial": polynomial,
        "mask": mask,
        "points": points,
        "accepted": accepted,
        "x_min": x_min,
        "x_max": x_max,
    }

    return rectified, debug_data


def create_debug_image(image, debug_data):

    debug = image.copy()

    polynomial = debug_data["polynomial"]

    points = debug_data["points"]
    accepted = debug_data["accepted"]

    x_min = debug_data["x_min"]
    x_max = debug_data["x_max"]

    # Показываем найденные компоненты.
    for point, ok in zip(
        points,
        accepted,
    ):

        cx, cy = point[:2]

        if ok:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.circle(
            debug,
            (int(cx), int(cy)),
            2,
            color,
            -1,
        )

    # Рисуем найденную параболу.
    previous = None

    for x in range(
        x_min,
        x_max + 1,
    ):

        y = int(
            np.polyval(
                polynomial,
                x,
            )
        )

        point = (x, y)

        if previous is not None:
            cv2.line(
                debug,
                previous,
                point,
                (255, 0, 0),
                1,
            )

        previous = point

    return debug


def process_image(
    input_path,
    output_path,
    debug=False,
):

    image = cv2.imread(
        str(input_path)
    )

    if image is None:
        raise RuntimeError(
            f"Не удалось открыть {input_path}"
        )

    rectified, debug_data = rectify_image(
        image
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        rectified,
    )

    print(
        f"[OK] {input_path.name} "
        f"-> {output_path.name}"
    )

    if debug:

        debug_image = create_debug_image(
            image,
            debug_data,
        )

        debug_path = (
            output_path.parent
            / f"{output_path.stem}_debug.jpg"
        )

        mask_path = (
            output_path.parent
            / f"{output_path.stem}_mask.jpg"
        )

        cv2.imwrite(
            str(debug_path),
            debug_image,
        )

        cv2.imwrite(
            str(mask_path),
            debug_data["mask"],
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input",
        help="Файл или папка с изображениями",
    )

    parser.add_argument(
        "--output",
        default="rectified",
        help="Папка для результата",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Сохранить маску и найденную кривую",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    output_dir = Path(args.output)

    if input_path.is_file():

        files = [input_path]

    elif input_path.is_dir():

        files = sorted(
            file
            for file in input_path.iterdir()
            if file.suffix.lower()
            in IMAGE_EXTENSIONS
        )

    else:

        raise RuntimeError(
            f"Не найден путь: {input_path}"
        )

    if not files:

        raise RuntimeError(
            "В папке не найдено изображений"
        )

    print(
        f"Найдено изображений: {len(files)}"
    )

    for file in files:

        output_file = (
            output_dir
            / f"{file.stem}_rectified.jpg"
        )

        try:

            process_image(
                file,
                output_file,
                debug=args.debug,
            )

        except Exception as error:

            print(
                f"[ERROR] {file.name}: {error}"
            )


if __name__ == "__main__":
    main()