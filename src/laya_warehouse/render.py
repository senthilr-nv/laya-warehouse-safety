"""Pygame renderer for recorded warehouse episodes."""

from __future__ import annotations

from typing import Any


def render_record(record: dict[str, Any], *, fps: int = 4) -> None:
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError(
            "Visual replay requires pygame. Install with: pip install -e '.[visual]'"
        ) from exc

    pygame.init()
    cell = 58
    margin = 28
    grid_width = 11 * cell
    grid_height = 9 * cell
    hud_width = 300
    screen_size = (grid_width + hud_width + margin * 3, grid_height + margin * 2)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Laya Warehouse Safety")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)
    small = pygame.font.SysFont("Arial", 15)

    colors = {
        "background": (20, 24, 31),
        "grid": (56, 64, 76),
        "aisle": (34, 41, 51),
        "goal": (48, 111, 82),
        "robot": (118, 185, 0),
        "worker": (74, 144, 226),
        "forklift": (244, 164, 66),
        "text": (235, 239, 245),
        "muted": (163, 171, 184),
        "danger": (225, 84, 84),
    }

    def draw_text(text: str, x: int, y: int, color: tuple[int, int, int], *, body=False) -> None:
        screen.blit((small if body else font).render(text, True, color), (x, y))

    frames = record["frames"]
    frame_index = 0
    running = True
    paused = False
    while running and frames:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    frame_index = min(frame_index + 1, len(frames) - 1)
                elif event.key == pygame.K_LEFT:
                    frame_index = max(frame_index - 1, 0)

        frame = frames[frame_index]
        state = frame["after"]
        screen.fill(colors["background"])

        origin_x, origin_y = margin, margin
        for y in range(state["height"]):
            for x in range(state["width"]):
                rect = pygame.Rect(origin_x + x * cell, origin_y + y * cell, cell, cell)
                fill = (
                    colors["goal"]
                    if y == 0 and x in state["goal"]["columns"]
                    else colors["aisle"]
                )
                pygame.draw.rect(screen, fill, rect)
                pygame.draw.rect(screen, colors["grid"], rect, 1)

        for actor in state["actors"]:
            center = (
                origin_x + actor["x"] * cell + cell // 2,
                origin_y + actor["y"] * cell + cell // 2,
            )
            color = colors[actor["kind"]]
            if actor["kind"] == "worker":
                pygame.draw.circle(screen, color, center, cell // 4)
            else:
                pygame.draw.rect(
                    screen,
                    color,
                    pygame.Rect(
                        center[0] - cell // 3,
                        center[1] - cell // 5,
                        cell * 2 // 3,
                        cell * 2 // 5,
                    ),
                    border_radius=5,
                )

        robot = state["robot"]
        robot_center = (
            origin_x + robot["x"] * cell + cell // 2,
            origin_y + robot["y"] * cell + cell // 2,
        )
        pygame.draw.circle(screen, colors["robot"], robot_center, cell // 3)
        pygame.draw.circle(screen, colors["text"], robot_center, cell // 3, 2)

        hud_x = origin_x + grid_width + margin
        draw_text("Laya Warehouse Safety", hud_x, margin, colors["text"])
        draw_text(
            f"Frame {frame_index + 1}/{len(frames)}",
            hud_x,
            margin + 38,
            colors["muted"],
            body=True,
        )
        draw_text(
            f"Controller: {record['controller']}",
            hud_x,
            margin + 64,
            colors["text"],
            body=True,
        )
        draw_text(
            f"Requested: {frame['step']['requested_action']}",
            hud_x,
            margin + 102,
            colors["text"],
            body=True,
        )
        draw_text(
            f"Applied: {frame['step']['applied_action']}",
            hud_x,
            margin + 128,
            colors["text"],
            body=True,
        )
        draw_text(
            f"Latency: {frame['decision']['latency_ms']:.2f} ms",
            hud_x,
            margin + 154,
            colors["text"],
            body=True,
        )
        if frame["step"]["safety_override"]:
            draw_text("Safety override", hud_x, margin + 190, colors["danger"])
        draw_text(frame["step"]["reason"][:34], hud_x, margin + 224, colors["muted"], body=True)
        draw_text("Space: pause", hud_x, margin + 300, colors["muted"], body=True)
        draw_text("←/→: inspect frames", hud_x, margin + 326, colors["muted"], body=True)

        pygame.display.flip()
        if not paused:
            frame_index += 1
            if frame_index >= len(frames):
                frame_index = len(frames) - 1
                paused = True
        clock.tick(max(1, fps))

    pygame.quit()
