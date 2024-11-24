import sys
import json
import os
import pygame
from src.datasample import *
from src.rot_3d import *
import copy
import time



BASE_FPS = 15 # Do not change this value
FPS = 60
WIDTH, HEIGHT = 500, 500
scale = 10000
object_position = [WIDTH//2, HEIGHT//2]

if len(sys.argv) < 2:
    print("Usage: python play_datasample.py <json_file/dir>")
    exit(1)

selected_sample: int = 0
samples: list[tuple[DataSample2, str]] = []

try:
    samples = [(DataSample2.from_json_file(sys.argv[1]), sys.argv[1])]
except:
    for file in os.listdir(sys.argv[1]):
        if file.endswith(".json"):
            samples.append((DataSample2.from_json_file(f"{sys.argv[1]}/{file}"), f"{sys.argv[1]}/{file}"))
            # samples[-1][0].noise_sample()
samples.sort(key=lambda x: x[1])
samples.sort(key=lambda x: len(x[1]))
# sample.scale_sample(y=0)
# sample.mirror_sample(x=True, y=False, z=False)
# sample.translate_sample(0.1, 0, 0)
# sample.reframe(15)


p = pygame.init()
run = True
win = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()


def project_3d_to_2d(x, y, z):
    """Simple perspective projection."""
    # Perspective projection formula
    factor = scale / (z + 5)  # Adjust the distance of the projection
    x_2d = x * factor + object_position[0]
    y_2d = -y * factor + object_position[1]  # Invert y-axis for correct orientation
    return (int(x_2d), int(y_2d))

def from_coord_list_xyz(coord_list_xyz):
    return project_3d_to_2d(coord_list_xyz[0], -coord_list_xyz[1], coord_list_xyz[2])

def draw_line_between(win, a: list[float, float, float], b: list[float, float, float], color=(128, 128, 128)):
    if a is None or b is None:
        return
    pygame.draw.line(win, color, from_coord_list_xyz(a), from_coord_list_xyz(b), 2)

frame = 0
rot_x = 0
rot_y = 0
play_animation = False
pause_animation = 0

def render_hand(win, hand_frame: DataGestures):
    draw_line_between(win, hand_frame.r_wrist, hand_frame.r_index_mcp, (128, 255, 128))
    draw_line_between(win, hand_frame.l_wrist, hand_frame.l_index_mcp, (255, 128, 128))
    draw_line_between(win, hand_frame.r_wrist, hand_frame.r_pinky_mcp, (128, 255, 128))
    draw_line_between(win, hand_frame.l_wrist, hand_frame.l_pinky_mcp, (255, 128, 128))
    draw_line_between(win, hand_frame.r_wrist, hand_frame.r_thumb_cmc, (128, 255, 128))
    draw_line_between(win, hand_frame.l_wrist, hand_frame.l_thumb_cmc, (255, 128, 128))

    draw_line_between(win, hand_frame.r_index_mcp, hand_frame.r_middle_mcp, (128, 255, 128))
    draw_line_between(win, hand_frame.l_index_mcp, hand_frame.l_middle_mcp, (255, 128, 128))
    draw_line_between(win, hand_frame.r_middle_mcp, hand_frame.r_ring_mcp, (128, 255, 128))
    draw_line_between(win, hand_frame.l_middle_mcp, hand_frame.l_ring_mcp, (255, 128, 128))
    draw_line_between(win, hand_frame.r_ring_mcp, hand_frame.r_pinky_mcp, (128, 255, 128))
    draw_line_between(win, hand_frame.l_ring_mcp, hand_frame.l_pinky_mcp, (255, 128, 128))

    draw_line_between(win, hand_frame.r_thumb_cmc, hand_frame.r_thumb_mcp, (128, 255, 128))
    draw_line_between(win, hand_frame.l_thumb_cmc, hand_frame.l_thumb_mcp, (255, 128, 128))
    draw_line_between(win, hand_frame.r_thumb_mcp, hand_frame.r_thumb_ip, (128, 255, 128))
    draw_line_between(win, hand_frame.l_thumb_mcp, hand_frame.l_thumb_ip, (255, 128, 128))
    draw_line_between(win, hand_frame.r_thumb_ip, hand_frame.r_thumb_tip, (128, 255, 128))
    draw_line_between(win, hand_frame.l_thumb_ip, hand_frame.l_thumb_tip, (255, 128, 128))

    draw_line_between(win, hand_frame.r_index_mcp, hand_frame.r_index_pip, (128, 255, 128))
    draw_line_between(win, hand_frame.l_index_mcp, hand_frame.l_index_pip, (255, 128, 128))
    draw_line_between(win, hand_frame.r_index_pip, hand_frame.r_index_dip, (128, 255, 128))
    draw_line_between(win, hand_frame.l_index_pip, hand_frame.l_index_dip, (255, 128, 128))
    draw_line_between(win, hand_frame.r_index_dip, hand_frame.r_index_tip, (128, 255, 128))
    draw_line_between(win, hand_frame.l_index_dip, hand_frame.l_index_tip, (255, 128, 128))

    draw_line_between(win, hand_frame.r_middle_mcp, hand_frame.r_middle_pip, (128, 255, 128))
    draw_line_between(win, hand_frame.l_middle_mcp, hand_frame.l_middle_pip, (255, 128, 128))
    draw_line_between(win, hand_frame.r_middle_pip, hand_frame.r_middle_dip, (128, 255, 128))
    draw_line_between(win, hand_frame.l_middle_pip, hand_frame.l_middle_dip, (255, 128, 128))
    draw_line_between(win, hand_frame.r_middle_dip, hand_frame.r_middle_tip, (128, 255, 128))
    draw_line_between(win, hand_frame.l_middle_dip, hand_frame.l_middle_tip, (255, 128, 128))

    draw_line_between(win, hand_frame.r_ring_mcp, hand_frame.r_ring_pip, (128, 255, 128))
    draw_line_between(win, hand_frame.l_ring_mcp, hand_frame.l_ring_pip, (255, 128, 128))
    draw_line_between(win, hand_frame.r_ring_pip, hand_frame.r_ring_dip, (128, 255, 128))
    draw_line_between(win, hand_frame.l_ring_pip, hand_frame.l_ring_dip, (255, 128, 128))
    draw_line_between(win, hand_frame.r_ring_dip, hand_frame.r_ring_tip, (128, 255, 128))
    draw_line_between(win, hand_frame.l_ring_dip, hand_frame.l_ring_tip, (255, 128, 128))

    draw_line_between(win, hand_frame.r_pinky_mcp, hand_frame.r_pinky_pip, (128, 255, 128))
    draw_line_between(win, hand_frame.l_pinky_mcp, hand_frame.l_pinky_pip, (255, 128, 128))
    draw_line_between(win, hand_frame.r_pinky_pip, hand_frame.r_pinky_dip, (128, 255, 128))
    draw_line_between(win, hand_frame.l_pinky_pip, hand_frame.l_pinky_dip, (255, 128, 128))
    draw_line_between(win, hand_frame.r_pinky_dip, hand_frame.r_pinky_tip, (128, 255, 128))
    draw_line_between(win, hand_frame.l_pinky_dip, hand_frame.l_pinky_tip, (255, 128, 128))


    for pos in hand_frame.__dict__.values():
        if pos is None:
            continue
        dist = pow(((-pos[2] + 1) / 2) * 2.5, 10) * 0.5
        # print(dist)
        pygame.draw.circle(win, (min(255, int(30 * dist)), 0, 0), from_coord_list_xyz(pos), dist)

print(f"""
Controls:
    - Arrow keys to rotate
    - Space to play/pause animation
    - S to go back one frame
    - Z to go forward one frame
    - R to reset rotation
    - PageUp to select previous sample (if multiple samples are loaded)
    - PageDown to select next sample (if multiple samples are loaded)
    - W to save current sample to file
    - M to swap hands (In case the hands are inverted)
      """)

while run:
    sample: DataSample2 = samples[selected_sample][0]
    frame %= len(sample.gestures)
    print(f"\r\033[Kframe: {frame + 1}/{len(sample.gestures)}, Sample: {selected_sample}/{len(samples)} [{samples[selected_sample][1]}]", end="")
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                run = False
            if event.key == pygame.K_s:
                frame -= 1
            if event.key == pygame.K_z:
                frame += 1
            if event.key == pygame.K_SPACE:
                play_animation = not play_animation
            if event.key == pygame.K_r:
                rot_x = 0
                rot_y = 0
            if event.key == pygame.K_PAGEUP:
                selected_sample = (selected_sample - 1) % len(samples)
            if event.key == pygame.K_PAGEDOWN:
                selected_sample = (selected_sample + 1) % len(samples)
            if event.key == pygame.K_w:
                sample.to_json_file(samples[selected_sample][1])
            if event.key == pygame.K_m:
                sample.swap_hands()

    if pygame.key.get_pressed()[pygame.K_LEFT]:
        rot_y -= 0.2 * BASE_FPS / FPS
    if pygame.key.get_pressed()[pygame.K_RIGHT]:
        rot_y += 0.2 * BASE_FPS / FPS
    if pygame.key.get_pressed()[pygame.K_UP]:
        rot_x -= 0.2 * BASE_FPS / FPS
    if pygame.key.get_pressed()[pygame.K_DOWN]:
        rot_x += 0.2 * BASE_FPS / FPS

    win.fill((0, 0, 0))

    sample_cpy: DataSample2 = copy.deepcopy(sample)

    # sample_cpy.round_gesture_coordinates(round_decimal)
    # sample_cpy.randomize_points()
    # sample_cpy.deform_hand(1.5, 1, 1)

    hand_frame: DataGestures = copy.deepcopy(sample_cpy.gestures[frame % len(sample_cpy.gestures)])
    hand_frame.rotate(rot_x, rot_y, 0)

    render_hand(win, hand_frame)

    clock.tick(FPS)
    if play_animation:
        if pause_animation == 0 and frame >= len(sample_cpy.gestures) - 1:
            pause_animation = time.time()
        elif time.time() - pause_animation > 2:
            pause_animation = 0
            frame += 1
    pygame.display.update()
