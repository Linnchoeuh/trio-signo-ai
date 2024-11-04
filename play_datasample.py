import sys
import json
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
    print("Usage: python play_datasample.py <json_file>")
    exit(1)

sample: DataSample2 = DataSample2.from_json_file(sys.argv[1])
# sample.mirror_sample(x=True, y=False, z=False)
# sample.translate_sample(0.1, 0, 0)
# sample.reframe(15)

print(len(sample.gestures))

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

while run:
    frame %= len(sample.gestures)
    print(f"\r\033[Kframe: {frame + 1}/{len(sample.gestures)}", end="")
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

    # pygame.draw.line(win, (128, 128, 128), from_coord_list_xyz(hand_frame.wrist), from_coord_list_xyz(hand_frame.index_mcp), line_size)
    # pygame.draw.line(win, (128, 128, 128), from_coord_list_xyz(hand_frame.wrist), from_coord_list_xyz(hand_frame.pinky_mcp), line_size)
    # pygame.draw.line(win, (128, 128, 128), from_coord_list_xyz(hand_frame.wrist), from_coord_list_xyz(hand_frame.thumb_cmc), line_size)

    # pygame.draw.line(win, (128, 128, 128), from_coord_list_xyz(hand_frame.index_mcp), from_coord_list_xyz(hand_frame.middle_mcp), line_size)
    # pygame.draw.line(win, (128, 128, 128), from_coord_list_xyz(hand_frame.middle_mcp), from_coord_list_xyz(hand_frame.ring_mcp), line_size)
    # pygame.draw.line(win, (128, 128, 128), from_coord_list_xyz(hand_frame.ring_mcp), from_coord_list_xyz(hand_frame.pinky_mcp), line_size)

    # pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.thumb_cmc), from_coord_list_xyz(hand_frame.thumb_mcp), line_size)
    # pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.thumb_mcp), from_coord_list_xyz(hand_frame.thumb_ip), line_size)
    # pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.thumb_ip), from_coord_list_xyz(hand_frame.thumb_tip), line_size)

    # pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.index_mcp), from_coord_list_xyz(hand_frame.index_pip), line_size)
    # pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.index_pip), from_coord_list_xyz(hand_frame.index_dip), line_size)
    # pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.index_dip), from_coord_list_xyz(hand_frame.index_tip), line_size)

    # pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.middle_mcp), from_coord_list_xyz(hand_frame.middle_pip), line_size)
    # pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.middle_pip), from_coord_list_xyz(hand_frame.middle_dip), line_size)
    # pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.middle_dip), from_coord_list_xyz(hand_frame.middle_tip), line_size)

    # pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.ring_mcp), from_coord_list_xyz(hand_frame.ring_pip), line_size)
    # pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.ring_pip), from_coord_list_xyz(hand_frame.ring_dip), line_size)
    # pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.ring_dip), from_coord_list_xyz(hand_frame.ring_tip), line_size)

    # pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.pinky_mcp), from_coord_list_xyz(hand_frame.pinky_pip), line_size)
    # pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.pinky_pip), from_coord_list_xyz(hand_frame.pinky_dip), line_size)
    # pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.pinky_dip), from_coord_list_xyz(hand_frame.pinky_tip), line_size)


    for pos in hand_frame.__dict__.values():
        if pos is None:
            continue
        dist = pow(((-pos[2] + 1) / 2) * 2.5, 10) * 0.5
        # print(dist)
        pygame.draw.circle(win, (min(255, int(30 * dist)), 0, 0), from_coord_list_xyz(pos), dist)

    clock.tick(FPS)
    if play_animation:
        if pause_animation == 0 and frame >= len(sample_cpy.gestures) - 1:
            pause_animation = time.time()
        elif time.time() - pause_animation > 2:
            pause_animation = 0
            frame += 1
    pygame.display.update()
