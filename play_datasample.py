import sys
import json
import pygame
from src.datasample import *
from src.rot_3d import *
import copy



BASE_FPS = 15 # Do not change this value
FPS = 60
WIDTH, HEIGHT = 500, 500
scale = 10000
object_position = [WIDTH//2, HEIGHT//2]

if len(sys.argv) < 2:
    print("Usage: python play_datasample.py <json_file>")
    exit(1)

with open(sys.argv[1], 'r', encoding="utf-8") as f:
    sample: DataSample = DataSample.from_json(json.load(f))

# sample.mirror_sample(mirror_x=True, mirror_y=False, mirror_z=False)





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



frame = 0
rot_x = 0
rot_y = 0

while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                run = False

    if pygame.key.get_pressed()[pygame.K_LEFT]:
        rot_y -= 0.2 * BASE_FPS / FPS
    if pygame.key.get_pressed()[pygame.K_RIGHT]:
        rot_y += 0.2 * BASE_FPS / FPS
    if pygame.key.get_pressed()[pygame.K_UP]:
        rot_x -= 0.2 * BASE_FPS / FPS
    if pygame.key.get_pressed()[pygame.K_DOWN]:
        rot_x += 0.2 * BASE_FPS / FPS


    win.fill((0, 0, 0))

    sample_cpy: DataSample = copy.deepcopy(sample)
    # sample_cpy.randomize_points()
    # sample_cpy.deform_hand(1.5, 1.5, 1)

    hand_frame: GestureData = copy.deepcopy(sample_cpy.gestures[frame % len(sample_cpy.gestures)])
    for point_name in hand_frame.__dict__.keys():
        field_value: list[float] = getattr(hand_frame, point_name)
        field_value = rot_3d_x(field_value, rot_x)
        field_value = rot_3d_y(field_value, rot_y)
        setattr(hand_frame, point_name, field_value)
    line_size = scale // 3000


    pygame.draw.line(win, (128, 128, 128), from_coord_list_xyz(hand_frame.wrist), from_coord_list_xyz(hand_frame.index_mcp), line_size)
    pygame.draw.line(win, (128, 128, 128), from_coord_list_xyz(hand_frame.wrist), from_coord_list_xyz(hand_frame.pinky_mcp), line_size)
    pygame.draw.line(win, (128, 128, 128), from_coord_list_xyz(hand_frame.wrist), from_coord_list_xyz(hand_frame.thumb_cmc), line_size)

    pygame.draw.line(win, (128, 128, 128), from_coord_list_xyz(hand_frame.index_mcp), from_coord_list_xyz(hand_frame.middle_mcp), line_size)
    pygame.draw.line(win, (128, 128, 128), from_coord_list_xyz(hand_frame.middle_mcp), from_coord_list_xyz(hand_frame.ring_mcp), line_size)
    pygame.draw.line(win, (128, 128, 128), from_coord_list_xyz(hand_frame.ring_mcp), from_coord_list_xyz(hand_frame.pinky_mcp), line_size)

    pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.thumb_cmc), from_coord_list_xyz(hand_frame.thumb_mcp), line_size)
    pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.thumb_mcp), from_coord_list_xyz(hand_frame.thumb_ip), line_size)
    pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.thumb_ip), from_coord_list_xyz(hand_frame.thumb_tip), line_size)

    pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.index_mcp), from_coord_list_xyz(hand_frame.index_pip), line_size)
    pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.index_pip), from_coord_list_xyz(hand_frame.index_dip), line_size)
    pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.index_dip), from_coord_list_xyz(hand_frame.index_tip), line_size)

    pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.middle_mcp), from_coord_list_xyz(hand_frame.middle_pip), line_size)
    pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.middle_pip), from_coord_list_xyz(hand_frame.middle_dip), line_size)
    pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.middle_dip), from_coord_list_xyz(hand_frame.middle_tip), line_size)

    pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.ring_mcp), from_coord_list_xyz(hand_frame.ring_pip), line_size)
    pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.ring_pip), from_coord_list_xyz(hand_frame.ring_dip), line_size)
    pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.ring_dip), from_coord_list_xyz(hand_frame.ring_tip), line_size)

    pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.pinky_mcp), from_coord_list_xyz(hand_frame.pinky_pip), line_size)
    pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.pinky_pip), from_coord_list_xyz(hand_frame.pinky_dip), line_size)
    pygame.draw.line(win, (255, 255, 255), from_coord_list_xyz(hand_frame.pinky_dip), from_coord_list_xyz(hand_frame.pinky_tip), line_size)


    for pos in hand_frame.__dict__.values():
        pygame.draw.circle(win, (255, 0, 0), from_coord_list_xyz(pos), pow(((-pos[2] + 1) / 2) * 2.5, 10))

    pygame.display.update()
    clock.tick(FPS)
    frame += 1
