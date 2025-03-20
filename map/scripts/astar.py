import cv2
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue

def read_pgm(file_path):
    """PGM 파일을 읽고 이미지 데이터를 반환합니다."""
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    print("Image shape:", img.shape)  # 이미지의 크기를 출력
    print("Unique values in image:", np.unique(img))  # 이미지의 고유 값을 출력
    return img

def create_grid(img, grid_size):
    """이미지를 격자로 변환합니다."""
    rows, cols = img.shape
    grid = np.zeros((rows // grid_size, cols // grid_size), dtype=int)
    
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            # (i * grid_size)부터 (i * grid_size + grid_size) 범위 내에서 하얀색 (255)이 있는지 확인
            if np.max(img[i * grid_size:(i + 1) * grid_size, j * grid_size:(j + 1) * grid_size]) == 255:
                grid[i, j] = 1  # 장애물이 있는 셀
            else:
                grid[i, j] = 0  # 이동 가능한 셀

    return grid

def visualize_multiple_paths(img, grid, points, grid_size):
    """여러 점을 순차적으로 방문하는 경로를 시각화합니다."""
    img_copy = cv2.resize(img, (grid.shape[1] * grid_size, grid.shape[0] * grid_size), interpolation=cv2.INTER_NEAREST)
    img_copy = np.tile(img_copy[:, :, None], (1, 1, 3))  # 그레이스케일을 RGB로 변환

    total_path = []

    for i in range(len(points)-1):
        start = points[i]
        goal = points[(i + 1) % len(points)]
        path_segment = astar(grid, start, goal)
        total_path.extend(path_segment)

    for point in total_path:
        y, x = point
        img_copy[y * grid_size:(y + 1) * grid_size, x * grid_size:(x + 1) * grid_size] = [0, 255, 0]  # 경로를 녹색으로 표시

    for point in points:
        y, x = point
        img_copy[y * grid_size:(y + 1) * grid_size, x * grid_size:(x + 1) * grid_size] = [255, 0, 0]  # 각 점을 빨간색으로 표시

    plt.imshow(img_copy, origin='lower')
    plt.title('Path Planning')
    plt.show()

def heuristic_cost_estimate(start, goal):
    """휴리스틱 비용 추정 함수 (맨해튼 거리 사용)."""
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

def astar(grid, start, goal):
    """A* 알고리즘을 사용하여 최적 경로를 찾습니다."""
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = {}
    cost_so_far = {start: 0}

    while not frontier.empty():
        current_cost, current = frontier.get()

        if current == goal:
            break

        for next_step in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            neighbor = (current[0] + next_step[0], current[1] + next_step[1])
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor[0], neighbor[1]] != 1:  # 장애물이 없는 경우
                    new_cost = cost_so_far[current] + 1
                    if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                        cost_so_far[neighbor] = new_cost
                        priority = new_cost + heuristic_cost_estimate(neighbor, goal)
                        frontier.put((priority, neighbor))
                        came_from[neighbor] = current

    path = []
    current = goal
    if current in came_from:
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        print(path)
    else:
        print("No path found")

    return path

# PGM 파일 경로 지정
pgm_file_path = '/home/jeong/heven_ws/src/map/maps/cloudGlobal_gimp.pgm'

# PGM 파일 읽기
pgm_img = read_pgm(pgm_file_path)

# 격자 크기 설정
grid_size = 10  # 예: 10픽셀 * 10픽셀

# PGM 이미지를 격자로 변환
grid = create_grid(pgm_img, grid_size)

# 방문할 점들 설정 (격자 단위로 변환)
points = [(257, 236), (928,872), (1079, 315)]  # 예: 여러 개의 점 지정

# 경로 시각화
visualize_multiple_paths(pgm_img, grid, points, grid_size)