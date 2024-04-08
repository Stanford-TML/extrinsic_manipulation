import matplotlib.patches as patches
from matplotlib.patches import Arc
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.patches as mpatches

# Change size of figure
plt.rcParams['figure.figsize'] = [20, 16]

def drawPitch(width, height, color="w"):
    fig = plt.figure()
    ax = plt.axes(xlim=(-10, width + 10), ylim=(-15, height + 5))
    plt.axis('off')

    # Grass around pitch
    rect = patches.Rectangle((-5, -5), width + 10, height + 10, linewidth=1, edgecolor='gray', facecolor='#3f995b',
                             capstyle='round')
    ax.add_patch(rect)

    # Pitch boundaries
    rect = plt.Rectangle((0, 0), width, height, ec=color, fc="None", lw=2)
    ax.add_patch(rect)

    # Middle line
    plt.plot([width / 2, width / 2], [0, height], color=color, linewidth=2)

    # Dots
    dots_x = [11, width / 2, width - 11]
    for x in dots_x:
        plt.plot(x, height / 2, 'o', color=color, linewidth=2)

    # Penalty box
    penalty_box_dim = [16.5, 40.3]
    penalty_box_pos_y = (height - penalty_box_dim[1]) / 2

    rect = plt.Rectangle((0, penalty_box_pos_y), penalty_box_dim[0], penalty_box_dim[1], ec=color, fc="None", lw=2)
    ax.add_patch(rect)
    rect = plt.Rectangle((width, penalty_box_pos_y), -penalty_box_dim[0], penalty_box_dim[1], ec=color, fc="None", lw=2)
    ax.add_patch(rect)

    # Goal box
    goal_box_dim = [5.5, penalty_box_dim[1] - 11 * 2]
    goal_box_pos_y = (penalty_box_pos_y + 11)

    rect = plt.Rectangle((0, goal_box_pos_y), goal_box_dim[0], goal_box_dim[1], ec=color, fc="None", lw=2)
    ax.add_patch(rect)
    rect = plt.Rectangle((width, goal_box_pos_y), -goal_box_dim[0], goal_box_dim[1], ec=color, fc="None", lw=2)
    ax.add_patch(rect)

    # Goals
    rect = plt.Rectangle((0, penalty_box_pos_y + 16.5), -3, 7.5, ec=color, fc=color, lw=2, alpha=0.3)
    ax.add_patch(rect)
    rect = plt.Rectangle((width, penalty_box_pos_y + 16.5), 3, 7.5, ec=color, fc=color, lw=2, alpha=0.3)
    ax.add_patch(rect)

    # Middle circle
    mid_circle = plt.Circle([width / 2, height / 2], 9.15, color=color, fc="None", lw=2)
    ax.add_artist(mid_circle)

    # Penalty box arcs
    left = patches.Arc([11, height / 2], 2 * 9.15, 2 * 9.15, color=color, fc="None", lw=2, angle=0, theta1=308,
                       theta2=52)
    ax.add_patch(left)
    right = patches.Arc([width - 11, height / 2], 2 * 9.15, 2 * 9.15, color=color, fc="None", lw=2, angle=180,
                        theta1=308, theta2=52)
    ax.add_patch(right)

    # Arcs on corners
    corners = [[0, 0], [width, 0], [width, height], [0, height]]
    angle = 0
    for x, y in corners:
        c = patches.Arc([x, y], 2, 2, color=color, fc="None", lw=2, angle=angle, theta1=0, theta2=90)
        ax.add_patch(c)
        angle += 90
    return fig, ax


WIDTH = 105
HEIGHT = 68
X_RESIZE = WIDTH
Y_RESIZE = HEIGHT / 0.42


def scale_x(x):
    return (x + 1) * (X_RESIZE / 2)


def scale_y(y):
    return (y + 0.42) * (Y_RESIZE / 2)


def draw_team(team_element, team):
  X = []
  Y = []
  for i in range(11):
    X.append(scale_x(team[i][0]))
    Y.append(scale_y(team[i][1]))
  team_element.set_data(X, Y)

def draw_ball(ball_element, ball):
  ball_element.set_data(scale_x(ball[0]), scale_y(ball[1]))


def draw_result(team_left, team_right, ball, ball_gt):

    fig, ax = drawPitch(WIDTH, HEIGHT)
    ax.invert_yaxis()

    team_left_element, = ax.plot([], [], 'o', markersize=20, markerfacecolor="r", markeredgewidth=0, markeredgecolor="white")
    team_right_element, = ax.plot([], [], 'o', markersize=20, markerfacecolor="b", markeredgewidth=0, markeredgecolor="white")

    ball_element, = ax.plot([], [], 'o', markersize=20, markerfacecolor="black", markeredgewidth=2, markeredgecolor="orange")
    ball_element_gt, = ax.plot([], [], 'o', markersize=20, markerfacecolor="white", markeredgewidth=2, markeredgecolor="orange")

    # Draw players
    draw_team(team_left_element, team_left)
    draw_team(team_right_element, team_right)

    draw_ball(ball_element, ball)
    draw_ball(ball_element_gt, ball_gt)

    return fig
