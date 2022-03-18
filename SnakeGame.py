import os
from neat import nn, population
import pygame
import random
import math
import sys
import pickle
from matplotlib import pyplot as plt
import numpy as np
import neat


class Food():
    foodColors = [
        (255, 255, 0),  # yellow
        (0, 255, 255),  # cyan
        (255, 0, 255),  # magenta
        (128, 128, 255),  # light blue
    ]

    def __init__(self, field):
        self.field = field
        self.x = random.randint(0, field.width - 1)
        self.y = random.randint(0, field.height - 1)
        self.color = random.choice(self.foodColors)

    def draw(self):
        size = self.field.blockSize
        rect = (self.x * size, self.y * size, size, size)
        pygame.draw.rect(self.field.screen, self.color, rect, 0)


class Field():
    fieldColor = (0, 50, 0)

    def __init__(self, screen, width, height, blockSize, color=12800):
        self.screen = screen
        self.width = width
        self.height = height
        self.blockSize = blockSize
        self.fieldColor = ((color & 0xFF0000) / 256 ** 2, ((color & 0x00FF00) / 256) % 256, color & 0x0000FF)
        # print self.fieldColor

    def draw(self):
        self.screen.fill(self.fieldColor)


class Snake:
    """A snake has a body that consists of several points.
    The body has a head (first point) and a tail (last point).
    """

    headColor = (50, 255, 0)
    bodyColor = (0, 200, 0)
    damageColor = (255, 0, 0)

    def __init__(self, field,color, len = 5, x = 10, y = 10):
        """Start with a snake in a horizontal position."""
        self.field = field
        self.body = []
        self.growLength = 0
        for i in range(0, len):
            self.body.append((x - i, y))
        self.bodyColor = ((color & 0xFF0000) / 256 **2,((color & 0x00FF00) / 256) % 256,color & 0x0000FF)
        self.headColor = ((color & 0xFF0000) / 256 **2,((color & 0x00FF00) / 256) % 256,color & 0x0000FF)

    def move(self, dx, dy):
        """Move the snake in the given direction."""
        (x, y) = self.body[0] # get old position of head
        x += dx
        y += dy # calculate new position of head
        if self.growLength > 0:
            self.growLength -= 1 # grow
        else:
            self.body.pop() # remove tail
        self.body.insert(0, (x, y)) # insert new position of head
        if self.body[0] in self.body[1:]: # does the head collide with the rest of the body?
            return False # yes, then this is an invalid move
        elif x < 0 or x >= self.field.width or y < 0 or y >= self.field.height: # outside screen?
            return False
        else:
            return True # no

    def grow(self, length = 1):
        self.growLength = length

    def draw(self, damage = False):
        """Draw the body of the snake."""
        for i in range(0, len(self.body)):
            if damage:
                color = self.damageColor
            elif i == 0:
                color = self.headColor
            else:
                color = self.bodyColor

            (x, y) = self.body[i]
            size = self.field.blockSize
            rect = (x * size, y * size, size, size)
            pygame.draw.rect(self.field.screen, color, rect, 0)


rendering = True
debuggin = False
renderdelay = 0

input_names = ['dist_straight_wall', 'dist_straight_food', 'dist_straight_tail', 'dist_left_wall', 'dist_left_food',
               'dist_left_tail', 'dist_right_wall', 'dist_right_food', 'dist_right_tail', 'dist_straight_left_wall',
               'dist_straight_left_food', 'dist_straight_left_tail', 'dist_left_left_wall', 'dist_left_left_food',
               'dist_left_left_tail', 'dist_straight_right_wall', 'dist_straight_right_food',
               'dist_straight_right_tail', 'dist_right_right_wall', 'dist_right_right_food', 'dist_right_right_tail']

blockSize = 16  # size of blocks
width = 16  # size of field width in blocks
height = 16
screenSize = (width * blockSize, height * blockSize)
speed = 1  # milliseconds per step
bg_color = 0x000000
snake_color = 0xFFFFFF
temp_speed = 0

best_foods = 0
best_fitness = 0
loop_punishment = 0.25
near_food_score = 0.2
moved_score = 0.01

# Initialize pygame and open a window
pygame.init()
screen = pygame.display.set_mode(screenSize)

pygame.time.set_timer(pygame.USEREVENT, speed)
clock = pygame.time.Clock()
scr = pygame.surfarray.pixels2d(screen)

dx = 1
dy = 0
generation_number = 0


def get_game_matrix(scr):
    global bg_color
    global snake_color
    res_matrix = []

    for i, x in enumerate(scr):
        res_arr = []
        if (i % blockSize == 0):
            for j, y in enumerate(x):
                if j % blockSize == 0:
                    if scr[i][j] == snake_color:
                        res_arr += [1]
                    elif scr[i][j] == bg_color:
                        res_arr += [0]
                    else:
                        res_arr += [2]
            res_matrix += [res_arr]

    # print res_matrix
    return res_matrix


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


def positive(x):
    return x if x > 0 else 0


def left(orientation):
    (dx, dy) = orientation
    if (dx, dy) == (-1, 0):
        dx, dy = 0, 1
    elif (dx, dy) == (0, 1):
        dx, dy = 1, 0
    elif (dx, dy) == (1, 0):
        dx, dy = 0, -1
    elif (dx, dy) == (0, -1):
        (dx, dy) = (-1, 0)
    return (dx, dy)


def semi_left(orientation):
    (dx, dy) = orientation
    if (dx, dy) == (-1, 0):
        dy = 1
    elif (dx, dy) == (0, 1):
        dx = 1
    elif (dx, dy) == (1, 0):
        dy = -1
    elif (dx, dy) == (0, -1):
        dx = -1
    return (dx, dy)


def right(orientation):
    (dx, dy) = orientation
    if (dx, dy) == (-1, 0):
        (dx, dy) = (0, -1)
    elif (dx, dy) == (0, -1):
        (dx, dy) = (1, 0)
    elif (dx, dy) == (1, 0):
        (dx, dy) = (0, 1)
    elif (dx, dy) == (0, 1):
        (dx, dy) = (-1, 0)
    return (dx, dy)


def semi_right(orientation):
    (dx, dy) = orientation
    if (dx, dy) == (-1, 0):
        dy = -1
    elif (dx, dy) == (0, -1):
        dx = 1
    elif (dx, dy) == (1, 0):
        dy = 1
    elif (dx, dy) == (0, 1):
        dx = -1
    return (dx, dy)


def look_to(orient, pos, game_matrix):
    dx, dy = orient
    px, py = pos

    body_found = False
    food_found = False

    dist_food = width
    dist_tail = width
    dist_wall = width

    dist = 0

    while px >= 0 and px < width and py >= 0 and py < height:
        if not body_found and game_matrix[px][py] == 1:
            dist_tail = dist
            body_found = True
        if not food_found and game_matrix[px][py] == 2:
            dist_food = dist
            food_found = True

        dist_wall = dist
        px += dx
        py += dy
        dist += 1

    return (dist_wall, dist_food, dist_tail)


def get_inputs(game_matrix, position, orientation):  # (dx,dy)

    dx, dy = orientation
    px, py = position

    dist_straight_wall, dist_straight_food, dist_straight_tail = look_to((dx, dy), position, game_matrix)

    dx, dy = left(orientation)
    dist_left_wall, dist_left_food, dist_left_tail = look_to((dx, dy), position, game_matrix)

    dx, dy = right(orientation)
    dist_right_wall, dist_right_food, dist_right_tail = look_to((dx, dy), position, game_matrix)

    dx, dy = semi_left(orientation)
    dist_straight_left_wall, dist_straight_left_food, dist_straight_left_tail = look_to((dx, dy), position, game_matrix)

    dx, dy = semi_left(left(orientation))
    dist_left_left_wall, dist_left_left_food, dist_left_left_tail = look_to((dx, dy), position, game_matrix)

    dx, dy = semi_right(orientation)
    dist_straight_right_wall, dist_straight_right_food, dist_straight_right_tail = look_to((dx, dy), position,
                                                                                           game_matrix)

    dx, dy = semi_right(right(orientation))
    dist_right_right_wall, dist_right_right_food, dist_right_right_tail = look_to((dx, dy), position, game_matrix)

    return [dist_straight_wall, dist_straight_food, dist_straight_tail, dist_left_wall, dist_left_food, dist_left_tail,
            dist_right_wall, dist_right_food, dist_right_tail, dist_straight_left_wall, dist_straight_left_food,
            dist_straight_left_tail, dist_left_left_wall, dist_left_left_food, dist_left_left_tail,
            dist_straight_right_wall, dist_straight_right_food, dist_straight_right_tail, dist_right_right_wall,
            dist_right_right_food, dist_right_right_tail]


def save_best_generation_instance(instance, filename='trained/best_generation_instances.pickle'):
    instances = []
    if os.path.isfile(filename):
        instances = load_object(filename)
    instances.append(instance)
    save_object(instances, filename)


def eval_fitness(genomes, config):
    global best_fitness
    global best_foods
    global screen
    global width
    global height
    global blockSize
    global scr
    global generation_number
    global pop
    global bg_color
    global snake_color
    # global dx
    # global dy
    # global speed
    best_instance = None
    genome_number = 0
    for _, g in genomes:
        net = neat.nn.feed_forward.FeedForwardNetwork.create(g, config)
        dx = 1
        dy = 0
        score = 0.0
        hunger = 100
        # Create the field, the snake and a bit of food
        theField = Field(screen, width, height, blockSize, bg_color)
        theFood = Food(theField)
        theSnake = Snake(theField, snake_color, 5, x=int(width / 2), y=int(height / 2))
        snake_head_x, snake_head_y = theSnake.body[0]
        dist = math.sqrt((snake_head_x - theFood.x) ** 2 + (snake_head_y - theFood.y) ** 2)
        error = 0
        countFrames = 0

        pastPoints = set()

        foods = 0

        while True:
            countFrames += 1

            event = pygame.event.wait()

            if event.type == pygame.QUIT:  # window closed
                print("Quittin")
                # save_object(pop, 'trained/population.dat')  ## export population
                pygame.quit()
                sys.exit()

            if event.type == pygame.USEREVENT:  # timer elapsed
                matrix = get_game_matrix(scr)
                # print matrix
                head_x, head_y = theSnake.body[0]
                head_x += dx
                head_y += dy
                inputs = get_inputs(matrix, (head_x, head_y), (dx, dy))
                if debuggin:
                    print("---------ah---------")
                    for i in range(0, 21):
                        print(input_names[i], " - ", inputs[i])
                outputs = net.activate(inputs)
                direction = outputs.index(max(outputs))
                if direction == 0:  # dont turn
                    # print "Straight"
                    pass

                if direction == 1:  # turn left
                    # print "Left"
                    (dx, dy) = left((dx, dy))
                if direction == 2:  # turn right
                    # print "Right"
                    (dx, dy) = right((dx, dy))

                hunger -= 1
                if not theSnake.move(dx, dy) or hunger <= 0:
                    break
                else:
                    inputs = get_inputs(matrix, (head_x, head_y), (dx, dy))

                    # current_state = inputs[1] < inputs[0]
                    #
                    # wall, bread, wall_left, bread_left, wall_right, bread_right = (inputs)
                    ##score += math.sqrt((theFood.x - theSnake.body[0][0]) ** 2 + (theFood.y - theSnake.body[0][1]) ** 2)
                    score += moved_score
                    pass

            # loop punishment
            if theSnake.body[0] in pastPoints:
                score -= loop_punishment
            pastPoints.add(theSnake.body[0])

            # food
            if theSnake.body[0] == (theFood.x, theFood.y):
                pastPoints = set()
                theSnake.grow()
                theFood = Food(theField)  # make a new piece of food
                score += 5
                hunger += 100
                foods += 1
            else:
                # near food score
                if abs(theSnake.body[0][0] - theFood.x + theSnake.body[0][1] - theFood.y) <= 1:
                    score += near_food_score

            if rendering:
                theField.draw()
                theFood.draw()
                theSnake.draw()
                pygame.display.update()
                pygame.time.wait(renderdelay)

            if event.type == pygame.KEYDOWN:  # key pressed
                if event.key == pygame.K_LEFT:
                    temp_speed = 200
                    pygame.time.set_timer(pygame.USEREVENT, temp_speed)
                elif event.key == pygame.K_RIGHT:
                    temp_speed = speed
                    pygame.time.set_timer(pygame.USEREVENT, temp_speed)

        # Game over!
        if rendering:
            for i in range(0, 2):
                theField.draw()
                theFood.draw()
                theSnake.draw(damage=(i % 2 == 0))
                pygame.display.update()

        # pygame.time.wait(100)
        # score = positive(score)
        g.fitness = score / 100

        if not best_instance or g.fitness > best_fitness:
            best_instance = {
                'num_generation': generation_number,
                'fitness': g.fitness,
                'score': score,
                'genome': g,
                'net': net,
            }
        best_foods = max(best_foods, foods)
        best_fitness = max(best_fitness, g.fitness)
        # if debuggin:
        print(
            f"Generation {generation_number} \tGenome {genome_number} \tFoods {foods} \tBF {best_foods} \tFitness {g.fitness} \tBest fitness {best_fitness} \tScore {score}")
        genome_number += 1

    # save_best_generation_instance(best_instance)
    generation_number += 1
    if generation_number % 20 == 0:
        # save_object(pop, 'trained/population.dat')
        # print("Exporting population")
        # export population
        # save_object(pop,'population.dat')
        # export population
        pass

    global list_best_fitness
    global fig
    list_best_fitness.append(best_fitness)
    line_best_fitness.set_ydata(np.array(list_best_fitness))
    line_best_fitness.set_xdata(list(range(len(list_best_fitness))))
    plt.xlim(0, len(list_best_fitness) - 1)
    plt.ylim(0, max(list_best_fitness) + 0.5)
    fig.canvas.draw()
    fig.canvas.flush_events()


if __name__ == '__main__':
    list_best_fitness = []
    plt.ion()
    fig = plt.figure()
    plt.title('Best fitness')
    ax = fig.add_subplot(111)
    line_best_fitness, = ax.plot(list_best_fitness, 'r-')  # Returns a tuple of line objects, thus the comma

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'configurations/SnakeGame.txt')

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    if len(sys.argv) > 1:
        pop = load_object(sys.argv[1])
        print("Reading popolation from " + sys.argv[1])
    pop.run(eval_fitness, 10000)

