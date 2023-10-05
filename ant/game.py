import sys
import pygame
from ant import Ant
from food import Food
from colony import Colony
import numpy as np

from pygame.locals import *


class Game:

  ants: list[Ant] = []
  foods: list[Food] = []

  colonies: list[Colony] = []

  clock: pygame.time.Clock

  antSprites: list[pygame.Surface] = []

  # Animation variables
  animation_delay = 100  # Delay between frames in milliseconds
  current_frame = 0
  last_frame_time = pygame.time.get_ticks()

  screenX = 1000
  screenY = 700
  screen: pygame.Surface

  def __init__(self, FPS = 60):
    self.FPS = FPS

    self.clock = pygame.time.Clock()

    self.ant_imageSizeX = 538 * .036
    self.ant_imageSizeY = 759 * .036

    for number in range(1, 15):
      if number < 10:
        number = f"0{number}"
      sprite = pygame.image.load(f"assets/__black_ant_walk_{number}.png")
      sprite = pygame.transform.scale(sprite, (self.ant_imageSizeX, self.ant_imageSizeY))
      sprite = pygame.transform.rotate(sprite, -90)

      self.antSprites.append(sprite)

    # ANTS
    for _ in range(1):
      newAnt = Ant(self.antSprites, self.ant_imageSizeX, self.ant_imageSizeY)
      newAnt.spawn(self.screenX, self.screenY)
      self.ants.append(newAnt)

    self.foods.append(Food(900, 40))
    self.foods.append(Food(53, 240))
    self.foods.append(Food(860, 530))

    self.colonies.append(Colony(500, 350))

  def init(self):
    pygame.init()

    self.screen = pygame.display.set_mode((self.screenX, self.screenY))

    self.last_frame_time = pygame.time.get_ticks()

  def get_state(self):
    state = []
    for ant in self.ants:
      for stat in ant.get_state():
        state.append(stat)

    return np.array(state, dtype=int)

  def restart(self):
    print("Restart - Not yet implemented")

  def tick(self):

    for event in pygame.event.get():
      if event.type == QUIT:
        pygame.quit()
        sys.exit()

    self.screen.fill("#9b7653")

    # Calculate the elapsed time since the last frame
    current_time = pygame.time.get_ticks()
    elapsed_time = current_time - self.last_frame_time

    mx, my = pygame.mouse.get_pos()

    # Each ant movement
    for ant in self.ants:
      ant.draw(self.screen, ant.x, ant.y)

      if elapsed_time >= self.animation_delay:
        self.last_frame_time = current_time
        ant.animate()

      # Ant collision
      for food in self.foods:
        ant.collide(food.x, food.y, food.radius, "food")
      for colony in self.colonies:
        isCollide = ant.collide(colony.x, colony.y, colony.radius, "colony")
        if isCollide:
          colony.collect()

    for food in self.foods:
      food.draw(self.screen)

    for colony in self.colonies:
      colony.draw(self.screen)
      

    pygame.display.update()
    self.clock.tick(self.FPS)