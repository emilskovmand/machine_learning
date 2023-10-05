import pygame

class Food:
  x: int
  y: int
  radius: float

  def __init__(self, x: int, y: int):
    self.x = x
    self.y = y
    self.radius = 10

  def draw(self, screen: pygame.Surface):
    self.image = pygame.draw.circle(screen, "green", (self.x, self.y), self.radius, self.radius * 2)    