import pygame

class Colony:
  x: int
  y: int
  radius: float

  foodCollected = 0

  def __init__(self, x: int, y: int):
    self.x = x
    self.y = y
    self.radius = 30

  def collect(self):
    self.foodCollected += 1

  def draw(self, screen: pygame.Surface):
    self.image = pygame.draw.circle(screen, "black", (self.x, self.y), self.radius, self.radius * 2)    