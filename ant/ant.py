import pygame
import math
import random
import numpy as np

image_path = "assets/antskin.png"

class Ant:
  image: pygame.Surface
  x: int
  y: int
  rotation: int = 0

  originalImages: list[pygame.Surface]
  originalImage: pygame.Surface

  centerImageX: int
  centerImageY: int

  radius: int = 5

  centerX: int = 0
  centerY: int = 0

  hasFood: bool = False

  # Point is gained whenever 
  points: int = 0

  speed: int = 1

  angle: int = 0

  animationFrame: int = random.randrange(1, 15)
  
  def __init__(self, sprites: list[pygame.Surface], imageSizeX: int, imageSizeY: int):

    self.imageSizeX = imageSizeX
    self.imageSizeY = imageSizeY
    self.centerImageX = self.imageSizeX / 2
    self.centerImageY = self.imageSizeY / 2

    self.originalImages = sprites
    self.originalImage = self.originalImages[self.animationFrame - 1]
    self.image = self.originalImage

  def draw(self, surface: pygame.Surface, x: int, y: int):
    self.x = x
    self.y = y
    rect = surface.blit(self.image, (self.x, self.y))
    self.centerX = rect.centerx
    self.centerY = rect.centery

  def spawn(self, displaySizeX: int, displaySizeY: int):
    self.x = (displaySizeX / 2) - (self.imageSizeX / 2)
    self.y = (displaySizeY / 2) - (self.imageSizeY / 2)

  def animate(self):
    self.animationFrame += 1
    if self.animationFrame > 14:
      self.animationFrame = 1

    self.originalImage = self.originalImages[self.animationFrame - 1]

  def get_state(self):
    state = [
      self.centerX,
      self.centerY,
      self.angle,
    ]

    return np.array(state, dtype=int)

  def moveTo(self, x, y):
    dx = x - self.centerX
    dy = y - self.centerY

    angle = math.atan2(dx, dy)
    normal_vx = math.sin(angle)
    normal_vy = math.cos(angle)

    angleDegrees = (angle * (180/math.pi)) - 90

    self.image = pygame.transform.rotate(self.originalImage, angleDegrees)

    self.x += normal_vx * self.speed
    self.y += normal_vy * self.speed

  def play(self, state):
    print("state", state)

  def found_food(self):
    if self.hasFood == False:
      self.hasFood = True
      print("FOUND FOOD")

  def found_colony(self):
    if self.hasFood == True:
      self.hasFood = False
      self.points += 1
      print("FOUND COLONY WITH FOOD")

  def collide(self, x, y, radius, type: str):
    
    dx = abs(self.centerX - x)
    dy = abs(self.centerY - y)
    
    distance = math.sqrt(dx**2 + dy**2)
    
    if distance < radius + self.radius and type == "food":
      self.found_food()
      return True
    if distance < radius + self.radius and type == "colony":
      self.found_colony()
      return True
    
    return False