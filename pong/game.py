import pygame
from agent import DQNAgent, ReplayBuffer
import numpy as np
import torch as T

pygame.init()

# Font that is used to render the text
font20 = pygame.font.Font('freesansbold.ttf', 20)

# RGB values of standard colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

# Basic parameters of the screen
WIDTH, HEIGHT = 900, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong")

clock = pygame.time.Clock() 
FPS = 30

# Striker class


class Striker:
		# Take the initial position, dimensions, speed and color of the object
	def __init__(self, posx, posy, width, height, speed, color):
		self.posx = posx
		self.posy = posy
		self.width = width
		self.height = height
		self.speed = speed
		self.color = color
		# Rect that is used to control the position and collision of the object
		self.geekRect = pygame.Rect(posx, posy, width, height)
		# Object that is blit on the screen
		self.geek = pygame.draw.rect(screen, self.color, self.geekRect)

	# Used to display the object on the screen
	def display(self):
		self.geek = pygame.draw.rect(screen, self.color, self.geekRect)

	def update(self, yFac: int):
		self.posy = self.posy + self.speed*yFac

		# Restricting the striker to be below the top surface of the screen
		if self.posy <= 0:
			self.posy = 0
		# Restricting the striker to be above the bottom surface of the screen
		elif self.posy + self.height >= HEIGHT:
			self.posy = HEIGHT-self.height

		# Updating the rect with the new values
		self.geekRect = (self.posx, self.posy, self.width, self.height)

	def get_state(self):

		return np.array([
			self.posx,
			self.posy,
			self.speed,
			self.height,
			self.width
		], dtype=int)

	def displayScore(self, text, score, x, y, color):
		text = font20.render(text+str(score), True, color)
		textRect = text.get_rect()
		textRect.center = (x, y)

		screen.blit(text, textRect)

	def getRect(self):
		return self.geekRect

# Ball class


class Ball:
	def __init__(self, posx, posy, radius, speed, color):
		self.posx = posx
		self.posy = posy
		self.radius = radius
		self.speed = speed
		self.color = color
		self.xFac = 1
		self.yFac = -1
		self.ball = pygame.draw.circle(
			screen, self.color, (self.posx, self.posy), self.radius)
		self.firstTime = 1

	def display(self):
		self.ball = pygame.draw.circle(
			screen, self.color, (self.posx, self.posy), self.radius)

	def update(self):
		self.posx += self.speed*self.xFac
		self.posy += self.speed*self.yFac

		# If the ball hits the top or bottom surfaces, 
		# then the sign of yFac is changed and 
		# it results in a reflection
		if self.posy <= 0 or self.posy >= HEIGHT:
			self.yFac *= -1

		if self.posx <= 0 and self.firstTime:
			self.firstTime = 0
			return 1
		elif self.posx >= WIDTH and self.firstTime:
			self.firstTime = 0
			return -1
		else:
			return 0

	def reset(self):
		self.posx = WIDTH//2
		self.posy = HEIGHT//2
		self.xFac *= -1
		self.firstTime = 1

	def get_state(self):
		return np.array([
			self.posx,
			self.posy,
			self.xFac,
			self.yFac,
			self.speed,
			self.radius
		], dtype=int)

	# Used to reflect the ball along the X-axis
	def hit(self):
		self.xFac *= -1

	def getRect(self):
		return self.ball

# Game Manager


def main():
	running = True

	replay_buffer = ReplayBuffer(capacity=10000)

	# Defining the objects
	geekAi = Striker(20, 0, 10, 100, 10, GREEN)
	geek = Striker(WIDTH-30, 0, 10, 100, 10, GREEN)
	ball = Ball(WIDTH//2, HEIGHT//2, 7, 7, WHITE)

	# Initial parameters of the players
	geekAiScore, geek2Score = 0, 0
	geek2YFac = 0

	input_dims = np.concatenate((geekAi.get_state(), geek.get_state(), ball.get_state()), dtype=int)

	num_actions = 3
	dqn_agent = DQNAgent(input_dims=input_dims, num_actions=num_actions)

	earlier_state = input_dims

	while running:
		screen.fill(BLACK)

		# Event handling
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_UP:
					geek2YFac = -1
				if event.key == pygame.K_DOWN:
					geek2YFac = 1
			if event.type == pygame.KEYUP:
				if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
					geek2YFac = 0

		aiHit = 0
		if pygame.Rect.colliderect(ball.getRect(), geekAi.getRect()):
			ball.hit()
			aiHit = 1
		if pygame.Rect.colliderect(ball.getRect(), geek.getRect()):
			ball.hit()

		game_state = np.concatenate((geekAi.get_state(), geek.get_state(), ball.get_state()))

		action = dqn_agent.select_action(game_state)

		# Updating the objects
		geekAi.update(action - 1)
		geek.update(geek2YFac)
		point = ball.update()

		if point == -1:
			geekAiScore += 1
		elif point == 1:
			geek2Score += 1

		if point: 
			ball.reset()

		replay_buffer.push(earlier_state, action, game_state, aiHit)

		earlier_state = game_state

		dqn_agent.train(replay_buffer=replay_buffer)

		# Displaying the objects on the screen
		geekAi.display()
		geek.display()
		ball.display()

		# Displaying the scores of the players
		geekAi.displayScore("AI : ", 
						geekAiScore, 100, 20, WHITE)
		geek.displayScore("User : ", 
						geek2Score, WIDTH-100, 20, WHITE)

		pygame.display.update()
		clock.tick(FPS)	 


if __name__ == "__main__":
	main()
	pygame.quit()


# Game made by teja00219 on https://www.geeksforgeeks.org/create-a-pong-game-in-python-pygame/