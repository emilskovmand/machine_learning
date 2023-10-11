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
FPS = 100

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
			self.geek.centery,
			self.height * 0.5
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

	replay_buffer_1 = ReplayBuffer(capacity=10000)
	replay_buffer_2 = ReplayBuffer(capacity=10000)

	# Defining the objects
	geekAi1 = Striker(20, 0, 10, 100, 10, GREEN)
	geekAi2 = Striker(WIDTH-30, 0, 10, 100, 10, GREEN)
	ball = Ball(WIDTH//2, HEIGHT//2, 7, 7, WHITE)

	# Initial parameters of the players
	geekAi1Score, geekAi2Score = 0, 0
	geek2YFac = 0

	input_dims_1 = np.concatenate((geekAi1.get_state(), ball.get_state()), dtype=int)
	input_dims_2 = np.concatenate((geekAi2.get_state(), ball.get_state()), dtype=int)

	num_actions = 3
	dqn_agent_1 = DQNAgent(input_dims=input_dims_1, num_actions=num_actions)
	dqn_agent_2 = DQNAgent(input_dims=input_dims_2, num_actions=num_actions)

	earlier_state_1 = input_dims_1
	earlier_state_2 = input_dims_2

	aiHit_1_hit_last = False
	aiHit_2_hit_last = False

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

		aiHit_1 = 0
		aiHit_2 = 0
		if pygame.Rect.colliderect(ball.getRect(), geekAi1.getRect()):
			if not aiHit_1_hit_last:
				ball.hit()
				aiHit_1 = 1
				aiHit_1_hit_last = True
		else:
			aiHit_1_hit_last = False
		if pygame.Rect.colliderect(ball.getRect(), geekAi2.getRect()):
			if not aiHit_2_hit_last:
				ball.hit()
				aiHit_2 = 1
				aiHit_2_hit_last = True
		else:
			aiHit_2_hit_last = False

		game_state_1 = np.concatenate((geekAi1.get_state(), ball.get_state()))
		game_state_2 = np.concatenate((geekAi2.get_state(), ball.get_state()))

		ai1_action = dqn_agent_1.select_action(game_state_1)
		ai2_action = dqn_agent_2.select_action(game_state_2)

		# Updating the objects
		geekAi1.update(ai1_action - 1)
		geekAi2.update(ai2_action - 1)
		point = ball.update()

		if point == -1:
			geekAi1Score += 1
		elif point == 1:
			geekAi2Score += 1

		if point: 
			ball.reset()

		replay_buffer_1.push(earlier_state_1, ai1_action, game_state_1, aiHit_1)
		replay_buffer_2.push(earlier_state_2, ai2_action, game_state_2, aiHit_2)

		earlier_state_1 = game_state_1
		earlier_state_2 = game_state_2

		dqn_agent_1.train(replay_buffer=replay_buffer_1)
		dqn_agent_2.train(replay_buffer=replay_buffer_2)

		# Displaying the objects on the screen
		geekAi1.display()
		geekAi2.display()
		ball.display()

		# Displaying the scores of the players
		geekAi1.displayScore("AI_1 : ", 
						geekAi1Score, 100, 20, WHITE)
		geekAi2.displayScore("AI_2 : ", 
						geekAi2Score, WIDTH-100, 20, WHITE)

		pygame.display.update()
		clock.tick(FPS)	 


if __name__ == "__main__":
	main()
	pygame.quit()


# Game made by teja00219 on https://www.geeksforgeeks.org/create-a-pong-game-in-python-pygame/