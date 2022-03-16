import copy
import random as r
from time import sleep
import numpy as np
from PIL import Image, ImageDraw
import cv2


class SimpleBlockRenderer:

    @staticmethod
    def render(cells, cellSize):

        crd = lambda x, y: (x * cellSize, y * cellSize)

        img = Image.new("RGB", (len(cells[0]) * cellSize, len(cells) * cellSize), color="white")
        d = ImageDraw.Draw(img)

        for y, row in enumerate(cells):
            for x, cell in enumerate(row):
                d.rectangle([crd(x, y), crd(x + 1, y + 1)], fill=cells[y][x])

        return img


class Player:
    STAY, LEFT, RIGHT = 0, 1, 2

    def __init__(self):
        self.movementGrid = None

    def getRandomValue(self):
        return r.random() * 2 - 1

    def setup(self, playingfield):
        self.movementGrid = [[self.getRandomValue() for j in range(len(playingfield[0]))] for i in
                             range(len(playingfield))]
        print(self.movementGrid)

    def choose(self, gameState):
        dir = 0
        for y in range(len(gameState)):
            for x in range(len(gameState)):
                if gameState[y][x] == 1:
                    dir += self.movementGrid[y][x]
        if dir < 0:
            return Player.LEFT
        else:
            return Player.RIGHT


class EvadeGame:

    def render(self):
        renderCopy = copy.deepcopy(self.playingField)
        for y in range(len(renderCopy)):
            for x in range(len(renderCopy[0])):
                renderCopy[y][x] = 'black' if renderCopy[y][x] == 1 else 'white'
        renderCopy[len(renderCopy) - 1][int(len(renderCopy[0]) / 2)] = 'red'
        cv2.waitKey(100)
        cv2.imshow("game", np.array(SimpleBlockRenderer.render(renderCopy, 100)))

    def __init__(self, width, height, player, blockChance=0.1, blockChanceIncrease=0.005, render=False):
        self.width = width
        self.height = height
        self.playingField = [[0 for j in range(width)] for i in range(height)]
        self.blockChance = blockChance
        self.blockChanceIncrease = blockChanceIncrease
        self.player = player
        self.player.setup(self.playingField)
        self.running = False
        self.ticks = 0
        self.playerCoordinate = {"x": int(width / 2), "y": height - 1}
        self.shouldRender = render

    def run(self):
        self.running = True
        while self.running:
            self.tick()
        print(f"algorithm ran for {self.ticks} steps")
        return self.ticks

    def getBlockBuffer(self):
        return [1 if r.random() < self.blockChance else 0 for _ in range(self.width)]

    def yTick(self):
        for i in range(self.height - 1):
            self.playingField[self.height - 1 - i] = self.playingField[self.height - 2 - i]
        self.playingField[0] = self.getBlockBuffer()

    def print(self):
        for row in self.playingField:
            print(row)
        print("\n")

    def lost(self):
        return self.playingField[self.playerCoordinate["y"]][self.playerCoordinate["x"]] == 1

    def tick(self):
        result = self.player.choose(self.playingField)
        if result == player.LEFT:
            self.moveLeft()
        else:
            self.moveRight()

        self.yTick()
        if self.shouldRender:
            self.render()

        if self.lost():
            self.running = False

        self.ticks += 1

    def moveRight(self):
        for y in range(len(self.playingField)):
            self.playingField[y] = self.playingField[y][-1:] + self.playingField[y][:-1]

    def moveLeft(self):
        for y in range(len(self.playingField)):
            self.playingField[y] = self.playingField[y][1:] + self.playingField[y][:1]


player = Player()

while True:
    game = EvadeGame(11, 8, player, render=False)
    game.run()
