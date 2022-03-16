import copy
import random
import random as r
from time import sleep
from typing import Tuple

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
    IDCount = 0
    FitnessVarianceLog = [0.001]

    def __init__(self):
        self.movementGrid = None
        self.ID = -1
        self.getNewID()
        self.mutationCount = 0
        self.fitnessLog = []
        self.age = 0
        self.isSetup = False

    def reset(self):
        self.getNewID()
        self.mutationCount = 0
        self.fitnessLog = []
        self.age = 0

    def getRandomValue(self):
        return r.random() * 2 - 1

    def setup(self, playingfield):
        if not self.isSetup:
            self.movementGrid = [[self.getRandomValue() for j in range(len(playingfield[0]))] for i in range(len(playingfield))]
            self.isSetup = True

    def choose(self, gameState):
        dir = 0
        for y in range(len(gameState)):
            for x in range(len(gameState[0])):
                if gameState[y][x] == 1:
                    dir += self.movementGrid[y][x]
        if dir < 0:
            return Player.LEFT
        else:
            return Player.RIGHT

    def mutate(self, chance):
        for y in range(len(self.movementGrid)):
            for x in range(len(self.movementGrid[0])):
                if r.random() < chance:
                    self.movementGrid[y][x] = self.getRandomValue()
                    self.mutationCount += 1

    def getNewID(self):
        self.ID = Player.IDCount
        Player.IDCount += 1

    def logFitness(self, fitness):
        if fitness not in self.fitnessLog:
            self.fitnessLog.append(fitness)

    def getFitnessVariance(self):
        try:
            variance = (max(self.fitnessLog) - min(self.fitnessLog))
            Player.FitnessVarianceLog.append(variance)
            return variance
        except:
            return 0.0


    def __repr__(self):
        return f"Player(ID: {self.ID}, Mutations: {self.mutationCount}, fitnessVariance: {'{:.3f}'.format(self.getFitnessVariance())}, age: {self.age} )"



class EvadeGame:

    def render(self):
        renderCopy = copy.deepcopy(self.playingField)
        for y in range(len(renderCopy)):
            for x in range(len(renderCopy[0])):
                renderCopy[y][x] = 'black' if renderCopy[y][x] == 1 else 'white'
        renderCopy[len(renderCopy) - 1][int(len(renderCopy[0]) / 2)] = 'red'
        cv2.waitKey(100)
        cv2.imshow("game", np.array(SimpleBlockRenderer.render(renderCopy, 100)))

    def __init__(self, width: int, height: int, player: Player, blockChance=0.1, blockChanceIncrease=0.005, render=False):
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
        # print(f"algorithm ran for {self.ticks} steps")
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
        if result == Player.LEFT:
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


class GeneticAlgorithm:
    def __init__(self, populationSize=10, stopFitness=100, crossOverChance=0.8, mutationChance=0.01, crossOverPercentage=0.5, amountOfGamesToDecideAverage=20, topXIndividualsConsidered=6, elitism=0):
        self.populationSize = populationSize
        self.stopFitness = stopFitness
        self.crossOverChance = crossOverChance
        self.mutationChance = mutationChance
        self.amountOfGamesToDecideAverage = amountOfGamesToDecideAverage
        self.topXIndividualsConsidered = topXIndividualsConsidered
        self.elitism = elitism
        self.population = []
        for _ in range(populationSize):
            p = Player()
            self.population.append((p, self.determineFitness(p)))


        self.crossOverPercentage = crossOverPercentage

    def crossOverPlayers(self, parentA: Player, parentB: Player) -> Tuple[Player, Player]:
        childA, childB = copy.deepcopy(parentA), copy.deepcopy(parentB)
        childA.reset()
        childB.reset()
        for y in range(len(parentA.movementGrid)):
            for x in range(len(parentA.movementGrid[0])):
                if r.random() < self.crossOverPercentage:
                    childA.movementGrid[y][x] = parentB.movementGrid[y][x]
                    childB.movementGrid[y][x] = parentA.movementGrid[y][x]
        return childA, childB

    def determineFitness(self, player: Player) -> float:
        results = []
        player.age += 1
        for i in range(self.amountOfGamesToDecideAverage):
            result = EvadeGame(11, 8, player, render=False).run()
            results.append(result)
        fitness = sum(results)/len(results)
        player.logFitness(fitness)
        return fitness

    def getAverageFitness(self) -> float:
        return sum([player[1] for player in self.population])/len(self.population)

    def stripFitness(self, players):
        return [player[0] for player in players]

    def printSortedPopulation(self, pop):
        for e in sorted(pop, key=lambda x: x[1], reverse=True):
            print(e)
        print("")

    def tick(self):
        oldPopulation = sorted(self.population, key=lambda x: x[1], reverse=True)
        print("oldPopulation")
        self.printSortedPopulation(oldPopulation)


        newPopulation = []

        elites = []
        if self.elitism > 0:
            elites = self.stripFitness(oldPopulation[:self.elitism])
            newPopulation += elites

        while len(newPopulation) < self.populationSize:
            parents = random.sample(oldPopulation[:self.topXIndividualsConsidered], k=2)
            if r.random() < self.crossOverChance:
                newPopulation += self.crossOverPlayers(parents[0][0], parents[1][0])
            else:
                newPopulation += [parents[0][0], parents[1][0]]

        for player in newPopulation:
            if player not in elites:
                player.mutate(self.mutationChance)

        fittedPopulation = [(player, self.determineFitness(player)) for player in newPopulation]

        print("newFittedPopulation")
        self.printSortedPopulation(fittedPopulation)

        self.population = fittedPopulation




        print(f"Average fitness of population = {self.getAverageFitness()}\n"
              f"Average variance = {(sum(Player.FitnessVarianceLog))/len(Player.FitnessVarianceLog)} over {len(Player.FitnessVarianceLog)} entries \n\n")

    def run(self):
        while self.getAverageFitness() < self.stopFitness:
            self.tick()


geneticAlgorithm = GeneticAlgorithm(populationSize=10, elitism=2, mutationChance=0.005, crossOverChance=0.8, amountOfGamesToDecideAverage=100, topXIndividualsConsidered=6)
geneticAlgorithm.run()



#player = Player()

# while True:
#     game = EvadeGame(11, 8, player, render=True)
#     game.run()
