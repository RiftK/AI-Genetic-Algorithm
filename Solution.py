from SetCoveringProblemCreator import *
import time
import random
import numpy as np
from matplotlib import pyplot as plt

MeanbestFitnessesForSubset = {}

def getFitness(individual, listOfAllSubsets):
    """
    individal -> boolean list
    listOfAllSubsets -> list of lists, the original problem

    returns fitness score of individual
    """
    elementTaken = [0] * 101

    for index, value in enumerate(individual):
        if (value == 1):
            for element in listOfAllSubsets[index]:
                elementTaken[element] = 1

    coveredCount = sum(elementTaken)
    uncoveredCount = 100 - coveredCount

    fitnessScore = 0
    numberOfUntakenSets = len(listOfAllSubsets) - sum(individual)
    if (coveredCount == 100):
        fitnessScore = 100 + numberOfUntakenSets/len(listOfAllSubsets) * 100
    else:
        fitnessScore = 100 - uncoveredCount

    return fitnessScore


def reproduce(parent1, parent2):
    if (random.random() <= 0.5):
        parent1, parent2 = parent2, parent1
    
    random_number = random.randint(1, len(parent1) - 1)
    child = parent1[0:random_number] + parent2[random_number:]

    return child


def mutate(individual, mutationRate):

    if (random.random() < 0.2):
        for index, value in enumerate(individual):
            if (random.random() < mutationRate):
                individual[index] = 1 - value


def getNormalizedFitness(fitnessList):

    totalSum = sum(fitnessList)
    normalizedFitness = [fitness/totalSum for fitness in fitnessList]      
    return normalizedFitness


def initPopulation(populationSize, listOfAllSubsets):
    population = []

    for _ in range(populationSize):
        individual = []
        for _ in range(len(listOfAllSubsets)):
            individual.append(np.random.choice([0, 1], p=[0.5, 0.5]))
        population.append(individual)

    return population


def GeneticAlgorithm(populationSize, listOfAllSubsets, mutationRate, numberOfGenerations = 1000):
    """
    populationSize -> size of the population
    listOfAllSubsets -> list of lists, the original problem
    mutationRate -> mutation rate of the algorithm
    numberOfGenerations -> number of generations to run the algorithm
    """
    population = initPopulation(populationSize, listOfAllSubsets)
    startTime = time.time()

    for currentGeneration in range(numberOfGenerations):

        newPopulation = []

        # Keeping best member of the population
        fitnessList = [getFitness(individual, listOfAllSubsets) for individual in population]
        bestIndividual = population[np.argmax(fitnessList)]
        newPopulation.append(bestIndividual)
        
        for _ in range(populationSize - 1):

            normalizedFitness = getNormalizedFitness(fitnessList)

            indexList = [i for i in range(len(population))]
            parent1Index, parent2Index = np.random.choice(a = indexList, size = 2, p = normalizedFitness)
            parent1 = population[parent1Index]
            parent2 = population[parent2Index]
            
            child = reproduce(parent1, parent2)

            mutate(child, mutationRate)
            
            newPopulation.append(child)

        population = newPopulation

        timeElapsed = time.time() - startTime

        if (timeElapsed > 40):
            break
    
    bestIndividual = []
    bestFitness = -1
    for individual in population:
        currentFitness = getFitness(individual, listOfAllSubsets)
        if (currentFitness > bestFitness):
            bestIndividual = individual
            bestFitness = currentFitness

    return bestIndividual


def printDetails(individual, listOfAllSubsets):
    elementTaken = [0] * 101

    for index, value in enumerate(individual):
        if (value == 1):
            for element in listOfAllSubsets[index]:
                elementTaken[element] = 1

    subsetsTaken = sum(individual)

    print("Roll Number: 2021A7PS1441G")
    print("Number of subsets in scp_test.json file :", len(listOfAllSubsets))
    print("Solution:")
  
    for index, value in enumerate(individual):
        if (index == len(individual) - 1):
            print(f"{index}:{value}")
        else:
            print(f"{index}:{value}", end = ", ")
    
    print("Fitness value of best state :", getFitness(individual, listOfAllSubsets))
    print("Minimum number of subsets that can cover the Universe-set :", subsetsTaken)


def main():
    scp = SetCoveringProblemCreator()

    startTime = time.time()
    listOfSubsets = scp.ReadSetsFromJson("scp_test.json")
    bestIndividual = GeneticAlgorithm(populationSize=50, listOfAllSubsets=listOfSubsets, mutationRate=0.01, numberOfGenerations=5000)
    printDetails(bestIndividual, listOfSubsets)
    timeTaken = round(time.time() - startTime, 2)
    print(f"Time taken : {timeTaken} seconds")


if __name__=='__main__':
    main()