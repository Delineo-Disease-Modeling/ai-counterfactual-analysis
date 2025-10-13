#graph implementation of agent tracking
#infectivity represented by number of children in the graph

import os
import csv
import math
import scipy.stats as st
import agent_tracker as agt

class Node:
    def __init__(self, id: int):
        self.id = id
        self.next = None
        self.victims = []

    def traverseAndPrint(head):
        currentNode = head
        while currentNode:
            print("%d ->" % currentNode.id)
            currentNode = currentNode.next
        print("null")
    
    def addVictim(self, v):
        self.victims.append(v)

    def searchByID(self, head, target: int):
        currentNode = head
        resultNode = head
        if resultNode.id == target:
            return resultNode
        if currentNode and (len(currentNode.victims) > 0):
            for n in currentNode.victims: 
                resultNode = self.searchByID(self, n, target)
                if resultNode.id == target:
                    return resultNode
        return None

def build_agent_graph():
    return 1

def main():
    return 1
    
if __name__ == "__main__":
    main()