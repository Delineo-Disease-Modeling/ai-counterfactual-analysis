#graph implementation of agent tracking
#infectivity represented by number of children in the graph

import os
import csv
import math
import scipy.stats as st
import agent_tracker as agt

class Node:
    def __init__(self, id: int, time: int, pid: int):
        self.id = id
        self.time = time
        self.pid = pid
        self.victims = []
    
    def addVictim(self, v):
        self.victims.append(v)

    def searchByID(self, head, target: int):
        if (head):
            currentNode = head
            resultNode = head
            if resultNode == None:
                return None
            elif resultNode.id == target:
                return resultNode
            elif currentNode and (len(currentNode.victims) > 0):
                for n in currentNode.victims: 
                    resultNode = self.searchByID(n, target)
                    if resultNode and resultNode.id == target:
                        return resultNode
        return None
    
    def printNodes(self, head):
        print("%d infected by %d at time %d ->" % (head.id, head.pid, head.time))
        for v in head.victims:
            self.printNodes(v)
        return

    
def graph_test_1():
    start = Node(-1, -1, -1)
    build_agent_graph(start)
    start.printNodes(start)
    return 0

def build_agent_graph(start: Node):
    run = int(input("Please type the run to test on.\n"))
    data_dir = agt.find_dir(run)
    with open(os.path.join(data_dir, "infection_logs.csv"), mode="r") as ifile: 
        itable = csv.reader(ifile)
        for row in itable:
            if str(row[6]) != "infector_person_id":
                if str(row[6]) == "":
                    new_node = Node(int(row[1]), 0, -1)
                    start.victims.append(new_node)
                else:
                    infector = start.searchByID(start, int(row[6]))
                    new_node = Node(int(row[1]), int(row[0]), int(row[6]))
                    infector.victims.append(new_node)
    return 0

def main():
    start_flag = int(input("For testing graph features, type 0. For analysis work, type 1.\n"))
    if (start_flag == 0):
        testnum = int(input("Please select which test to run.\n"))
        if (testnum == 1):
            graph_test_1()
        else: 
            print("Not implemented!\n")
    else:
        print("Not implemented!\n")
    return 1
    
if __name__ == "__main__":
    main()