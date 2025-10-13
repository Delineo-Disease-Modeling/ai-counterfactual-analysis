#graph implementation of agent tracking
#infectivity represented by number of children in the graph

import os
import csv
import math
import scipy.stats as st
import agent_tracker as agt

class Node:
    def __init__(self, id: int, time: int, pid: int, parent):
        self.id = id
        self.time = time
        self.pid = pid
        self.parent = parent
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
    
    def destructive_remove(self, target):
        if not target:
            print("Target not a node: does not exist\n")
            return
        if not target.parent:
            print("Target is head: cannot remove\n")
            return
        ((target.parent).victims).remove(target)
        return
    
    def destructive_remove_by_id(self, head, target_int: int):
        if (target_int < 1): 
            print("Not a valid id: cannot remove\n")
            return
        target = self.searchByID(head, target_int)
        while target:
            self.destructive_remove(target)
            target = self.searchByID(head, target_int)
        return
    
    def remove_and_rebuild(self, target):
        if not target:
            print("Target not a node: does not exist\n")
            return
        if not target.parent:
            print("Target is head: cannot remove\n")
            return
        for v in target.victims:
            ((target.parent).victims).append(v)
        ((target.parent).victims).remove(target)
        return
    
    def remove_and_rebuild_by_id(self, head, target_int: int):
        if (target_int < 1): 
            print("Not a valid id: cannot remove\n")
            return
        target = self.searchByID(head, target_int)
        while target:
            self.remove_and_rebuild(target)
            target = self.searchByID(head, target_int)
        return

    
def graph_test_1():
    run = int(input("Please type the run to test on.\n"))
    data_dir = agt.find_dir(run)
    print("with dupes\n")
    start = Node(-1, -1, -1, None)
    build_agent_graph(start, data_dir)
    start.printNodes(start)
    print("\n")
    print("without same-infector dupes\n")
    start2 = Node(-1, -1, -1, None)
    build_agent_graph_diffpeople(start2, data_dir)
    start2.printNodes(start2)
    print("\n")
    print("without dupes at all\n")
    start3 = Node(-1, -1, -1, None)
    build_agent_graph_nodupes(start3, data_dir)
    start3.printNodes(start3)
    print("\n")
    return 0

def graph_test_2():
    run = int(input("Please type the run to test on.\n"))
    data_dir = agt.find_dir(run)
    start = Node(-1,-1,-1,None)
    build_agent_graph(start, data_dir)
    print("with dupes: %d\n" % direct_infectivity(start, 15))
    start2 = Node(-1,-1,-1,None)
    build_agent_graph_nodupes(start2, data_dir)
    print("no dupes: %d\n" % direct_infectivity(start2, 15))
    return 0

def build_agent_graph(start: Node, data_dir: str):
    with open(os.path.join(data_dir, "infection_logs.csv"), mode="r") as ifile: 
        itable = csv.reader(ifile)
        for row in itable:
            if str(row[6]) != "infector_person_id":
                if str(row[6]) == "":
                    new_node = Node(int(row[1]), 0, -1, start)
                    start.victims.append(new_node)
                else:
                    infector = start.searchByID(start, int(row[6]))
                    new_node = Node(int(row[1]), int(row[0]), int(row[6]), infector)
                    infector.victims.append(new_node)
    return 0

def build_agent_graph_nodupes(start: Node, data_dir: str):
    with open(os.path.join(data_dir, "infection_logs.csv"), mode="r") as ifile: 
        itable = csv.reader(ifile)
        for row in itable:
            if str(row[6]) != "infector_person_id":
                if str(row[6]) == "":
                    new_node = Node(int(row[1]), 0, -1, start)
                    start.victims.append(new_node)
                else:
                    infector = start.searchByID(start, int(row[6]))
                    infected = start.searchByID(start, int(row[1]))
                    if not infected:
                        new_node = Node(int(row[1]), int(row[0]), int(row[6]), infector)
                        infector.victims.append(new_node)
    return 0

def build_agent_graph_diffpeople(start: Node, data_dir: str):
    with open(os.path.join(data_dir, "infection_logs.csv"), mode="r") as ifile: 
        itable = csv.reader(ifile)
        for row in itable:
            if str(row[6]) != "infector_person_id":
                if str(row[6]) == "":
                    new_node = Node(int(row[1]), 0, -1, start)
                    start.victims.append(new_node)
                if str(row[6]) != "":
                    infector = start.searchByID(start, int(row[6]))
                    add_flag = True
                    if infector and len(infector.victims) > 0:
                        for v in infector.victims:
                            if int(row[1]) == v.id:
                                add_flag = False
                    if add_flag == True:
                        new_node = Node(int(row[1]), int(row[0]), int(row[6]), infector)
                        infector.victims.append(new_node)
    return 0

def direct_infectivity(head: Node, target: int):
    count = 0
    target_node = head.searchByID(head, target)
    if target_node:
        for v in target_node.victims:
            count += 1
    return count

def main():
    start_flag = int(input("For testing graph features, type 0. For analysis work, type 1.\n"))
    if (start_flag == 0):
        testnum = int(input("Please select which test to run.\n"))
        if (testnum == 1):
            graph_test_1()
        elif (testnum == 2):
            graph_test_2()
        else: 
            print("Not implemented!\n")
    else:
        print("Not implemented!\n")
    return 1
    
if __name__ == "__main__":
    main()