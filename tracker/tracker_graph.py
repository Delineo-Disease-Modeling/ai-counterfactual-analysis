#graph implementation of agent tracking
#infectivity represented by number of children in the graph

import os
import csv
import math
import scipy.stats as st
import agent_tracker as agt

#Definition of Node class / linked list internal operations

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
            #print("Target not a node: does not exist\n")
            return 1
        if not target.parent:
            #print("Target is head: cannot remove\n")
            return 2
        if target not in (target.parent).victims:
            return 3
        ((target.parent).victims).remove(target)
        target.parent = None
        return 0
    
    def destructive_remove_by_id(self, head, target_int: int):
        if (target_int < 1): 
            #print("Not a valid id: cannot remove\n")
            return 1
        #print("rem check ")
        target = self.searchByID(head, target_int)
        #print("search ")
        while target:
            dummy = self.destructive_remove(target)
            #print("destroy %d " % dummy)
            target = self.searchByID(head, target_int)
            #print("target found %d" % target.id)
        return 0
    
    def remove_and_rebuild(self, target):
        if not target:
            #print("Target not a node: does not exist\n")
            return 1
        if not target.parent:
            #print("Target is head: cannot remove\n")
            return 1
        if target not in (target.parent).victims:
            return 1
        for v in target.victims:
            ((target.parent).victims).append(v)
        ((target.parent).victims).remove(target)
        target.parent = None
        return 0 
    
    def remove_and_rebuild_by_id(self, head, target_int: int):
        if (target_int < 1): 
            #print("Not a valid id: cannot remove\n")
            return 1
        target = self.searchByID(head, target_int)
        while target:
            self.remove_and_rebuild(target)
            target = self.searchByID(head, target_int)
        return 0
    
#Test functions
    
def graph_test_1():
    run = int(input("Please type the run to test on.\n"))
    data_dir = agt.find_dir(run)
    print("without dupes at all\n")
    start3 = Node(-1, -1, -1, None)
    build_agent_graph_nodupes(start3, data_dir)
    start3.printNodes(start3)
    print("\n")
    return 0

def graph_test_2():
    run = int(input("Please type the run to test on.\n"))
    data_dir = agt.find_dir(run)
    start2 = Node(-1,-1,-1,None)
    build_agent_graph_nodupes(start2, data_dir)
    print("no dupes: \n")
    print("direct 15: %d\n" % direct_infectivity(start2, 15))
    print("total 15: %d\n" % total_infectivity_nodupes(start2, 15))
    print("direct 36: %d\n" % direct_infectivity(start2, 36))
    print("total 36: %d\n" % total_infectivity_nodupes(start2, 36))
    return 0

#Construct graph, find the infectivity of one agent

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

def direct_infectivity(head: Node, target: int):
    count = 0
    target_node = head.searchByID(head, target)
    if target_node:
        for v in target_node.victims:
            count += 1
    return count

def total_infectivity_nodupes(head: Node, target_int: int): 
    count = 0
    target = head.searchByID(head, target_int)
    if not target:
        return 0
    count += direct_infectivity(head, target_int)
    if len(target.victims) > 0:
        for v in target.victims:
            count += total_infectivity_nodupes(head, v.id)
    return count

#Calculating mean and variance

def infectivity_mean(head: Node, usable_ids: list[int]):
    mean = float(0.0)
    num_people = len(usable_ids)
    for target_id in usable_ids:
        mean += total_infectivity_nodupes(head, target_id)
    mean = float(mean) / float(num_people)
    return mean

def infectivity_var(head: Node, usable_ids: list[int], mean: float):
    var = float(0.0)
    curr_count = 0
    num_people = len(usable_ids)
    for target_id in usable_ids:
        var += pow(total_infectivity_nodupes(head, target_id) - mean, 2)
    var = float(var) / float(num_people)
    return var

def infectivity_ci_multi():
    runcount = int(input("Please type how many runs you would like to analyze.\n"))
    runlist = []
    i = 0
    while (i < runcount):
        runlist.append(int(input("Please type ONE of the runs you are using. Runs input so far: %f\n" % i)))
        i += 1
    #get an alpha value
    alpha = float(input("Please type an OPPOSITE decimal from 0 to 1 representing the confidence (ex. for a 95 percent CI, type 0.05).\n"))
    zscore = st.norm.ppf(1.0 - (alpha / 2))
    #define the flags through which we will filter people
    flag_vals = []
    flag_vals.append(input("Please enter a minimum age to observe, as an integer. Type -1 if irrelevant.\n"))
    flag_vals.append(input("Please enter a maximum age to observe, as an integer. Type -1 if irrelevant.\n"))
    flag_vals.append(input("Please enter which sex to track as an integer, 0 for males, 1 otherwise. Type -1 if irrelevant.\n"))
    flag_vals.append(input("Please write Vaccinated or Unvaccinated (case-sensitive) to pick a vaccination status at start of run to track. Type -1 if irrelevant.\n"))
    #verify parameters 
    print("You have entered: alpha = %f, min age = %f, max age = %f, sex = %f, vaccination = %s" % (alpha, int(flag_vals[0]), int(flag_vals[1]), int(flag_vals[2]), flag_vals[3]))
    n = 0
    mean = float(0)
    var = float(0)
    #get the sample count and mean
    for run in runlist: 
        data_dir = agt.find_dir(run)
        usable_ids = agt.get_all_ids(data_dir, flag_vals)
        start1 = Node(-1, -1, -1, None)
        build_agent_graph_nodupes(start1, data_dir)
        n += len(usable_ids)
        if (len(usable_ids) > 0):
            mean += (infectivity_mean(start1, usable_ids) * len(usable_ids))
    if (n == 0):
        print("Since we didn't find anyone, we can't run a CI. Returning.\n")
        return 0
    mean = float(mean) / float(n)
    #now that we have our full sample mean, we can get the variance
    for run in runlist:
        data_dir = agt.find_dir(run)
        usable_ids = agt.get_all_ids(data_dir, flag_vals)
        start2 = Node(-1, -1, -1, None)
        build_agent_graph_nodupes(start2, data_dir)
        if (len(usable_ids) > 0):
            var += (infectivity_var(start2, usable_ids, mean) * len(usable_ids))
    var = float(var) / float(n)
    sd = math.sqrt(var)
    rootn = math.sqrt(n)
    hi = mean + (zscore * (sd / rootn))
    lo = mean - (zscore * (sd / rootn))
    print("Given the specified parameters, your CI is [%f,%f]" % (lo, hi))
    outlier_flag = int(input("Type 1 to check if a specific indivdual is an outlier and 0 otherwise.\n"))
    if (outlier_flag > 0):
        #return outlier_check(mean, sd, n, alpha) (outlier check unfinished!)
        return 0
    return 0

#Main

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