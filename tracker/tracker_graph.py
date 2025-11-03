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

def mean_var_test():
    run = int(input("Please type the run to test on.\n"))
    data_dir = agt.find_dir(run)
    start2 = Node(-1,-1,-1,None)
    build_agent_graph_nodupes(start2, data_dir)
    id_fifteen = [15]
    print("mean 15: %f\n" % infectivity_mean(start2, id_fifteen))
    print("var 15: %f\n" % infectivity_var(start2, id_fifteen, infectivity_mean(start2, id_fifteen)))
    flag_vals = [30, 99, 1, -1]
    usable_ids = agt.get_all_ids(data_dir, flag_vals)
    print("mean run1: %f\n" % infectivity_mean(start2, usable_ids))
    print("var run1: %f\n" % infectivity_var(start2, usable_ids, infectivity_mean(start2, usable_ids)))
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
        added = total_infectivity_nodupes(head, int(target_id))
        mean += added
    mean = float(mean) / float(num_people)
    return mean

def infectivity_var(head: Node, usable_ids: list[int], mean: float):
    var = float(0.0)
    num_people = len(usable_ids)
    for target_id in usable_ids:
        var += pow(total_infectivity_nodupes(head, int(target_id)) - mean, 2)
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
    outlier_flag = int(input("Type 1 to check if a specific indivdual is an outlier, 2 to check if a specific run is an outlier, and 0 otherwise.\n"))
    if (outlier_flag == 1):
        return person_outlier_check(mean, sd, n, alpha)
    if (outlier_flag == 2):
        return run_outlier_check(mean, sd, alpha, flag_vals)
    return 0

def person_outlier_check(ci_min: float, ci_max: float):
    run = int(input("Please type the run your outlier is located in.\n"))
    person = int(input("Please type the ID of the person to analyze.\n"))
    data_dir = agt.find_dir(run)
    start1 = Node(-1, -1, -1, None)
    build_agent_graph_nodupes(start1, data_dir)
    val = total_infectivity_nodupes(start1, int(person))
    if (val > ci_max) or (val < ci_min):
        print("This individual is an outlier as per your CI.")
    else: 
        print("This person falls within the bounds of your CI.")
    return 0

def run_outlier_check(mean: float, sd: float, alpha: float, flag_vals: list[str]):
    run = int(input("Please type the run you want to check.\n"))
    data_dir = agt.find_dir(run)
    usable_ids = agt.get_all_ids(data_dir, flag_vals)
    n = len(usable_ids)
    benchmark = st.t.ppf(1 - (alpha / 2), n - 1)
    start1 = Node(-1, -1, -1, None)
    build_agent_graph_nodupes(start1, data_dir)
    #print("directory %s, usable_id %f\n" % (data_dir, usable_ids[0]))
    sample_mean = infectivity_mean(start1, usable_ids)
    #print("Your individual infected %f people.\n" % (val))
    test_stat = ((sample_mean - mean)/(sd / math.sqrt(n)))
    if (math.fabs(test_stat) > benchmark):
        print("This run is an outlier among the runs included in your study.\n")
    else:
        print("We have failed to find that this run is a significant outlier among runs included in your study.\n")
    print("The test statistic is %f relative to a benchmark of %f.\n" % (math.fabs(test_stat), benchmark))

#"Wonderful Life" analysis method for superspreaders

#naive baseline implementation of infectivity quotient: just take the mean of EVERYONE in the sample
def infectivity_quotient(runs: list[int]):
    #get the mean of infectivity means of all people across all runs
    iqu = float(0.0)
    for r in runs:
        data_dir = agt.find_dir(r)
        start = Node(-1,-1,-1,None)
        build_agent_graph_nodupes(start, data_dir)
        no_flags = [-1, -1, 1, -1]
        usable_ids = agt.get_all_ids(data_dir, no_flags)
        iqu += float(infectivity_mean(start, usable_ids))
    iqu /= float(len(runs))
    #normalize onto the continuous [0,1)
    if iqu > 1: 
        normalizer = int(iqu) + 1
        iqu /= float(normalizer)
    return iqu

#given iqu and the run, calculate likelihood someone would've been infected without their actual infector existing
def reverse_estimator(victim_int: int, infector_int: int, iqu: float, data_dir: str, victims: list[int]):
    final = float(1.0)
    prob = float(1.0)
    #use contact logs to see how many times our victim runs into a DIFFERENT infected person
    with open(os.path.join(data_dir, "contact_logs.csv"), mode="r") as ifile: 
        ctable = csv.reader(ifile)
        for row in ctable:
            if (str(row[0]) != "timestep"):
                our_victim_position = 0
                other_person_position = 0
                other_person_infectious = 0
                if (int(row[1]) == victim_int):
                    our_victim_position = 1
                if (int(row[2]) == victim_int):
                    our_victim_position = 2
                if (our_victim_position > 0):
                    other_person_position = (our_victim_position % 2) + 1
                    other_person_infectious = other_person_position + 5
                    if (bool(row[other_person_infectious]) == True):
                        if (int(row[other_person_position]) != infector_int) and (int(row[other_person_position]) not in victims):
                            prob *= float(float(1.0) - iqu)
    final = float(1.0 - prob)
    return final

#calculation for if any one individual in a sample is a superspreader
def superspreader(head: Node, superspreader_int: int, curr_int: int, iqu: float, data_dir: str):
    count = float(0.0)
    target = head.searchByID(head, curr_int)
    if not target:
        return count
    if len(target.victims) > 0:
        victim_id_list = []
        for v in target.victims:
            victim_id_list.append(int(v.id))
        for v in target.victims:
            #find likelihood this victim would've been infected anyways
            count += reverse_estimator(v.id, superspreader_int, iqu, data_dir, victim_id_list)
            #then apply the same logic to all of their victims etc
            count += superspreader(head, superspreader_int, v.id, iqu, data_dir)
    #result is on continuous [0,n) where n is the number of victims
    return count

def run_superspreader_check():
    runcount = int(input("Please type how many runs you would like to analyze to derive iqu.\n"))
    runlist = []
    i = 0
    while (i < runcount):
        runlist.append(int(input("Please type ONE of the runs you are using. Runs input so far: %f\n" % i)))
        i += 1
    iqu = infectivity_quotient(runlist)
    print("Given the selected runs, your infectivity quotient estimate is %f.\n" % iqu)
    target_run = int(input("Please type the run your suspected superspreader is in.\n"))
    person = int(input("Please type the ID of the person to analyze.\n"))
    data_dir = agt.find_dir(target_run)
    start1 = Node(-1, -1, -1, None)
    build_agent_graph_nodupes(start1, data_dir)
    spread_count = superspreader(start1, person, person, iqu, data_dir)
    spread_count /= total_infectivity_nodupes(start1, person)
    print("Our estimated superspreader probability is %f.\n" % spread_count)
    

#Main

def main():
    start_flag = int(input("For testing graph features, type 0. For CI, type 1. For superspreader check, type 2.\n"))
    if (start_flag == 0):
        testnum = int(input("Please select which test to run.\n"))
        if (testnum == 1):
            graph_test_1()
        elif (testnum == 2):
            graph_test_2()
        elif (testnum == 3):
            mean_var_test()
        else: 
           print("Not implemented!\n")
    if (start_flag == 1):
        infectivity_ci_multi()
    if (start_flag == 2):
        run_superspreader_check()
    else:
        print("Not implemented!\n")
    return 1
    
if __name__ == "__main__":
    main()