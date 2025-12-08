#graph implementation of agent tracking
#infectivity represented by number of children in the graph

import os
import csv
import math
import scipy.stats as st
import agent_tracker as agt

#Definition of Node class / linked list internal operations

class Node:
    def __init__(self, id: int, pid: int, parent):
        self.id = id
        self.pid = pid
        self.parent = parent
        self.fault = float(0.0)
        self.edges = []

    def addEdge(self, victim, location, time, fault):
        new = Edge(victim, location, time, fault)
        self.edges.append(new)

    def setFault(self, val):
        self.fault = val

    def searchByID(self, head, target: int):
        if (head):
            currentNode = head
            resultNode = head
            if resultNode == None:
                return None
            elif resultNode.id == target:
                return resultNode
            elif currentNode and (len(currentNode.edges) > 0):
                for n in currentNode.edges: 
                    resultNode = self.searchByID(n.victim, target)
                    if resultNode and resultNode.id == target:
                        return resultNode
        return None
    
    def printNodes(self, head):
        print("%d infected by %d at time %d ->" % (head.id, head.pid, head.time))
        for v in head.edges:
            self.printNodes(v.victim)
        return

#Definition of Edge class: stores victim as a node and location as an int

class Edge:
    def __init__(self, victim: Node, location: int, time: int, fault: float):
        self.victim = victim
        self.location = location    
        self.time = time
        self.fault = fault

#harmonic complexity analysis

def is_descendant(st: Node, dst: Node, len: int):
        #check if the current node is the target
        if (st.id == dst.id): 
            return len
        
        #check if any of this node's descendants  
        #lead to the target
        for v in st.edges:
            dfs = is_descendant(v.victim, dst, len + 1)
            if dfs != -1: 
                return dfs
            
        #we can't access the target from this node
        return -1

def dfs(st: Node, visited: set):
    visited.add(st)
    for v in st.edges:
        if v.victim not in visited:
            dfs(v.victim, visited)

def harmonic_complexity(st: Node, trgt: Node):
    count = float(0.0)

    #DFS to grab every node
    visited = set()
    dfs(st, visited)

    # test print to ensure dfs works (it works)
    # for item in visited:
    #     print(item.id, end = ' ')
    # print('\n')

    #find the harmonic complexity of the target
    for v in visited:
        if v.id != trgt.id: 
            dummy = is_descendant(trgt, v, 0)
            if dummy != -1:
                count += float(1 / dummy)

    #divide by normalizing constant
    nminus1 = len(visited) - 1
    count = float(count / nminus1)

    #update node's internal complexity then return
    trgt.setFault(count)
    return count

def calculate_all_harmonic(st: Node):
    visited = set()
    dfs(st, visited)

    #order every single location an infection occurred at by relative weight
    for trgt in visited:
        if trgt.id != -1:
            harmonic_complexity(st, trgt)
            cpx_total = float(0.0)
            for e in trgt.edges:
                cpx_total += harmonic_complexity(st, e.victim)
            for e in trgt.edges:
                if cpx_total > 0.0:
                    e.fault = (float(1 / (2 * len(trgt.edges))) + (((e.victim).fault / cpx_total) / 2)) * trgt.fault
                else:
                    e.fault = trgt.fault

    return 0

def check_sse(st: Node):
    visited = set()
    dfs(st, visited)

    #order every single node in the graph except start by harmonic complexity
    targets = list()
    for trgt in visited:
        if trgt.id != -1:
            pair = (trgt.id, harmonic_complexity(st, trgt))
            targets.append(pair)
    targets.sort(key=lambda x: x[1], reverse=True)

    #check if top (< 20%) of agents account for at least 50% of infections
    total_hc = float(0.0)
    total_elements = 0
    curr = 0
    top20 = float(0.2 * len(targets))
    while total_hc < 0.5:
        total_hc += targets[curr][1]
        curr += 1

    if curr < top20:
        print("Superspreader event detected! Potential superspreaders are: ")
        for i in range(0, curr + 1): 
            print(targets[i][0], end = ' ')
        print('\n')
        return True

    print("This run was not a superspreader event.\n")
    return False

def location_impact(st: Node):
    visited = set()
    dfs(st, visited)
    calculate_all_harmonic(st)

    #order every single location an infection occurred at by relative weight
    weighted_locs = list()
    for trgt in visited:
        if trgt.id != -1:
            harmonic_complexity(st, trgt)
            for e in trgt.edges:
                if any(e.location == x[0] for x in weighted_locs):
                    for y in weighted_locs:
                        if e.location == y[0]:
                            y[1] += e.fault
                            break 
                else: 
                    pair = [e.location, e.fault]
                    weighted_locs.append(pair)
    weighted_locs.sort(key=lambda x: x[1], reverse=True)

    count = min(len(weighted_locs), 5)
    print("Top %d locations by relative weight: " % count)
    for i in range(0, count):
        print(weighted_locs[i][0], end = ' ')
    print('\n')

    return 0

#Test functions for harmonic complexity analysis

def build_harmonic_test_graph(): 
    #handmake test graph
    start = Node(-1,-1,None)
    node_a = Node(1,-1,start)
    node_b = Node(2,-1,start)
    node_c = Node(3,-1,start)
    start.addEdge(node_a, -1, 0, 0)
    start.addEdge(node_b, -1, 0, 0)
    start.addEdge(node_c, -1, 0, 0)
    node_d = Node(4,1,node_a)
    node_e = Node(5,1,node_a)
    node_a.addEdge(node_d, 1, 1, 0)
    node_a.addEdge(node_e, 2, 1, 0)
    node_f = Node(6,5,node_e)
    node_e.addEdge(node_f, 2, 2, 0)
    node_g = Node(7,3,node_c)
    node_c.addEdge(node_g, 2, 1, 0)
    node_h = Node(8,7,node_c)
    node_g.addEdge(node_h, 3, 2, 0)
    #graph layout should be
    #        Start
    #       /  |  \
    #      A   B   C
    #    1 2        2
    #   D  E         G
    #       2       3 
    #        F     H
    return start

def test_descendant():
    #test is_descendant
    start = build_harmonic_test_graph()
    node_a = start.searchByID(start, 1)
    node_e = start.searchByID(start, 5)
    node_f = start.searchByID(start, 6)
    node_c = start.searchByID(start, 3)
    node_b = start.searchByID(start, 2)
    node_d = start.searchByID(start, 4)
    node_g = start.searchByID(start, 7)
    node_h = start.searchByID(start, 8)
    print(len(node_a.edges))
    dec_a = is_descendant(node_a, node_a, 0)
    dec_b = is_descendant(node_a, node_e, 0)
    dec_c = is_descendant(node_a, node_f, 0)
    dec_d = is_descendant(start, node_f, 0)
    dec_e = is_descendant(node_a, node_c, 0)
    print("Expected: 0, 1, 2, 3, -1\n")
    print("Got: %d, %d, %d, %d, %d\n" % (dec_a, dec_b, dec_c, dec_d, dec_e))

    #test harmonic_complexity
    calculate_all_harmonic(start)

    visited = set()
    dfs(start, visited)

    print("vertex fault: \n")
    for item in visited:
        if item.id != -1:
            print("%d: %f " % (item.id, item.fault))
            if len(item.edges) > 0:
                print("with edge weights ")
            for e in item.edges:
                print("%f for edge to %d" % (e.fault, (e.victim).id))
            print("\n")


    #test SSE checker
    check_sse(start)

    #test location impact checker
    location_impact(start)
    return 0
    
#Test functions for basic graph functionality
    
def graph_test_1():
    run = int(input("Please type the run to test on.\n"))
    data_dir = agt.find_dir(run)
    print("without dupes at all\n")
    start3 = Node(-1, -1, None)
    build_agent_graph_nodupes(start3, data_dir)
    start3.printNodes(start3)
    print("\n")
    return 0

def graph_test_2():
    run = int(input("Please type the run to test on.\n"))
    data_dir = agt.find_dir(run)
    start2 = Node(-1,-1,None)
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
    start2 = Node(-1,-1,None)
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
                    new_node = Node(int(row[1]), -1, start)
                    start.addEdge(new_node, int(row[11]), 0, 0)
                else:
                    infector = start.searchByID(start, int(row[6]))
                    infected = start.searchByID(start, int(row[1]))
                    if not infected:
                        new_node = Node(int(row[1]), int(row[6]), infector)
                        infector.addEdge(new_node, int(row[11]), int(row[0]), 0)
    return 0

def direct_infectivity(head: Node, target: int):
    count = 0
    target_node = head.searchByID(head, target)
    if target_node:
        for v in target_node.edges:
            count += 1
    return count

def total_infectivity_nodupes(head: Node, target_int: int): 
    count = 0
    target = head.searchByID(head, target_int)
    if not target:
        return 0
    count += direct_infectivity(head, target_int)
    if len(target.edges) > 0:
        for v in target.edges:
            vic = v.victim
            count += total_infectivity_nodupes(head, vic.id)
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
        start1 = Node(-1, -1, None)
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
        start2 = Node(-1, -1, None)
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
    start1 = Node(-1, -1, None)
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
    start1 = Node(-1, -1, None)
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
        start = Node(-1,-1,None)
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
    if len(target.edges) > 0:
        victim_id_list = []
        for v in target.edges:
            vic = v.victim
            victim_id_list.append(int(vic.id))
        for v in target.edges:
            vic = v.victim
            #find likelihood this victim would've been infected anyways
            count += reverse_estimator(vic.id, superspreader_int, iqu, data_dir, victim_id_list)
            #then apply the same logic to all of their victims etc
            count += superspreader(head, superspreader_int, vic.id, iqu, data_dir)
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
    start1 = Node(-1, -1, None)
    build_agent_graph_nodupes(start1, data_dir)
    spread_count = superspreader(start1, person, person, iqu, data_dir)
    if (total_infectivity_nodupes(start1, person) != 0):
        spread_count /= total_infectivity_nodupes(start1, person)
    print("Our estimated superspreader probability is %f.\n" % spread_count)
    

#Main

def main():
    test_descendant()
    return 1
    # code for testing stat analysis functions
    # start_flag = int(input("For testing graph features, type 0. For CI, type 1. For superspreader check, type 2.\n"))
    # if (start_flag == 0):
    #     testnum = int(input("Please select which test to run.\n"))
    #     if (testnum == 1):
    #         graph_test_1()
    #     elif (testnum == 2):
    #         graph_test_2()
    #     elif (testnum == 3):
    #         mean_var_test()
    #     else: 
    #        print("Not implemented!\n")
    # if (start_flag == 1):
    #     infectivity_ci_multi()
    # if (start_flag == 2):
    #     run_superspreader_check()
    # else:
    #     print("Not implemented!\n")
    # return 1
    
if __name__ == "__main__":
    main()