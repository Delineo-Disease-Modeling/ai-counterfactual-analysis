#agent tracker
#using the CSV files output by ai-counterfactual-analysis,
#generates "audits" for individual people or locations

import os
import csv

def find_dir(run: int):
    run_name = "run"
    run_name += str(run)
    data_dir = os.getcwd() + "/data/raw/" + run_name
    return data_dir

def agent_track(data_dir: str):
    output_file = open(os.path.join(data_dir, "agent_track_output.txt"), "w")
    track_id = int(input("Please enter the ID of the agent you want to track (int).\n"))
    timestamp = 0
    total_movements = 0
    total_infections = 0
    total_catches = 0
    output_file.write("Tracking of agent " + str(track_id) + "\n")
    while (timestamp <= 1440):
        moved_flag = 0
        infected_flag = 0
        victim_flag = 0
        with open(os.path.join(data_dir, "movement_logs.csv"), mode="r") as move_file:
            move_table = csv.reader(move_file)
            for row in move_table:
                if str(row[0]) == str(timestamp) and str(row[1]) == str(track_id):
                    total_movements += 1
                    moved_flag = 1
                    move_str = "At time " + str(timestamp) + ", your agent moved from "
                    move_str += str(row[3]) + " " + str(row[2]) + " to "
                    move_str += str(row[5]) + " " + str(row[4]) + ".\n"
                    output_file.write(move_str)
            if moved_flag == 0: 
                move_str = "Your agent didn't move at time " + str(timestamp) + ".\n"
                output_file.write(move_str)
        with open(os.path.join(data_dir, "infection_logs.csv"), mode="r") as infect_file:
            infect_table = csv.reader(infect_file)
            for row in infect_table: 
                if str(row[0]) == str(timestamp) and str(row[6]) == str(track_id):
                    total_infections += 1
                    infected_flag = 1
                    infect_str = "At time " + str(timestamp) + ", your agent infected agent "
                    infect_str += str(row[1]) + " with disease " + str(row[15]) + ".\n"
                    output_file.write(infect_str)
                if str(row[0]) == str(timestamp) and str(row[1]) == str(track_id):
                    total_catches += 1
                    victim_flag = 1
                    victim_str = "At time " + str(timestamp) + ", your agent was infected "
                    victim_str += "with " + str(row[15])
                    if timestamp == 0:
                        victim_str += " at the start of the experiment.\n"
                    else: 
                        victim_str += " by agent " + str(row[6]) + ".\n"
                    output_file.write(victim_str)
            if infected_flag == 0: 
                infect_str = "Your agent didn't infect anyone at time " + str(timestamp) + ".\n"
                output_file.write(infect_str)
            if victim_flag == 0:
                victim_str = "Your agent did not catch any new diseases at time " + str(timestamp) + ".\n"
                output_file.write(victim_str)
        timestamp += 60
    last_str = "In total, your agent moved " + str(total_movements) + " time(s),\n"
    last_str += "directly infected " + str(total_infections) + " person(s),\n"
    last_str += "and caught " + str(total_catches) + " disease(s)."
    output_file.write(last_str)
    output_file.close()
    return 0

def location_track(data_dir: str):
    output_file = open(os.path.join(data_dir, "location_track_output.txt"), "w")
    track_id = int(input("Please enter the ID of the location you want to track (int).\n"))
    timestamp = 0
    total_visits = 0
    total_exits = 0
    total_infections = 0
    intro_write_flag = 0
    with open(os.path.join(data_dir, "location_logs.csv"), mode="r") as intro_file: 
        intro_table = csv.reader(intro_file)
        for row in intro_table:
            if intro_write_flag == 0 and str(row[1]) == str(track_id): 
                output_file.write("Tracking of " + str(row[2]) + " " + str(row[1]) + "\n")
                intro_write_flag = 1
    while (timestamp <= 1440):
        visit_flag = 0
        exit_flag = 0
        catch_flag = 0
        with open(os.path.join(data_dir, "movement_logs.csv"), mode="r") as move_file:
            move_table = csv.reader(move_file)
            for row in move_table:
                if str(row[0]) == str(timestamp) and str(row[4]) == str(track_id):
                    total_visits += 1
                    visit_flag = 1
                    move_str = "At time " + str(timestamp) + ", agent "
                    move_str += str(row[1]) + " entered this location.\n"
                    output_file.write(move_str)
                if str(row[0]) == str(timestamp) and str(row[2]) == str(track_id): 
                    total_exits += 1
                    exit_flag = 1
                    move_str = "At time " + str(timestamp) + ", agent "
                    move_str += str(row[1]) + " exited this location.\n"
                    output_file.write(move_str)
            if visit_flag == 0: 
                move_str = "Nobody entered this location at time " + str(timestamp) + ".\n"
                output_file.write(move_str)
            if exit_flag == 0:
                move_str = "Nobody exited this location at time " + str(timestamp) + ".\n"
                output_file.write(move_str)
        with open(os.path.join(data_dir, "infection_logs.csv"), mode="r") as infect_file:
            infect_table = csv.reader(infect_file)
            for row in infect_table: 
                if str(row[0]) == str(timestamp) and str(row[11]) == str(track_id):
                    total_infections += 1
                    catch_flag = 1
                    infect_str = "At time " + str(timestamp) + ", agent " + str(row[6]) + " infected "
                    infect_str += str(row[1]) + " with disease " + str(row[15]) + "at your location.\n"
                    output_file.write(infect_str)
            if catch_flag == 0: 
                infect_str = "Your agent didn't infect anyone at time " + str(timestamp) + ".\n"
                output_file.write(infect_str)
        timestamp += 60
    last_str = "In total, agents went to this location " + str(total_visits) + " time(s),\n"
    last_str += "left the location " + str(total_exits) + " time(s),\n"
    last_str += "and infected people here " + str(total_infections) + " time(s)."
    output_file.write(last_str)
    output_file.close()
    return 0

def deadliest_agent(data_dir: str):
    deadliest_id = 0
    curr_max = 0
    deadliest_dict = dict({0: 0})
    with open(os.path.join(data_dir, "infection_logs.csv"), mode="r") as ifile: 
        itable = csv.reader(ifile)
        for row in itable:
            if str(row[6]) != "infector_person_id" and str(row[6]) != "":
                if deadliest_dict.get(row[6]) == None: 
                    deadliest_dict[row[6]] = 1
                else: 
                    deadliest_dict[row[6]] += 1
                if deadliest_dict[row[6]] > curr_max: 
                    deadliest_id = int(row[6])
                    curr_max = deadliest_dict[row[6]]

    print("The deadliest agent is the one with ID " + str(deadliest_id) + ".\n") 
    return 0

def deadliest_location(data_dir: str):
    deadliest_id = 0
    curr_max = 0
    deadliest_dict = dict({0: 0})
    with open(os.path.join(data_dir, "infection_logs.csv"), mode="r") as ifile: 
        itable = csv.reader(ifile)
        for row in itable:
            if str(row[11]) != "infection_location_id" and str(row[11]) != "":
                if deadliest_dict.get(row[11]) == None: 
                    deadliest_dict[row[11]] = 1
                else: 
                    deadliest_dict[row[11]] += 1
                if deadliest_dict[row[11]] > curr_max: 
                    deadliest_id = int(row[11])
                    curr_max = deadliest_dict[row[11]]

    print("The deadliest location is the one with ID " + str(deadliest_id) + ".\n") 
    return 0

def main():
    data_dir = find_dir(input("Please type what run you want to analyze (int).\n"))
    check_num = input("Please type 0 to track an agent or 1 to track a location. To find deadliest agent, type 2. To find deadliest location, type 3.\n")
    if int(check_num) == 0:
        agent_track(data_dir)
    if int(check_num) == 1:
        location_track(data_dir)
    if int(check_num) == 2:
        deadliest_agent(data_dir)
    if int(check_num) == 3:
        deadliest_location(data_dir)
    if int(check_num) > 3:
        print("Invalid number!\n")

    
    

if __name__ == "__main__":
    main()