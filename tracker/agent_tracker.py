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
    last_str = "In total, your agent moved " + str(total_movements) + " times,\n"
    last_str += "directly infected " + str(total_infections) + " people,\n"
    last_str += "and caught " + str(total_catches) + " disease(s)."
    output_file.write(last_str)
    output_file.close()
    return 0

def location_track(data_dir: str):
    print("NOT YET IMPLEMENTED\n")
    return 0

def main():
    data_dir = find_dir(input("Please type what run you want to analyze (int).\n"))
    check_num = input("Please type 0 to track an agent or 1 to track a location.\n")
    if int(check_num) == 0:
        agent_track(data_dir)
    elif int(check_num) == 1:
        location_track(data_dir)
    else:
        print("Invalid number!\n")

    
    

if __name__ == "__main__":
    main()