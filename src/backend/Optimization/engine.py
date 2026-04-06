#import scheduling algorithms
from algorithms.greedy import greedy_schedule
from algorithms.genetic import genetic_schedule


class SchedulingEngine:
    #engine to select algorithm, apply constraints, and return schedule

    def __init__(self, algorithm_name, constraints):
        #store algorithm name and constraints checker
        self.algorithm_name = algorithm_name
        self.constraints = constraints

        #select the algorithm function
        self.algorithm = self.select_algorithm()

    #choose the scheduling algorithm
    def select_algorithm(self):
        if self.algorithm_name == "greedy":
            return greedy_schedule
        elif self.algorithm_name == "genetic":
            return genetic_schedule
        else:
            #invalid algorithm
            raise ValueError(f"Invalid algorithm: {self.algorithm_name}")

    #run the scheduling process
    def run(self, data):
        #check if data is missing or empty
        if not data:
            raise ValueError("Data is empty or not provided")

        #check required keys exist
        required_keys = ["courses", "rooms", "timeslots", "instructors", "course_instructor_map"]
        if not all(key in data for key in required_keys):
            missing = [key for key in required_keys if key not in data]
            raise ValueError(f"Missing data: {missing}")

        #print which algorithm is used
        print(f"Using {self.algorithm_name} algorithm")

        #run the algorithm with data and constraints
        result = self.algorithm(
            courses=data["courses"],
            rooms=data["rooms"],
            timeslots=data["timeslots"],
            instructors=data["instructors"],
            mapping=data["course_instructor_map"],
            constraints=self.constraints
        )

        print("Schedule completed")

        #return final schedule
        return result