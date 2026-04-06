#import scheduling algorithms
from algorithms.greedy import greedy_schedule
from algorithms.genetic import genetic_schedule


class SchedulingEngine:
    def __init__(self, algorithm_name, constraints):
        self.algorithm_name = algorithm_name
        self.constraints = constraints
        self.algorithm = self.select_algorithm()

    #choose the scheduling algorithm
    def select_algorithm(self):
        if self.algorithm_name == "greedy":
            return greedy_schedule
        elif self.algorithm_name == "genetic":
            return genetic_schedule
        else:
            raise ValueError("Invalid algorithm")

    #run the scheduling process
    def run(self, data):

        #check all required data is provided
        if not data:
            raise ValueError("No data provided")

        if "courses" not in data:
            raise ValueError("Courses missing")

        if "rooms" not in data:
            raise ValueError("Rooms missing")

        if "timeslots" not in data:
            raise ValueError("Timeslots missing")

        if "instructors" not in data:
            raise ValueError("Instructors missing")

        if "course_instructor_map" not in data:
            raise ValueError("Course-instructor map missing")

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