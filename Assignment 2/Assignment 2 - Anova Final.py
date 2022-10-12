#######################################################################################################################
# pip install --upgrade --user ortools    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Please install user ortools using the code above 
#
# WELCOME TO ASSIGNMENT 1: LAST MILE DELIVERY WITH ELECTRIC VANS SIMULATION 
# !!! PLEASE ENLARGE YOUR SCREEN TO BE ABLE TO SEE ALL THE COMMENTS AND EXPLANATIONS
# GO TO "PARAMETERS" SECTION TO TRY DIFFERENT SETTINGS. 
   


# GROUP MEMBERS:
# Almir Gungor 10752670
# Berk Ceyhan 10761821
# Cagatay Onur 10781315
# Lorenzo Ghedini 10586137



#######################################################################################################################

# CONTENTS: (NEW: Introduced by us /// OLD: Base Case version //// OLD+NEW: Both of them together)

# 1. Summary 
# 2. Parameters
#    2.1 Parameters for Routing
#    2.2 Old Parameters (OLD)
#    2.3 Charging policy parameter (NEW)  
#    2.4 Stochastic speed parameter: 0: deterministic / 1: stochastic with avg_speed=30,stdev=2
# 3. Importing the libraries (OLD+NEW)
# 4. Routing of the vans for 60 customers and multiple vans (NEW)
#    4.1 "create_data_model" function (NEW)
#    4.2 "print_solution" function (NEW)
#    4.3 "get_routes" function (NEW)
#    4.4 "main" function (NEW)
#    4.4.1 "routing manager and routing model" (NEW)  
#    4.4.2 "distance_callback" function (NEW)    
# 5. New functions about the new charging station selection policy (NEW)
#    5.1 "extra_distance_time" function (NEW)
#    5.2 "expected_waiting_time" function (NEW)
#    5.3 "charging_station_score" function (NEW)
#    5.4 "best_station_id" function (NEW)
#    5.5  "best_station_time" function (NEW) 
#    5.6  "needed_energy" function (NEW) 
# 6. Other functions  
#    6.1 Old functions (OLD)
#    6.2 Old functions modified to support stochastic speed (NEW) 
# 7. Creation of arrays for the following sections
#    7.1 Routing visualization empty arrays creation (NEW)
#    7.2 Taking the customers and charging stations into arrays 
# 8. class Van
#    8.1 "__init__" function (OLD+NEW)
#       8.1.1 Old attributes (OLD)
#       8.1.2 New and modified attributes (NEW)
#    8.2 "get_routes" function (NEW)
#    8.3 "check_charge" function (OLD + NEW)
#       8.3.1  New charging station policy (NEW)
#           8.3.1.1 Battery about to die (NEW)
#           8.3.2.2 Early Charging (NEW)
#       8.3.2  Base case charging policy (OLD)
#    8.4 "move" function (OLD+NEW)
#    8.5 "charging" function (OLD+NEW)
#       8.5.1  New charging amount policy (NEW)
#       8.5.2  Base case charging policy (OLD)
# 9.  class Charging Station (OLD)
# 10. class GreenModel (OLD)
# 11. Running the simulation (OLD)
# 12. Routing visualization (NEW) 


#######################################################################################################################
                                                                        # 1.SUMMARY
# 
#       The code is divided into sections to make the reading process easier for you. Moreover, very detailed explanations are made for  
# in each section to explain newly introduced parts.
#       Our effort in this project focused in 3 parts:    
#           1. Routing of the vans
#           2. Introducing a new charging policy
#           3. Changing the speed to the stochastic speed
#
# Thanks to the routing of the vans, the model is adapted to multiple vans trying to optimize their routes. For 8 hours tour period and 35kwh battery size
# 4 vans are enough to complete the tour, even without charging. Thats why, to make the problem more challenging in order to force the vans to charge,
# the battery size can be changed. You can check new codes in "3. Routing of the vans for 60 customers and multiple vans" part.
# 
# Secondly, a new charging policy is introduced by us. Having detected the drawbacks of the current policy (e.g. going to the closest charging station
# is not necessarily the best choice if it has a very long queue), we introduced a new charging policy, to improve the performance of the last
# mile delivery. You can check new codes in "4. New functions about the new charging station selection policy"
#
# Also, the battery sizes are changed to support giving different battery sizes to different vans, by introducing
# battery size array as battery_size=[x,y,z,t] which shows the battery sizes of vans from 0 to 3
#
# lastly, the van capacities are introduced, depending on the number of customers served and average order weight. Tn this
# way the current load weights of the vans can be seen, which decreases after visiting customers. This is not printed
# due to the space constraint of the python console;however, you can go to the related section and can print it.
#
# To code these 5 parts, both new functions are introduced, and also some existing functions are modified. All the changes will be explained in detail
# in the related parts of the code.
# 
# 
# Additionally, in this script single factor experiments will be conducted using oneway anova test. 
# Experiment will be done by fixing one factor and changing level for other factor
# From Anova test result, the signifiance of factors on the system performance measure which is ax tour end times are observed.
# 
# 
# 

########################################################################################################################################
#                                                                   2. PARAMETERS
# 2.1. Parameters for Routing
num_customers = 60
num_van = 4 # !!!!! 4 is the optimal value, which is reached in the routing part. 
            # if you want to change it, use num_van<=4

# 2.2 Old Parameters
avg_speed = 30 #km/h - Average speed
warehouse_x = 35 # do NOT change this parameter
warehouse_y = 35 # do NOT change this parameter
num_stations = 130 # do NOT change this parameter
battery_size = [30,30,20,20] # you can play with this parameter, please set factor levels mentioned in the presentation


# 2.3. Charging policy parameters 
#      This will be used in the Assignment 2 as an independent variable.
#      charging_policy=1 : NEW charging policy
#      charging_policy=0 : Old (Base Case) charging policy 
#      
#      Thanks to this parameter, using "if" conditions, one of these 2 charging policies will be implemented
#      in the code. You can see this in the "check_charge" part of the code.
charging_policy=1 # set 0 or 1 
safety_energy_level=1.15 # extra battery amount that is charged for safety purpose

# 2.4. Stochastic speed parameters
stochastic_speed=1  # 0: deterministic / 1: stochastic
avg_speed = 30 #km/h - Average speed
stdev_speed=2 # standard deviation of speed

# 2.5.Statistics
alpha=0.05 # Type 1 error probability
N=100       # Number of runs, normally 800, here it is set to 100 to speed up the computation for you
            # ╩you can change it to 800 and run it if you would like to

# 2.6 Anova parameter
#Assign anova_num as 0 to fix charging policy and conduct experiment on varying battery sizes
#ATTENTION: if anova_num = 0, please set charging policy as 0 for Experiment 1 and set charging policy as 1 for Experiment 2
#Assign anova_num as 1 to fix battery sizes and conduct experiment on varying charging policies
#ATTENTION: if anova_num = 1, please set appropriate battery size for Experiment 3,4,5,6,7,8
anova_num=0
#######################################################################################################################
                                                                # 3.IMPORTING THE LIBRARIES
                                                
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2 
from ortools.constraint_solver import pywrapcp
from scipy.spatial.distance import pdist, squareform
import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation, BaseScheduler 
import random
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.stats import f_oneway
#######################################################################################################################
                                                                  # 4.ROUTING OF THE VANS
                                                

# !!!!!!!! THE NEXT FUNCTIONS are used in order to determine the routing of the Vans
# to choose the most convenient routing in order to save time. 

# 4.1. "create_data_model":    
# Here we create the matrix in which all the travel distance between the costumers and 
# warehouse will be included as well as the number of customers, the number of vans and the starting point for the routing
# the warehouse.  
def create_data_model():
    """Stores the data for the problem."""    
    data={}
    aaa= pd.read_excel("customer_location_"+str(num_customers)+".xlsx")
    distances = pdist(aaa.values, metric='cityblock')
    data['distance_matrix'] = squareform(distances) ### This part contributes to the 
                                                    ### creation of the distance matrix
    data['num_vehicles'] = num_van
    data['depot'] = num_customers
    return data ### The data consists of distance_matrix, so an array of distances between locations on meters.
                ### num_vehicles: The number of vehicles in the fleet.
                ### depot: The index of the depot. The location where all vehicles start and end their routes. 
                          #It's equal to num_cust because the warehouse is added on the last location in the dataset

Tot_distance=[]

# 4.2. "print_solution": 
# This code is responsible for the display of the routes and the computation 
# of the total distance travelled by the vans.
def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    #print(f'Objective: {solution.ObjectiveValue()}')
    max_route_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)   ### This is the starting point for the 
                                            ### creation of the routing
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(   ### This is the part in which we define the cost 
                previous_index, index, vehicle_id)            ### for the distance travelled 
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        plan_output += 'Distance of the route: {}km\n'.format(route_distance)
#        print(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
        Tot_distance.append(route_distance) ### This term is necessary to store the 
                                            ### maximum distance travelled by each van
#    print('Maximum of the route distances: {}km'.format(max_route_distance))
    

# 4.3. "get_routes": 
# This code is utilized to save the routes to a list or array. 
# This has the advantage of making the routes available for the successive parts
def get_routes(solution, routing, manager):
  """Get vehicle routes from a solution and store them in an array."""
  ### Get vehicle routes and store them in a two dimensional array whose
  ### i,j entry is the jth location visited by vehicle i along its route.
  routes = []
  for route_nbr in range(routing.vehicles()):
    index = routing.Start(route_nbr)
    route = [manager.IndexToNode(index)]
    while not routing.IsEnd(index):
      index = solution.Value(routing.NextVar(index))
      route.append(manager.IndexToNode(index))
    routes.append(route)
  return routes


# 4.4. "main": 
# This is the core of the routing part, as a matter of fact in this section
# it is created the index manager and the routing model. In order to get the routing we use the "RoutingIndexManager"
# function to convert the solver's internal indices to the number of locations
def main():
    """Entry point of the program."""
    ### Instantiate the data problem.
    data = create_data_model()

    # 4.4.1 "routing manager and routing model"
    # It aims to simplify variable index usage. 
    # The method manager.IndexToNode converts the solver's internal indices (which you can safely ignore) to the numbers for locations. 
    # manager.IndexToNode will be used in the following function, distance_call back
    # Location numbers correspond to the indices for the distance matrix.
    
    ### Create the routing index manager.     
    # The inputs to RoutingIndexManager are:
    # The number of rows of the distance matrix, which is the number of locations (including the depot).
    # The number of vehicles in the problem.
    # The node corresponding to the depot.        
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    ### Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)
 

    # 4.4.2 "distance call-back": 
    # which is necessary in order to use the routing solver, as a matter of fact
    # this is a function that takes any pair of locations and returns the distance between them. The easiest way to do this 
    # is using the distance matrix we defined before. In this part is also set the arc costs, which define the cost of travel, 
    # to be the distances of the arcs. In this way we can define the best routing possible by first creating a distance dimension
    # which computes the cumulative distance traveled by each vehicle along its route. Then we set a cost proportional to the 
    # maximum of the total distances along each route. In this part we set a large coefficient (200) for the global span of the 
    # routes, which is the maximum of the distances of the routes. This makes the global span the predominant factor in the 
    # objective function, so the program minimizes the length of the longest route.    
    ### Create and register a transit callback.
    def distance_callback(from_index, to_index):  ### The following function creates the distance callback, 
                                                  ### which returns the distances between locations, 
                                                  ### and passes it to the solver
        ### Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    ### Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index) ### This sets the arc(edge) costs, which defines the cost of travel
                                                                     ### as the distances of the arcs.

    # Distanca constaint is necessary to solve the problem. 
    # Cost will be defined and it will be proportional to the maximum of the total distances.
    ### Add Distance constraint.    
    dimension_name = 'Distance'   ### The routing solver uses an object called a dimension to keep track of quantities 
                                  ### that accumulate along a vehicle's route
    routing.AddDimension(
        transit_callback_index,
        0,  ### no slack, so no waiting times at the locations
        3000,  ### Maximum for the total quantity accumulated along each route, we put it 
                ### to a value that is sufficiently large to impose no restrictions on the routes
        True,  ### start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(200)  ### Allowable maximum difference of the maximum and minimum travelled distances 
                                                          ### A large coefficient 200 is chosen as a limit for better results with more freedom
                                                          ### Span cost makes the global span the predominant factor in the objective function, 
                                                          ### so the program minimizes the length of the longest route.


    ### To set the default search parameters and a heuristic method for finding the first solution
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC) ### The code sets the first solution strategy to PATH_CHEAPEST_ARC, which  
                                                                   ### creates an initial route for the solver by repeatedly adding edges with 
                                                                   ### the least weight that doesn't lead to a previously visited node 
                                                                   ### (other than the depot).
    ### Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    
    ### Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution)
        routes = get_routes(solution, routing, manager)
        
        ### The following for cycle can also be used for displaying the routes
        ### Display the routes.
        # for i, route in enumerate(routes):
        #     print('Route', i, route)
    else:
        print('No solution found !')       
    return routes        

if __name__ == '__main__':
    van_routes = main()

for i in range(num_van):
    van_routes[i].pop(0)


# !!! NOTE: In this part we determine the best possible routing in order to visit all the costumers in the given
# time. To have the best routing possible we need just 4 vans and no more since this is the number of vehicles 
# for which we can obtain the minimum longest distance of the route. As a matter of fact if we introduce one or 
# more additional vans, the program will make one of them remain at the warehouse since it is not possible to
# optimize the routing more because by introducing one or more vans the longest route percurred by a van, in 
# terms of Km, will be the same or greater.




########################################################################################################################

                                        # 5. NEW FUNCTIONS ABOUT THE NEW CHARGING STATION SELECTION POLICY       


# Our new charging station selection decision making policy improves the performance by two ways:
    
#   1. Choosing THE BEST CHARGING STATION: Choosing the CLOSEST CHARGING STATION is NOT necessarily the best option. 
#   In fact, if a very long queue is accumulated at that specific charging station, even if it is the closest one, 
#   our van will lose huge amount time. So, it will help us choosing the best option, not necessarily the closest one
#   by taking into account:
    
#       1.a. Extra travelling time compared to the original route 
#            (If the charging station is located in between the van and the next customer, this value is 0)
#
#       1.b. Expected waiting time at different charging stations (determined by probability calculations)  
#            (EXPECTED because the calculations are made BEFORE going to a charging station, to choose the best one)
#
#   The trade-offs between these 2 features (1.a. and 1.b.) determines which is the most convenient station
#   for a given van for a given time. Even if a charging station is the closest one (low extra travelling time)
#   it may have very high expected waiting time, making it a worse option.

#   2. Charging 20% of the battery size is NOT necessarily the best option. We can calculate the ENERGY NEEDED, and charge
#   this amount. This ENERGY NEEDED will be calculated in advance, given that the routes (consequently the travelling
#   distance) are known in advance for every single van. This will prevent time spent for unncessary charging 

   
# This policy can be easily implemented by companies using a mobile app with the features: 
    
#   . Getting the real time data about the location of the van, the location and 
#     the current queues of charging stations and the routing
#   . Making real-time decisions for the van driver, to select the most convenient charging station
#     and charging amount



# THE PERFORMANCE OF THIS NEW CHARGINg POLICY can easily be seen by tracking the actual waiting times of the vans at charging stations,
# which are in many cases "0", thanks to this new policy. Sometimes you can see the van wait more than 0 minutes,
# which is due to another car coming before our van. So, even though there is still some stochasticity that may cause
# some waiting time in the queue, our policy helps minimizing it. This issue will be deepened in the Assignment 2. 



# Below, we tried to explain the new functions supporting this policy as detailed as possible.
# After the explanation part finishes, you will see the functions.
# If there is still something not clear to you, please do not hesitate to contact to us.


# NEW FUNCTIONS:
    

# 5.1 "Extra_distance_time":    
# Explaning how much extra travel distance time will be needed, compared to the 
# original route. Once 
# If the charging station is located in between the van’s current location and the next customer,
# this score will be low, meaning that visiting this specific charging station do not make us lose any time, in terms of
# travelled distance. 
# So: the LOWER it is, the BETTER that charging station

def extra_distance_time(x1,x2,y1,y2):
    distance_to_current_position = [] ### the distance of the charging stations to the current position of the van
    for i in range(len(charging_stations_x)):
        distance_to_current_position.append(compute_distance(charging_stations_x[i], x1, charging_stations_y[i], y1))
    distance_to_next_customer = [] ### the distance between the charging stations and the next customer
    for i in range(len(charging_stations_x)):
        distance_to_next_customer.append(compute_distance(charging_stations_x[i], x2, charging_stations_y[i], y2))
    total_distance = np.add(distance_to_current_position,distance_to_next_customer)
    extra_distance = np.subtract(total_distance, compute_distance(x1,x2,y1,y2)) #!! the difference between the original
                                                                                # route and the route with the charging
                                                                                # station. 
    
    extra_distance_time= np.divide(extra_distance,avg_speed/60) ### converting the distance matrix to time matrix
    
    return extra_distance_time 



# 5.2 "expected_waiting_time":    
# This part is making probability calculations to find out how long the van will wait in the queue to be served by the charging station. 
# The LOWER this score, the BETTER the charging station.

# The variables such as the ones below are taken into account to make these calculations:
# a. Current position of the van 
# b. Current waiting times of the charging stations (The higher it is, the higher our van will wait in the queue)
# c. Positions of the charging stations (The closer it is to our van, it is less likely that another car come before us )
# d. Parameters : expected charging time (25 minutes) and expected arrival times (1/45 uniform distribution)

# FOR EXAMPLE: 
#If a charging station has 10 minutes of current waiting time, and the van is
# arriving there in 5 minutes, the expected waiting time will be:
# (Probability of a car arriving in 5 minutes * expected charging time of that car) +
# (Current waiting Time (queue) of that charging station – Travelling Time to the Charging Station)  = (5/45)*(25)+(10-5)

# These type of calculations are made for different scenarios. This will help the van to choose the most
# convenient charging station. However, "extra_travel_distance" function above is also important, not only
# "expected_waiting_time". Their scores will be combined in the "charging_station_score" function

# This calculations are made with respect to the parameters of this specific assignment. If companies want to adjust
# this code to their own cases
# (e.g. if instead of uniform arrivals, they have normal distribution of arrivals in the charging statios) , they can
# do their own calculations, and still use this idea.

#!! NOTE: The success of our policy can be seen in the "waiting_times_van" matrix, in which you can see in many cases
# our van waits "0" minutes, thanks to accuracy of our calculations, to choose the charging station that will make us
# wait as short as possible. In the Assignment 2, specific attention will be paid to this issue, comparing it with
# the Base Case policy. 

def expected_waiting_time (x,y,w):
    distance_to_current_position = [] # distance of the charging station to the position of our van.
                                      # this is important, because if the van is close to a charging station
                                      # it is less likely that another car will arrive before our van, and vice versa.
    for i in range(len(charging_stations_x)):
        distance_to_current_position.append(compute_distance(charging_stations_x[i], x, charging_stations_y[i], y))
        
    TTT= np.divide(distance_to_current_position,avg_speed/60) # converting the distance matrix to the time matrix
    CWT=w # current_waiting_time matrix (real time queue data) of the charging stations, which can be seen in the "check_charge" function
    EWT=[] # calculated expected waiting times, using "append" all the calculations will be added here
            # !!! NOTE: This is of course different than CWT, because EWT is trying to estimate what is going to happen
            # in the future with respect to waiting times of the charging stations for a given van,  
            # while CWT is the real-time waiting_times data of charging stations which is used to help estimating EWT
            
         
    for i in range(len(charging_stations_x)):
        if CWT[i]>=TTT[i]:
            EWT.append(CWT[i]-TTT[i]+TTT[i]/45*25)             
            # CWT[i]-TTT[i] : Waiting time caused by current queue. We substract TTT[i], because while the van is travelling
            #                 to that charging station, the current queue will decrease
            # 25: Expected charging time of the car
            # TTT[i]/45 : The expected value of the cars arriving before us (which will increase the current queue)
          
        elif CWT[i]<TTT[i]:
            if TTT[i]-CWT[i]>=25: 
                if CWT[i]/45*25+CWT[i]>TTT[i]: 
                    EWT.append(TTT[i]*25/45+CWT[i]-TTT[i]) # same logic as explained in the first                   
                if CWT[i]/45*25+CWT[i]<=TTT[i]:  
                    EWT.append(15.46)  # 15.46 is determined by running simulations specificly to identify
                                    # average waiting time at the charging stations

                    
                    
            elif TTT[i]-CWT[i]<25:
                
                if CWT[i]>45:
                    EWT.append((TTT[i]*25/45)+CWT[i]-TTT[i]) # same logic explained in the first case
                    
                elif CWT[i]<=45:
                    EWT.append(CWT[i]/45*(25+CWT[i]-TTT[i]+(TTT[i]-CWT[i])/45*25)+(45-CWT[i])/45*(TTT[i]-CWT[i])/45*(25-(TTT[i]-CWT[i])/2))
                    
                    # This part also have a similar logic, with few different features:
                    # . (TTT[i]-CWT)/2: Divided by 2, expected arrival time of a car coming before us
                    
    return EWT



# 5.3 "charging_station_score":
# Summing up two scores described above:
# charging_station_score = extra_distance_time + expected_waiting_time
# The LOWER the "charging_station_score", the BETTER that particular charging station, due to both it is close to
# the van’s current position, and also it has low expected waiting time .
def charging_station_score (extra_travel,expected_time,x1,x2,y1,y2,w):
    return np.add(extra_travel(x1,x2,y1,y2),expected_time(x1,y1,w))



# 5.4. "best_station_id":
# # Getting the id of the best station (the one that has the lowest charging_station_score)
# which will be used to move the van to that particular station
def best_station_id (x1,x2,y1,y2,w):
    z= charging_station_score(extra_distance_time,expected_waiting_time,x1,x2,y1,y2,w)
    abc = np.where(z == np.amin(z))
    result=np.amin(abc)
    return result



# 5.5. "best_station_time" :
# Giving the lowest "charging_station_score" (lowest means actually the best charging station)
# This will be used in the "check_charge" function to set some THRESHOLD, above which the van postpone its
# charging decision (unless it has very few battery) to find a better station (after visiting the next customer,
# it will evaluate the situation again)
def best_station_time (x1,x2,y1,y2,w):
    l= charging_station_score(extra_distance_time,expected_waiting_time,x1,x2,y1,y2,w)
    abcd=np.amin(l)
    return abcd



# 5.6 "needed_energy": 
# Calculating the extra energy needed fora van for its all tour, given the total distance it travels
# which is calculated easily after determining the routes.
def needed_energy(distance,battery_size):
    e = 0.218 + 1.359 / avg_speed - 0.003*avg_speed + 2.981*(10**(-5))*avg_speed**2
    travel_energy = distance*e*safety_energy_level
    needed_energy=0
    if travel_energy-battery_size>=0:
        needed_energy=travel_energy-battery_size
    else:
         needed_energy=0
    return needed_energy



########################################################################################################################
#                                                       6. BASE CASE FUNCTIONS AND PARAMETERS

# 6.1 Old (Base Case) functions:

# Function that imports the customers coordinates from a file:
def read_customers(number): # number is the number of customers to be visited    
    customers_x = []
    customers_y = []
    customers_ID = []
    path = "customers_location_" + str(number) + ".csv"
    h = open(path, "r")
    line_count = 0
    for line in h:
        if line_count == 0: # ensures that the first line (containing column names) is disregarded
            line_count += 1
            continue
        # the elements of the list a are strings, corresponding to the comma-separated values in the line
        a = line.split(',')
        customers_ID.append(float(a[2]))
        customers_x.append(float(a[0]))
        customers_y.append(float(a[1]))
    h.close()
    return(customers_x, customers_y, customers_ID)

# Function that imports the charging stations coordinates from a file:
def read_charging_stations(number): # number is the number of charging stations in the area
    charging_stations_x = []
    charging_stations_y = []
    path = "charging_stations_location_" + str(number) + ".csv"
    h = open(path, "r")
    line_count = 0
    for line in h:
        if line_count == 0: # ensures that the first line (containing column names) is disregarded
            line_count += 1
            continue
        # the elements of the list a are strings, corresponding to the comma-separated values in the line
        a = line.split(',')
        charging_stations_x.append(float(a[1]))
        charging_stations_y.append(float(a[2]))
    h.close()
    return(charging_stations_x, charging_stations_y)


# Function that computes the distance between the points with coordinates (x1,y1) and (x2,y2):
def compute_distance(x1, x2, y1, y2):
    return abs(x1 - x2) + abs(y1 - y2) #HP: Rectilinear distance

# Function that determines the closest charging station to the point with coordinates (x,y)
def closest_charging_station(x, y):
    distance = []
    for i in range(len(charging_stations_x)):
        distance.append(compute_distance(charging_stations_x[i], x, charging_stations_y[i], y))
    minimum = min(distance)
    return distance.index(minimum)



# 6.2. Old functions modified to support stochastic speed:
     
# Instead of writing "avg_speed", here "speed" will be used.
def compute_time(distance,speed):
    return (distance/speed)*60 

# Instead of writing "avg_speed", here "speed" will be used.
def compute_energy(x1, x2, y1, y2,speed): 
    distance = compute_distance(x1, x2, y1, y2) 
    e = 0.218 + 1.359 / speed - 0.003*speed + 2.981*(10**(-5))*speed**2
    energy = e*distance
    return energy


### just for plotting the route of the vans (including the charging stations they chose to go)(visualization)




#########################################################################################################################################
#                                                      7. CREATION OF ARRAYS FOR THE FOLLOWING SECTIONS
# 7.1 Routing Visualization: Empty Arrays Creation 
                                                                 
# Check part 12. Routing Visualization. This part only helps recording the data
XX=[]  
YY=[]  
XX_Charging=[] 
YY_Charging=[] 
for i in range (num_van): 
    XX.append([])
    YY.append([])
    XX_Charging.append([])
    YY_Charging.append([]) 

# 7.2 Taking the customers and charging stations into arrays
customers_ID = read_customers(num_customers)[0]
customers_x = read_customers(num_customers)[1]
customers_y = read_customers(num_customers)[2]
charging_stations_x = read_charging_stations(num_stations)[0]
charging_stations_y = read_charging_stations(num_stations)[1]




##########################################################################################################################################
#                                                                    8. CLASS VAN

class Van(Agent):
    
    # 8.1 "__init__" function
    
    def __init__(self, unique_id, model,routing,distance):
        super().__init__(unique_id, model)
        
#       8.1.1 Old attributes
        self.id_van = unique_id
        self.pos_x = warehouse_x # Current position of the van. The starting position is the warehouse
        self.pos_y = warehouse_y # Current position of the van.The starting position is the warehouse
        self.next_stop_x = 0 # This attribute will be updated with the x coordinate of the next customer/station/warehouse to be visited
        self.next_stop_y = 0 # This attribute will be updated with the y coordinate of the next customer/station/warehouse to be visited
        self.battery_size = battery_size[int(self.id_van[4:])] # [kWh] 
        self.remaining_energy = battery_size[int(self.id_van[4:])]
        self.task_endtime = 0 # Every time the van completes a task, this variable will be increased by a value equal to the task duration
        self.i_closest_station = 0 # This attribute could be initialized at any value (or this line of code could even be removed)
        self.need_to_charge = False # This attribute will be True if the van needs to charge
        self.tour_end = False # This attribute will be True if all the customers have been visited and the van has come back to the warehouse
        

        

        
#       8.1.2 New and modified attributes

        # Routing attributes
        self.route = routing
        self.next_customer = routing[0]  # Id of the next customer to be visited, considering the routing.

        # NEW charging station selection policy attributes
        self.waiting_times_van=[] # keeping the record of how long did a van waited in queue for charging.
        self.i_best_station=0 # The ID of the most convenient charging station for the specific van, check "best_station_id" function
        self.time_best_station=0 # The charging of score of the most convenient charging station, check "best_station_time" function
        self.charged=0 # Checking if the vehicle is already charged or not, if it is 1, it will not be charged again.
                       # This is due to the van is charging the amount energy needed (explained below) at once
        self.needed_energy = needed_energy(distance[int(self.id_van[4:])],self.battery_size) # to calculate how much extra energy needed overall for each van,
                                                                                              # to define the charging amount 
        # Stochastic speed attribute                                                                                       
        self.speed=0 # the speed of a van will be updated by "check_charge_function" to take into account stochasticity  
        
        #  Unloading time attribute, just to keep record      
        self.unloading_time=0        
   
        # task end
        self.task_end=0
    
    # 8.2 "get_routes" function
    # This is the function which make the van move according to the routing defined in the Van class     
    def get_routes (self):
        pos=self.route.index(self.next_customer)
        pos+=1
        return self.route[pos]
        
    
    
    # 8.3 "check_charge" function
    def check_charge(self):
        
        if stochastic_speed==1:
            self.speed=random.normalvariate(avg_speed,stdev_speed)  # speed is now STOCHASTIC, updated in every step
        elif stochastic_speed==0:
            self.speed=avg_speed
            
        self.unloading_time=random.uniform(5, 20) # actually same, to print the unloading times
                                                  # we used self.unloading_time to record it
                                                  
        # BASE CASE                                         
        if self.next_customer < num_customers: # If the next stop is a customer (strictly < because the id starts from 0):
            next_x = customers_x[self.next_customer]
            next_y = customers_y[self.next_customer]
        else:
            next_x = warehouse_x
            next_y = warehouse_y
            
        
        energy_to_next_customer_or_whs = compute_energy(next_x, self.pos_x, next_y, self.pos_y,self.speed)
        
      # NEW !!!
      # This code gets all the current waiting times of the charging stations, which will be
      # used in "expected_waiting_time" function to make calculations
        w=[]
        for i in range(130):
            w.append(self.model.schedule_stations.agents[i].waiting_time)
            
     
     
        # BASE CASE
        if self.next_customer < num_customers:
            i_closest_station_next_customer = closest_charging_station(next_x, next_y) #Index of the charging station closest to the next customer
            energy_to_closest_station = compute_energy(next_x, charging_stations_x[i_closest_station_next_customer], next_y, charging_stations_y[i_closest_station_next_customer],self.speed)
        else:
            energy_to_closest_station = 0 # Once come back at the whs, the van can charge there
            
      
    
    
        
        # 8.3.1 New charging policy
        if charging_policy==1: # The NEW charging station selection policy
            
            
            
         # 8.3.1.1 Battery about to die
         # In this part, if the battery is at the critical level, the BEST charging station(NOT necessarily the closest)
         # is choosen with the help of "charging_station_score" function and "i_best_station" function 
         # which gives the id of the best charging station
                       
            if self.remaining_energy - energy_to_next_customer_or_whs - energy_to_closest_station < 0.1*self.battery_size:
                self.i_best_station= best_station_id(self.pos_x,next_x,self.pos_y,next_y,w) # getting the id of                                                                                          
                self.next_stop_x = charging_stations_x[self.i_best_station] #!!!!!!!!!  instead of closest
                self.next_stop_y = charging_stations_y[self.i_best_station] #!!!!!!!!!  we use i_best_station
                self.need_to_charge = True                   
                XX_Charging[int(self.unique_id[4])].append(self.next_stop_x) ### just for plotting the route (visualization)
                YY_Charging[int(self.unique_id[4])].append(self.next_stop_y) ### just for plotting the route (visualization)
                
           
           
           

           
           
            # 8.3.1.2 Early charging:
            # This is one of the key parts of this new charging station selection policy
            # The van can charge its battery before it has a very low battery. In this way, it has more opportunity
            # to find a better station. Here we introduce some key concepts:
                # a) Early charging condition: When the van has enough empty space in its battery to charge "needed_energy"
                #    which is the extra energy needed for the whole tour of a van calculated in advance, it can charge it.
                #    If the van needs no energy (needed_energy==0), it will not charge itself at all.
                # b) Charging station score threshold: We set a threshold, above which the van postpones its decision making until
                #    the next customer is visited. In this way, the van searches for very convienient options, and does not accept
                #    the options worse than the threshold "charging_station_score" value.                
            elif self.remaining_energy+self.needed_energy<self.battery_size: # Early charging condition (explained above)
                self.i_best_station= best_station_id(self.pos_x,next_x,self.pos_y,next_y,w) # get the id of the most convenient station
                self.time_best_station= best_station_time(self.pos_x,next_x,self.pos_y,next_y,w) # get the "charging_score" of the most convenient station
                if self.needed_energy>0: # this is calculated in advance depending on the route. if the van needs no charging
                                         # according to our calculations,it wont do any charging.
                    if self.time_best_station<5: # Charging station score threshold: explained above
                        if self.charged<1: # if the van is already charged with the needed energy, do not charge it again
                            self.next_stop_x = charging_stations_x[self.i_best_station] # go to the best charging station
                            self.next_stop_y = charging_stations_y[self.i_best_station] # go to the best charging station                    
                            XX_Charging[int(self.unique_id[4])].append(self.next_stop_x)### just for plotting the route (visualization)
                            YY_Charging[int(self.unique_id[4])].append(self.next_stop_y)### just for plotting the route (visualization)
                            self.charged=self.charged+1 # showing that the van is already charged, preventing
                                                        # unneccessary charging
                            self.need_to_charge = True
                        else:
                            self.next_stop_x = next_x
                            self.next_stop_y = next_y
                            
                    else:
                        self.next_stop_x = next_x
                        self.next_stop_y = next_y
                               
                else:
                    self.next_stop_x = next_x
                    self.next_stop_y = next_y
            else:
                self.next_stop_x = next_x
                self.next_stop_y = next_y          
            #print("\nself_pox_x=",self.pos_x,"self_pos_y=",self.pos_y,"next_stop_x=",self.next_stop_x,"next_stop_y=",self.next_stop_y) 
            
#       8.3.2 Base case charging policy     
        elif charging_policy==0:
            if self.remaining_energy - energy_to_next_customer_or_whs - energy_to_closest_station < 0.1*self.battery_size:         
                self.i_closest_station = closest_charging_station(self.pos_x, self.pos_y)
                self.next_stop_x = charging_stations_x[self.i_closest_station]
                self.next_stop_y = charging_stations_y[self.i_closest_station]
                self.need_to_charge = True
            else:
                self.next_stop_x = next_x
                self.next_stop_y = next_y              

    # 8.4 "move" function:         
    # Function that simulates the travel of the van to the next stop (and the unloading of goods at the customers):  
    def move(self):
        
        # Base Case
        distance_next_stop = compute_distance(self.pos_x, self.next_stop_x, self.pos_y, self.next_stop_y)      
        self.task_endtime += compute_time(distance_next_stop,self.speed) #Travel time
        

        
        self.remaining_energy -= compute_energy(self.pos_x, self.next_stop_x, self.pos_y, self.next_stop_y,self.speed)
      
        if self.need_to_charge == False: #if the next stop is not a charging station
            if self.next_customer < num_customers: #if the next stop is a customer (and not the warehouse)
               self.task_endtime += self.unloading_time #Unloading time (between 5 and 20 minutes) 
               #print("\n"+self.unique_id,"visiting customer number",self.next_customer)
               self.next_customer = self.get_routes()
               
            else:
                #print("\n"+self.unique_id,"going back to the warehouse")
                self.tour_end = True
        # else:
        #     print("\n"+self.unique_id,"going to the charging station",self.i_best_station)
        #print("self.unloading_time=",self.unloading_time)
        # print("self_speed=",self.speed)
        # print("Travelled distance:",distance_next_stop,"km")
        # print("Task endtime:",self.task_endtime)
        # print("Remaining energy:",self.remaining_energy)
        # print("Needed_energy:", self.needed_energy)
                        
        self.pos_x = self.next_stop_x
        self.pos_y = self.next_stop_y
        
        if self.pos_x==warehouse_x and self.pos_y==warehouse_y:
            task_end_time.append(self.task_endtime)
            
        # New
        XX[int(self.unique_id[4])].append(self.next_stop_x) ### just for plotting the route (visualization)
        YY[int(self.unique_id[4])].append(self.next_stop_y) ### just for plotting the route (visualization)   
        
    
    # 8.5."charging" function:
    # Function that simulates the queuing at the charging station and the battery charging:
    def charging(self):
        #print("\nCHARGING TIME! YOU CAN SEE HOW LONG THE VAN WAITED IN THE QUEUE BELOW")
        
        # NEW: Taking the current waiting times of the charging stations
        b=[]
        for i in range(130):
            b.append(self.model.schedule_stations.agents[i].waiting_time) 
                                                                            
            
        # 8.5.1 New charging amount policy
        # The main difference in this part: charging_size=self.needed_energy // instead of 20% of the battery size
        if charging_policy==1: 
            self.waiting_times_van.append(b[self.i_best_station])  # to only keep record of how much did the van waited.                                                                  
            # print(self.unique_id,'charging in the charging station',self.i_best_station)
            # print("the van waits in the queue (minutes):",b[self.i_best_station])
            self.task_endtime += self.model.schedule_stations.agents[self.i_best_station].waiting_time 
            
            if self.needed_energy>0: # standard charging
                charging_size = self.needed_energy # !!!! charging size is equal to needed_energy, calculated in advance       
            elif self.needed_energy==0: # if the van has 0 extra energy needed, but since it has very low battery left
                                          # it goes to the charging station, and charge it 10% of the battery sizde
                charging_size=0.2*battery_size
                
            power = 10 #kW (power of the charging station)
            charging_time = (charging_size/power)*60
            self.model.schedule_stations.agents[self.i_best_station].waiting_time += charging_time
            self.task_endtime += charging_time 
            self.remaining_energy += charging_size
            self.need_to_charge = False
            # print("self.needed_energy is:",self.needed_energy)
            # print("charging time:",charging_time)        
            # print("Task endtime:",self.task_endtime)
            # print("Remaining energy:",self.remaining_energy)
            
        # 8.5.2 Base Case charging amount policy         
        elif charging_policy==0:
            self.waiting_times_van.append(b[self.i_closest_station]) 
            # print(self.unique_id,'charging in the charging station',self.i_closest_station)
            # print("the van waits in the queue (minutes):",b[self.i_closest_station])
            self.task_endtime += self.model.schedule_stations.agents[self.i_closest_station].waiting_time 
            charging_size = 0.2*self.battery_size        
            power = 10 #kW (power of the charging station)
            charging_time = (charging_size/power)*60
            self.model.schedule_stations.agents[self.i_closest_station].waiting_time += charging_time
            self.task_endtime += charging_time 
            self.remaining_energy += charging_size
            self.need_to_charge = False
            # print("self.needed_energy is:",self.needed_energy)
            # print("charging time:",charging_time)        
            # print("Task endtime:",self.task_endtime)
            # print("Remaining energy:",self.remaining_energy)

        

    # 8.6 "step" function: Same as the Base Case   
    def step(self):
        if self.tour_end == False: # The step function is executed only if the van must still go to other customers (or go back to the warehouse)
            if self.task_endtime <= self.model.system_time: #To make sure that only the vans which have completed their task are activated
                if self.need_to_charge == True:
                    self.charging()
                self.check_charge()
                self.move()


##############################################################################################################################################################
#                                                                   9. CLASS CHARGINGSTATION 

# Same as the Base Case
class ChargingStation(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.waiting_time = 0 # Time that a van must wait before beginning its charging process at this station

    def step(self):
        if random.randint(1,45) == 1: # On average, an "external" vehicle arrives at the station once every 45 min (-> 45 steps) (NB Excluding "our" vans)
            self.waiting_time += random.normalvariate(25,5.0) # The charging time of an "external" vehicle can be described with a normal distribution with avg = 20 min and std dev = 5 min
        
        if self.waiting_time >= 1: # 1 is the step duration
            self.waiting_time -= 1 # 1 is the step duration
        else:
            self.waiting_time = 0

##############################################################################################################################################################
#                                                                   10. CLASS GREENMODEL                                                                    

#Same as the Base Case
class GreenModel(Model):
    def __init__(self, num_van): 
        self.num_vans = num_van
        self.schedule_vans = RandomActivation(self)
        self.schedule_stations = BaseScheduler(self) # BaseScheduler activates all the agents at each step, one agent at a time, in the order they were added to the scheduler. We could use also RandomActivation 
        self.system_time = 0 # This attribute will keep track of the system time, advancing by 1 minute at each step
        
        #Creating vans:
        for i in range(self.num_vans):
            a = Van("Van_"+str(i), self,van_routes[i],Tot_distance)
            self.schedule_vans.add(a)

        #Creating charging stations:
        for i in range(len(charging_stations_x)):
            a = ChargingStation("Station_"+str(i), self)
            self.schedule_stations.add(a)

    def step(self):
        self.schedule_vans.step()
        self.schedule_stations.step()
        self.system_time += 1   #HP: 1 step = 1 minute

##############################################################################################################################################################
#                                                                   11. RUNNING THE SIMULATION
# 11.1 Task End Time Statistics
task_end_time_multiple_runs=[] #just to keep all tour end times of 4 vans for each run of simulation
task_end_time=[] #after completing each run, array waill hold tour end times of 4 vans.
max_task_end_time=[] #for 4800 runs (6 options for battery sizes and 800 runs for each options),
                     #maximum tour end time value after completing each run is going to be collected in this array. 
MEAN=[]
SAMPLE_VARIANCE=[]
SAMPLE_MEAN_VARIANCE=[] 

# 11.2 Statistics of waiting times of the vans at the charging station queues
waiting_times_van=np.empty(0)
MEAN_WT=[]
VARIANCE_WT=[]
  # to keep the track of the waiting times of the vans for the charging station queues

# 11.3 Anova Settings Battery Size
battery_size_change=0 #DO NOT CHANGE THIS VALUE, it is initialized as 0 to iterate over different battery sizes!
charging_policy_change=0 #DO NOT CHANGE THIS VALUE, it is initialized as 0 to iterate over different battery sizes!

# 11.4 Anova

if anova_num==0: #charging policy is fixed and experiments conducted on different battery size options
    if charging_policy==0:        
       print("\n***You are running an experiment with*** \n\nCharging Policy: The Base Case charging policy\nThe Battery Sizes: Varying")
    if charging_policy==1:        
       print("\n***You are running an experiment with*** \n\nCharging Policy: The New charging policy\nThe Battery Sizes: Varying")
    while True:       
        if battery_size_change==0:
            battery_size=[20,20,20,20]
        elif battery_size_change==1:
            battery_size=[25,25,25,25]    
        elif battery_size_change==2:
            battery_size=[30,30,30,30]
        elif battery_size_change==3:
            battery_size=[35,35,35,35]
        elif battery_size_change==4:
             battery_size=[30,30,20,20]
        elif battery_size_change==5:
             battery_size=[25,25,20,20]   
        
        for i in range(N): # to make multiple simulations (N = 800)     
            task_end_time=[] #after completing run array is nullified.
            model = GreenModel(num_van)
            for j in range(720):  # Simulation of 8 hours -> 480 minutes (each step corresponds to 1 minute)
                model.step()
            
   
            maximum=max(task_end_time) #taking the maximum of tour end times among 4 vans
            max_task_end_time.append(maximum) #for 4800 runs (6 options for battery sizes and 800 runs for each options), 
                                              #appending maximum tour end time value after completing each run
                    
        if battery_size_change==5: 
            y1=np.array(max_task_end_time[:N])      #y1: max tour end time values coming from 800 runs for the battery size option [20,20,20,20]
            y2=np.array(max_task_end_time[N:N*2])   #y2: max tour end time values coming from 800 runs for the battery size option [25,25,25,25]
            y3=np.array(max_task_end_time[N*2:N*3]) #y3: max tour end time values coming from 800 runs for the battery size option [30,30,30,30]
            y4=np.array(max_task_end_time[N*3:N*4]) #y4: max tour end time values coming from 800 runs for the battery size option [35,35,35,35]
            y5=np.array(max_task_end_time[N*4:N*5]) #y5: max tour end time values coming from 800 runs for the battery size option [30,30,20,20]
            y6=np.array(max_task_end_time[N*5:N*6]) #y6: max tour end time values coming from 800 runs for the battery size option [25,25,20,20]

            print("\n",f_oneway(y1,y2,y3,y4,y5,y6)) #result of oneway anova test

            data_to_plot=[y1,y2,y3,y4,y5,y6] #data_to_plot: max_task_end_time array including maximum tour end time value for 4800 runs 
                                             #(6 options for battery sizes and 800 runs for each options) 
            fig=plt.figure(1)
            ax=fig.add_subplot(111)
            bp=ax.boxplot(data_to_plot)
            ax.set_xticklabels(["[20,20,20,20]","[25,25,25,25]","[30,30,30,30]","[35,35,35,35]","[30,30,20,20]","[25,25,20,20]"])
            plt.xticks(fontsize=12,rotation=90)
            plt.yticks(np.arange(320,800,40),fontsize=12)

            left, right = plt.xlim()
            plt.hlines(480, xmin = left, xmax = right, color = "b", linestyles = "--") #threshold of 480 minutes for max tour end times  
            if charging_policy==0:
                plt.title("Base Case Charging Policy With Changing Batteries")
            if charging_policy==1:
                plt.title("New Charging Policy With Changing Batteries")
            break
        else:
            battery_size_change+=1 #iterate over various battery size options

    
    
    
elif anova_num==1: #battery size is fixed and experiments conducted on different charging policy options
    print("\n***You are running an experiment with*** \n\nThe fixed battery size:",battery_size,"\nThe charging policies: Varying")
    while True:
        if charging_policy_change==0: 
            charging_policy=0 #assign charging policy to base case charging polilcy(0)
        elif charging_policy_change==1:
            charging_policy=1 #assign charging policy to new charging polilcy (1)
    
        for i in range(N): # to make multiple simulations        
            task_end_time=[]
            model = GreenModel(num_van)
            for j in range(720):  # Simulation of 8 hours -> 480 minutes (each step corresponds to 1 minute)
                model.step()
            maximum=max(task_end_time) #taking the maximum of tour end times among 4 vans
            max_task_end_time.append(maximum) #for 4800 runs (6 options for battery sizes and 800 runs for each options), 
                                              #appending maximum tour end time value after completing each run
        if charging_policy_change==1:
            y1=np.array(max_task_end_time[:N])
            y2=np.array(max_task_end_time[N:N*2]) 
            print("\n",f_oneway(y1,y2)) #result of oneway anova test
            data_to_plot=[y1,y2]
            fig=plt.figure(1)
            ax=fig.add_subplot(111)
            bp=ax.boxplot(data_to_plot)
            ax.set_xticklabels(["Base Case \nCharging Policy","New \nCharging Policy"])
            plt.xticks(fontsize=12,rotation=90)
            plt.yticks(np.arange(320,800,40),fontsize=12)
            a=str(battery_size)
            plt.title("Battery Size:"+a+" ,Varying Charging Policy")
            left, right = plt.xlim()
            plt.hlines(480, xmin = left, xmax = right, color = "b", linestyles = "--")
            break
        else:
            charging_policy_change+=1 #iterate over different charging policies
            

###########################################################################################################################################################
#                                                                   12. ROUTING VISUALIZATION                

# #Plots for visualizing the routes 
# import matplotlib.pyplot as plt
# import random

# #To use random colors for each van
# number_of_colors = 8
# color = ["#"+''.join([random.choice('1456789ABCDEF') for j in range(6)])
#               for i in range(number_of_colors)]


# from matplotlib import pyplot as plt
# mylegend=["Van0","Van1","Van2","Van3", "ChargingStations"]
# plt.figure()
# for i in range(len(XX)):
#     x=XX[i]
#     y=YY[i]
#     x = np.insert(x, 0, 35.0, axis=0)
#     y = np.insert(y, 0, 35.0, axis=0)
#     plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1, color=color[i], label='Van %s' % i)
#     if i==len(XX)-1:
#         plt.plot(XX_Charging[i], YY_Charging[i], "s", color="b", label='Charging Stations')
#     else:
#         plt.plot(XX_Charging[i], YY_Charging[i], "s", color="b")

# plt.legend(prop={'size': 7},bbox_to_anchor=(1,1))
# plt.show()