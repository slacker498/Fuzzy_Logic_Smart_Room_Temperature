'''
    Group 9: Emmanuel Nkunim Amoah Owusu-Marfo, 
             Jachin Dzidumor Hugh Kpogli,
             Kingsley Wunzooya Nahyi

    Project Title: 5.1 Smart Room Temperature Controller
    Objective:
    Simulate an air conditioner controller that adjusts fan speed based on fuzzy inputs: temperature
    and occupancy level.
    Tasks:
    • Collect inputs: temperature (10 - 40°C), number of people (0 - 20).
    • Return fan speed (0 - 100%).
    • Create a 3D plot showing how input changes affect the output


'''

import numpy as np 
import skfuzzy as sk
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


'''
    Definition for fuzzy logic control system
'''
def getFuzzyCtrlSystem():
    #antecedents or inputs coupled with creation of the universes or ranges for the various values
    temperature = ctrl.Antecedent(np.arange(10,41,1), 'temperature')
    occupancy = ctrl.Antecedent(np.arange(0,21,1), 'occupancy')

    #consequent or outcome 
    fan_speed = ctrl.Consequent(np.arange(0,101,1), 'fan_speed')

    #definition of various membership functions for temperature
    temperature['cool'] = sk.trapmf(temperature.universe, [10,10,18,24])
    temperature['comfortable'] = sk.trimf(temperature.universe, [19,24,30])
    temperature['warm'] = sk.trapmf(temperature.universe, [24,30,40,40])

    #definition of various membership functions of occupancy
    occupancy['low'] = sk.trapmf(occupancy.universe, [0,0,5,10])
    occupancy['medium'] = sk.trimf(occupancy.universe, [5,10,15])
    occupancy['high'] = sk.trapmf(occupancy.universe, [10,15,20,20])

    #definition of various memberships for fan speed 
    fan_speed['low'] = sk.trapmf(fan_speed.universe, [0,0,30,50])
    fan_speed['medium'] = sk.trimf(fan_speed.universe, [30,50,70])
    fan_speed['high'] = sk.trapmf(fan_speed.universe, [50, 70, 100, 100])

    #rules for antecedents and outcomes. 
    rules = [
        # Cool temperature rules
        ctrl.Rule(temperature['cool'] & occupancy['low'], fan_speed['low']),
        ctrl.Rule(temperature['cool'] & occupancy['medium'], fan_speed['low']),
        ctrl.Rule(temperature['cool'] & occupancy['high'], fan_speed['medium']),
        
        # Comfortable temperature rules  
        ctrl.Rule(temperature['comfortable'] & occupancy['low'], fan_speed['low']),
        ctrl.Rule(temperature['comfortable'] & occupancy['medium'], fan_speed['medium']),
        ctrl.Rule(temperature['comfortable'] & occupancy['high'], fan_speed['high']),
        
        # Warm temperature rules
        ctrl.Rule(temperature['warm'] & occupancy['low'], fan_speed['medium']),
        ctrl.Rule(temperature['warm'] & occupancy['medium'], fan_speed['high']),
        ctrl.Rule(temperature['warm'] & occupancy['high'], fan_speed['high'])
    ]

    #building the control system 
    fan_ctrl = ctrl.ControlSystem(rules)
    fan_sim = ctrl.ControlSystemSimulation(fan_ctrl)

    return fan_sim

# Method to accept inputs and perform data validations
def getInputValues():
    print(" Smart Room Temperature Controller ".center(50, "*"))

    while True:
        try:
            num_of_input_pairs = int(input("Enter the number of (temperature, occupancy level) pairs you would be entering: "))
            if num_of_input_pairs <= 0:
                print("Please enter a positive integer value! Try again!")
                continue
            break
        except ValueError:
            print("Please enter a valid integer value! Try again!")

    temp_list = []
    print("Enter your temperature values (10 - 40°C): ")
    for i in range(num_of_input_pairs):
        while True:
            try:
                temp_value = float(input(f"{i+1}: "))
                if not (10 <= temp_value <= 40) :
                    print("Temperature values must be in the allowed range from (10 - 40°C). Please Try again!")
                    continue
                temp_list.append(temp_value)
                break
            except ValueError:
                print("Please enter a valid integer/floating-point value! Try again!")

    occ_list = []
    print("Enter the occupancy levels (0 - 20): ")
    for i in range(num_of_input_pairs):
        while True:
            try:
                occ_value = int(input(f"{i+1}: "))
                if not (0 <= occ_value <= 20) :
                    print("Occupancy levels must be in the allowed range from (0 - 20 people). Please Try again!")
                    continue
                occ_list.append(occ_value)
                break
            except ValueError:
                print("Please enter a valid integer value! Try again!")     

    temp_inputs = np.array(temp_list)
    occ_inputs = np.array(occ_list)

    return temp_inputs, occ_inputs
    

# Method to accept crisp input values and apply the fuzzy control system to obtain the accepted output values
def fuzzifyAndDefuzzify(temp_inputs, occ_inputs, fan_sim):
    fan_speeds = np.zeros(len(temp_inputs))

    for i in range(len(temp_inputs)):
        fan_sim.reset() # Safequard to prevent leftover values
        fan_sim.input['temperature']  = temp_inputs[i]
        fan_sim.input['occupancy'] = occ_inputs[i]
        fan_sim.compute()
        fan_speeds[i] = fan_sim.output['fan_speed'] 

    # Display a table of final outputs with its corresponding given inputs
    print("\nPlotted Values")
    print("Temperature (°C) --|--", "Occupancy Level (people) --|--", "Fan Speeds(%)")
    for temp, occ, speed in zip(temp_inputs , occ_inputs, fan_speeds):
        print(f"{temp:.1f} {occ:20.0f} {speed:29.0f}%")
    
    return fan_speeds


def generateScatterPlot():
    # Generate all (temp, occ) pairs with their corresponding fan speed levels
    temp_inputs, occ_inputs = np.meshgrid(np.linspace(10, 40, 75), np.linspace(0, 20, 75))
    fan_speeds = np.zeros_like(temp_inputs)

    for i in range(temp_inputs.shape[0]):
        for j in range(temp_inputs.shape[1]):
            fan_sim.reset() # Safequard to prevent leftover values
            fan_sim.input['temperature']  = temp_inputs[i,j]
            fan_sim.input['occupancy'] = occ_inputs[i,j]
            fan_sim.compute()
            fan_speeds[i,j] = fan_sim.output['fan_speed']

    # Create a figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a scatter plot
    ax.scatter(temp_inputs.ravel(), occ_inputs.ravel(), fan_speeds.ravel(), c = fan_speeds.ravel(), s=50)

    # Customize labels and settings
    ax.set_xlabel('Temperature (°C)')
    ax.set_xlim(10, 40)
    ax.set_ylabel('Occupancy level (number of people)')
    ax.set_ylim(0, 20)
    ax.set_yticks(range(0,21,4))
    ax.set_zlabel('Fan speeds (%)')
    ax.set_zlim(0, 100)
    ax.set_title("Smart Room Temperature Controller")

    # Show the plot
    plt.show()



# Main Program
if __name__ == "__main__":
    fan_sim = getFuzzyCtrlSystem()
    temperature, occupancy = getInputValues()
    fan_speeds = fuzzifyAndDefuzzify(temperature, occupancy, fan_sim)
    generateScatterPlot()