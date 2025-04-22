#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
POWER GENERATION IN FALL 2030
"""

from pyomo.environ import *
import matplotlib.pyplot as plt

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Demand data

df = pd.read_csv('EPREFERENCE_FlexNONEload_hourly.csv')
demand = df[['p116', 'p117']]
demand = demand.sum(axis=1)
years = df[['year']]

demand=demand[np.where(years==2030)[0]]
demand=demand.reset_index(drop=True)
demand1=demand[0:24]
demand2=demand[2208:2232]
demand3=demand[4392:4416]
demand4=demand[6576:6600]

# Wind Capacity Factors

Win1CF_dict = {
    1: 0.000, 2: 0.000, 3: 0.000, 4: 0.000, 5: 0.000, 6: 0.000, 7: 0.000, 8: 0.000, 
    9: 0.000, 10: 0.000, 11: 0.145, 12: 0.336, 13: 0.663, 14: 0.590, 15: 0.459, 16: 0.582, 
    17: 0.615, 18: 0.671, 19: 0.395, 20: 0.219, 21: 0.152, 22: 0.258, 23: 0.284, 24: 0.082}

# Generators fleet (*)

df = pd.read_csv('3_1_Generator_Y2023.csv', header=1)
states=df[["State"]]
gens=df[['Technology', 'Nameplate Capacity (MW)', 'Generator ID', 'Heat Rate (MMBtu/MWh)', 'Variable O&M ($/MWh)', 'Fuel Price ($/MMBtu)']]
gens=gens.drop(np.where(states!='WV')[0]).reset_index(drop=True)

# Concrete Model object

model = ConcreteModel()
gen_type=gens[['Technology']].values.flatten()

non_coal_generators=gens.drop(np.concatenate((np.where(gen_type == "Conventional Steam Coal")[0], np.where(gen_type == "Onshore Wind Turbine")[0]))).reset_index(drop=True)
coal_generators=gens.drop(np.where(gen_type!="Conventional Steam Coal")[0]).reset_index(drop=True)
wind_generators=gens.drop(np.where(gen_type!="Onshore Wind Turbine")[0]).reset_index(drop=True)
print(coal_generators)

# Set of generators

model.non_coal_generators = Set(initialize=non_coal_generators["Generator ID"])
model.coal_generators = Set(initialize=coal_generators["Generator ID"])
model.wind_generators = Set(initialize=wind_generators["Generator ID"])
model.times = RangeSet(1, 24)
model.generators=model.non_coal_generators|model.coal_generators|model.wind_generators

gen_ids = gens[['Generator ID']].values.flatten()
print(gen_ids)

# Parameters (*)

HeatRate_dict={}; FuelPrice_dict={}; VariableOM_dict={}; MaxCapacity_dict={}; MinStableLoad_dict={}; StartupCost_dict={}; Ramprate_dict={}; CO2_ER_dict={}; NOx_ER_dict={}
for i, gen in gens.iterrows():
    HeatRate_dict[gen["Generator ID"]]=float(gen["Heat Rate (MMBtu/MWh)"])
    FuelPrice_dict[gen["Generator ID"]]=float(gen["Fuel Price ($/MMBtu)"])
    VariableOM_dict[gen["Generator ID"]]=float(gen["Variable O&M ($/MWh)"])
    MaxCapacity_dict[gen["Generator ID"]]=float(gen["Nameplate Capacity (MW)"])

print(HeatRate_dict)
ctr=1; demand_dict={}
demand=demand4
for d in demand:
    demand_dict[ctr]=d
    ctr=ctr+1
print(demand_dict)

    #FuelPrice_dict[gen]=gens[]
    
model.pHeatRate = Param(model.generators,initialize=HeatRate_dict)
model.pFuelPrice = Param(model.generators,initialize=FuelPrice_dict)
model.pVariableOM = Param(model.generators,initialize=VariableOM_dict)
model.pMaxCapacity = Param(model.generators, initialize=MaxCapacity_dict)
model.pMinstableload = Param(model.generators, initialize=MinStableLoad_dict)
model.pDemand = Param(model.times, initialize=demand_dict)


#Wind Capacity factors
model.pWindCapacityfactor1 = Param(model.times, initialize=Win1CF_dict)

# Variables
model.vEnergy = Var(model.generators, model.times, within=NonNegativeReals)

# Objective function
def objFunc(model):
    return sum((model.pHeatRate[f] * model.pFuelPrice[f] + model.pVariableOM[f]) \
               * model.vEnergy[f, t] for f in model.generators for t in model.times)
model.vOC = Objective(rule=objFunc, sense=minimize)

# Constraints
def meetDemand(model, t):
    return sum(model.vEnergy[f, t] for f in model.generators) >= model.pDemand[t]
model.meetdemand = Constraint(model.times, rule=meetDemand)

def underMaxCapacity(model, f, t):
    if f in model.wind_generators:
        return model.vEnergy[f, t] <= model.pMaxCapacity[f] * model.pWindCapacityfactor1[t]
    else:
        return model.vEnergy[f, t] <= model.pMaxCapacity[f]

model.maxcap = Constraint(model.generators, model.times, rule=underMaxCapacity)
   
# Solve the model
solver = SolverFactory('glpk')  
results = solver.solve(model)

# Print model
model.pprint()

# Display results
if results.solver.termination_condition == TerminationCondition.optimal:
    print("Optimal solution found.")
    print(f"Total Cost: ${model.vOC():.2f}")

    print('Purchase decisions:')
    for t in model.times:
        for f in model.generators:
            print(f"Time{t}, Generator{f}: {model.vEnergy[f, t].value} ")    

else:
    print("Solver did not find an optimal solution.")
    
import matplotlib.pyplot as plt
import numpy as np

# Lists to store energy values for each type of generator
coal_energy = []
wind_energy = []
non_coal_energy = []

# Time periods
x_range = range(1, 25)

# Extract energy by type
for t in x_range:
    coal_gen_energy = sum(model.vEnergy[f, t].value for f in model.coal_generators)
    wind_gen_energy = sum(model.vEnergy[f, t].value for f in model.wind_generators)
    non_coal_gen_energy = sum(model.vEnergy[f, t].value for f in model.non_coal_generators)

    coal_energy.append(coal_gen_energy)
    wind_energy.append(wind_gen_energy)
    non_coal_energy.append(non_coal_gen_energy)

# Convertion to numpy arrays
coal_energy = np.array(coal_energy)
wind_energy = np.array(wind_energy)
non_coal_energy = np.array(non_coal_energy)

# Cumulative generation
L0_range = np.zeros(len(x_range))  
L1_range = L0_range + wind_energy  
L2_range = L1_range + coal_energy  
L3_range = L2_range + non_coal_energy

# Plot of the energy generation as a stacked area chart
fig, ax = plt.subplots(figsize=(10, 6))

ax.fill_between(x_range, L0_range, L1_range, label='Wind Power', color='#00FF00', alpha=0.6)
ax.fill_between(x_range, L1_range, L2_range, label='Coal and Green Ammonia', color='#00FFFF', alpha=0.6)
ax.fill_between(x_range, L2_range, L3_range, label='Hydropower and Natural Gas', color='#CCFF00', alpha=0.6)
y_max = max(L3_range)
plt.ylim(0, y_max * 1.1)  

# Labels and title
plt.xlabel('Time (Hour)')
plt.ylabel('Power Generation (MW)')
plt.title('Power Generation by Source in West Virginia (Fall 2030)')
plt.legend(loc='upper left')

# Show the plot
plt.grid(True)
plt.show()

# Dictionaries to storage total power and cost
total_cost_coal = 0
total_cost_wind = 0
total_cost_non_coal = 0

total_gen_coal = 0
total_gen_wind = 0
total_gen_non_coal = 0

for t in x_range:
    for f in model.coal_generators:
        total_cost_coal += (model.pHeatRate[f] * model.pFuelPrice[f] + model.pVariableOM[f]) * model.vEnergy[f, t].value
        total_gen_coal += model.vEnergy[f, t].value

    for f in model.wind_generators:
        total_cost_wind += (model.pHeatRate[f] * model.pFuelPrice[f] + model.pVariableOM[f]) * model.vEnergy[f, t].value
        total_gen_wind += model.vEnergy[f, t].value

    for f in model.non_coal_generators:
        total_cost_non_coal += (model.pHeatRate[f] * model.pFuelPrice[f] + model.pVariableOM[f]) * model.vEnergy[f, t].value
        total_gen_non_coal += model.vEnergy[f, t].value

print(f"Total Cost: ${model.vOC():.2f}")
print("Cost Breakdown by Fuel Type:")
print(f"Coal Generators: Total Cost = ${total_cost_coal:.2f}, Total Generation = {total_gen_coal:.2f} MW")
print(f"Wind Generators: Total Cost = ${total_cost_wind:.2f}, Total Generation = {total_gen_wind:.2f} MW")
print(f"Non-Coal Generators: Total Cost = ${total_cost_non_coal:.2f}, Total Generation = {total_gen_non_coal:.2f} MW")