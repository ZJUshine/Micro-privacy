'''
FilePath: anony_mopso.py
Author: zjushine
Date: 2023-04-06 13:53:14
LastEditors: zjushine
LastEditTime: 2023-04-23 14:23:43
Description: mospo 优化问题
Copyright (c) 2023 by ${zjushine}, All Rights Reserved. 
'''
from jmetal.algorithm.multiobjective.omopso import OMOPSO
from jmetal.operator import UniformMutation
from jmetal.operator.mutation import NonUniformMutation
from jmetal.problem import ZDT1
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.solution import (
    print_function_values_to_file,
    print_variables_to_file,
    read_solutions,
)
from define_mopso_problem import omopso

problem = omopso()

algorithm = OMOPSO(
    problem=problem,
    swarm_size=5,
    epsilon=0.0075,
    uniform_mutation=UniformMutation(probability=1.0/3, perturbation=0.5),  # probability = 1/number_of_variables
    non_uniform_mutation=NonUniformMutation(
        1.0/3, perturbation=0.5, max_iterations = 250
    ),  # max_iterations=max_evaluations / swarm_size
    leaders=CrowdingDistanceArchive(10),
    termination_criterion=StoppingByEvaluations(max_evaluations=5),
)

algorithm.run()
solutions = algorithm.get_result()
# save results to file
print_function_values_to_file(solutions, '/home/lxc/zero/Micro-privacy/omopso/dataFUN.' + problem.get_name())
print_variables_to_file(solutions, '/home/lxc/zero/Micro-privacy/omopso/dataVAR.' + problem.get_name())

print(f"Algorithm: {algorithm.get_name()}")
print(f"Problem: {problem.get_name()}")
print(f"Computing time: {algorithm.total_computing_time}")

# plot results
from jmetal.lab.visualization.plotting import Plot
from jmetal.util.solution import get_non_dominated_solutions

front = get_non_dominated_solutions(solutions)

plot_front = Plot(title='Pareto front approximation', axis_labels=['x', 'y'])
plot_front.plot(front, label='OMOPSO-ZDT1',filename="OMPPSO-ZDT1",format="png")

