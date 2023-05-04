'''
FilePath: anony_mopso.py
Author: zjushine
Date: 2023-04-06 13:53:14
LastEditors: zjushine
LastEditTime: 2023-04-29 23:04:05
Description: mospo 优化问题
Copyright (c) 2023 by ${zjushine}, All Rights Reserved. 
'''
from jmetal.algorithm.multiobjective.omopso import OMOPSO
from jmetal.operator import UniformMutation
from jmetal.operator.mutation import NonUniformMutation
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.solution import (
    print_function_values_to_file,
    print_variables_to_file
)
from define_mopso_problem import omopso
from datetime import datetime

_date = '{}'.format(datetime.now().strftime("%m%d"))
now = '{}'.format(datetime.now().strftime("%H%M"))

results_output_path = f"results/{_date}_{now}"

problem = omopso()
mutation_probability = 1.0 / problem.number_of_variables
max_evaluations = 200
swarm_size = 40

algorithm = OMOPSO(
    problem=problem,
    swarm_size = swarm_size,
    epsilon=0.75,
    uniform_mutation=UniformMutation(probability=mutation_probability, perturbation=0.2),
    non_uniform_mutation=NonUniformMutation(probability=mutation_probability, perturbation=0.2, max_iterations = int(max_evaluations / swarm_size)),
    leaders=CrowdingDistanceArchive(swarm_size),
    termination_criterion=StoppingByEvaluations(max_evaluations = max_evaluations),
)

algorithm.run()
solutions = algorithm.get_result()
# save results to file
print_function_values_to_file(solutions, f'{results_output_path}/dataFUN.' + problem.get_name())
print_variables_to_file(solutions, f'{results_output_path}/dataVAR.' + problem.get_name())


print(f"Algorithm: {algorithm.get_name()}")
print(f"Problem: {problem.get_name()}")
print(f"Computing time: {algorithm.total_computing_time}")

# plot results
from jmetal.lab.visualization.plotting import Plot
from jmetal.util.solution import get_non_dominated_solutions

front = get_non_dominated_solutions(solutions)

plot_front = Plot(title='Pareto front approximation', axis_labels=['x', 'y'])
plot_front.plot(front, label='OMOPSO-Micro_mopso',filename=f"{results_output_path}/OMOPSO-Micro_mopso",format="png")

