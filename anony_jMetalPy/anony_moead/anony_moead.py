'''
FilePath: anony_moead.py
Author: zjushine
Date: 2023-05-03 13:59:18
LastEditors: zjushine
LastEditTime: 2023-05-05 16:11:03
Description: 使用moead算法优化系数
Copyright (c) 2023 by ${zjushine}, All Rights Reserved. 
'''
from jmetal.algorithm.multiobjective.moead import MOEAD
from jmetal.operator import PolynomialMutation, DifferentialEvolutionCrossover
from jmetal.util.aggregative_function import Tschebycheff
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.solution import (
    print_function_values_to_file,
    print_variables_to_file
)
from define_moead_problem import moead
from datetime import datetime
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--max_evaluations', type=int, default=200, help='最大迭代次数')
parser.add_argument('--opt_target', type=str, default="both", help= "优化目标，可选项为：'stoi' 'wer' 'both'")
args = parser.parse_args()

_date = '{}'.format(datetime.now().strftime("%m%d"))
now = '{}'.format(datetime.now().strftime("%H%M"))

results_output_path = f"results/{_date}_{now}"
os.makedirs(results_output_path)
problem = moead(opt_target = args.opt_target)
max_evaluations = args.max_evaluations

with open(f'{results_output_path}/data.txt', "a") as f:
        f.write(f"max_evaluations:{args.max_evaluations},opt_target:{args.opt_target}\n")

algorithm = MOEAD(
    problem=problem,
    population_size=300,
    crossover=DifferentialEvolutionCrossover(CR=1.0, F=0.5, K=0.5),
    mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
    aggregative_function=Tschebycheff(dimension=problem.number_of_objectives),
    neighbor_size=20,
    neighbourhood_selection_probability=0.9,
    max_number_of_replaced_solutions=2,
    weight_files_path="resources/MOEAD_weights",
    termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
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
plot_front.plot(front, label='moead-Micro_moead',filename=f"{results_output_path}/moead-Micro_moead",format="png")

