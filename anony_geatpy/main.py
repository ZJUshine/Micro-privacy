# -*- coding: utf-8 -*-
from MyProblem import MyProblem  # 导入自定义问题接口
import geatpy as ea  # import geatpy

if __name__ == '__main__':
    # 实例化问题对象
    problem = MyProblem()
    # 构建不同多目标优化算法
    # algorithm = ea.moea_MOEAD_archive_templet(
    # algorithm = ea.moea_NSGA3_templet(
    algorithm = ea.moea_NSGA2_templet(
        problem,
        ea.Population(Encoding='BG', NIND=30),
        MAXGEN=30,  # 最大进化代数
        logTras=1)  # 表示每隔多少代记录一次日志信息，0表示不记录。
    # 求解
    res = ea.optimize(algorithm,
                      verbose=True,
                      drawing=1,
                      outputMsg=True,
                      drawLog=True,
                      saveFlag=True)