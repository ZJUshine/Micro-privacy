# -*- coding: utf-8 -*-
from MyProblem_wandb import MyProblem  # 导入自定义问题接口
import geatpy as ea  # import geatpy
import wandb
from datetime import datetime
if __name__ == '__main__':
    _date = '{}'.format(datetime.now().strftime("%m%d"))
    now = '{}'.format(datetime.now().strftime("%H%M"))
    arg_Encoding='RI'
    arg_NIND = 15  # 种群规模
    arg_MAXGEN = 15  # 最大进化代数
    run = wandb.init(
    project="geatpy_moea_MOEAD_archive_templet",
    name=f'Encoding:{arg_Encoding}_NIND:{arg_NIND}_MAXGEN:{arg_MAXGEN}_{_date}_{now}',
    )
    # 实例化问题对象
    problem = MyProblem()
    # 构建算法
    algorithm = ea.moea_MOEAD_archive_templet(
        problem,
        ea.Population(Encoding=arg_Encoding, NIND=arg_NIND),
        MAXGEN=arg_MAXGEN,  # 最大进化代数
        logTras=1)  # 表示每隔多少代记录一次日志信息，0表示不记录。
    # 求解
    res = ea.optimize(algorithm,
                      verbose=True,
                      drawing=1,
                      outputMsg=True,
                      drawLog=True,
                      saveFlag=True)
    print(res)