import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dt_ppo=pd.read_csv('PPO.csv')
dt_a3c=pd.read_csv('A3C.csv')
dt_dqn=pd.read_csv('output_10.csv')

rl_ppo=dt_ppo["PPO"]
rd_ppo=dt_ppo["Random"]
gd_ppo=dt_ppo["Greedy"]
x_ppo=dt_ppo.index

rl_dqn=dt_dqn["RL"]
rd_dqn=dt_dqn["Random"]
gd_dqn=dt_dqn["Greedy"]
x_dqn=dt_dqn.index

rl_a3c=dt_a3c["A3C"]
rd_a3c=dt_a3c["Random"]
gd_a3c=dt_a3c["Greedy"]
x_a3c=dt_a3c.index

plt.figure()
plt.plot(x_ppo,rl_ppo,'g')
plt.plot(x_ppo,rd_ppo,'gray')
plt.plot(x_ppo,gd_ppo,'k')
plt.legend(['PPO',"Random","Greedy"], loc='lower right')
plt.xlabel("time")
plt.ylabel("score")
plt.title("Performance")
plt.savefig("compare_PPO")

plt.figure()
plt.plot(x_dqn,rl_dqn,'r')
plt.plot(x_dqn,rd_dqn,'gray')
plt.plot(x_dqn,gd_dqn,'k')
plt.legend(['DQN',"Random","Greedy"], loc='lower right')
plt.xlabel("time")
plt.ylabel("score")
plt.title("Performance")
plt.savefig("compare_DQN")

plt.figure()
plt.plot(x_a3c,rl_a3c,'b')
plt.plot(x_a3c,rd_a3c,'gray')
plt.plot(x_a3c,gd_a3c,'k')
plt.legend(['A3C',"Random","Greedy"], loc='lower right')
plt.xlabel("time")
plt.ylabel("score")
plt.title("Performance")
plt.savefig("compare_A3C")

y_ppo=(rl_ppo-rd_ppo)
t_ppo,f_ppo=[i for i in y_ppo if i >=0],[i for i in y_ppo if i <0]
y_dqn=(rl_dqn-rd_dqn)
t_dqn,f_dqn=[i for i in y_dqn if i >=0],[i for i in y_dqn if i <0]
y_a3c=(rl_a3c-rd_a3c)
t_a3c,f_a3c=[i for i in y_a3c if i >=0],[i for i in y_a3c if i <0]
# print(len(t)/len(f))
plt.figure()
plt.plot(x_ppo,y_ppo,'g')
plt.xlabel("time")
plt.ylabel("score")
plt.title("Relative performance between DQN and Random model")
plt.savefig("PPO-RD")

plt.figure()
plt.plot(x_dqn,y_dqn,'r')
plt.xlabel("time")
plt.ylabel("score")
plt.title("Relative performance between DQN and Random model")
plt.savefig("DQN-RD")

plt.figure()
plt.plot(x_a3c,y_a3c,'r')
plt.xlabel("time")
plt.ylabel("score")
plt.title("Relative performance between A3C and Random model")
plt.savefig("A3C-RD")

plt.figure()
plt.plot(x_ppo,y_ppo,'g')
plt.plot(x_dqn,y_dqn,'r')
plt.plot(x_a3c,y_a3c,'b')
plt.xlabel("time")
plt.ylabel("score")
plt.legend(['PPO',"DQN", 'A3C'], loc='lower right')
plt.title("Relative performance")
plt.savefig("Rp")

y_ppo=(rl_ppo-rd_ppo)/(gd_ppo-rd_ppo)*100
y_dqn=(rl_dqn-rd_dqn)/(gd_dqn-rd_dqn)*100
y_a3c=(rl_a3c-rd_a3c)/(gd_a3c-rd_a3c)*100
plt.figure()
plt.plot(x_ppo,y_ppo,'g')
plt.xlabel("time")
plt.ylabel("score(%)")
plt.title("Relative performance between PPO and Random model (%)")
plt.savefig("PPO%")

plt.figure()
plt.plot(x_dqn,y_dqn,'r')
plt.xlabel("time")
plt.ylabel("score(%)")
plt.title("Relative performance between DQN and Random model (%)")
plt.savefig("DQN%")

plt.figure()
plt.plot(x_a3c,y_a3c,'b')
plt.xlabel("time")
plt.ylabel("score(%)")
plt.title("Relative performance between A3C and Random model (%)")
plt.savefig("A3C%")

plt.figure()
plt.plot(x_ppo,y_ppo,'g')
plt.plot(x_dqn,y_dqn,'r')
plt.plot(x_a3c,y_a3c,'b')
plt.xlabel("time")
plt.ylabel("score(%)")
plt.legend(['PPO',"DQN", 'A3C'], loc='lower right')
plt.title("Relative performance (%)")
plt.savefig("%")


dt_step=pd.read_csv('step.csv')

rl_ppo=dt_step["ppo"]
rd_ppo=dt_step["random_ppo"]
gd_ppo=dt_step["greedy_ppo"]
step=dt_step['step']

rl_dqn=dt_step["dqn"]
rd_dqn=dt_step["random_dqn"]
gd_dqn=dt_step["greedy_dqn"]

rl_a3c=dt_step["a3c"]
rd_a3c=dt_step["random_a3c"]
gd_a3c=dt_step["greedy_a3c"]

plt.figure()
plt.plot(step,rl_ppo,'g')
plt.plot(step,rd_ppo,'gray')
plt.plot(step,gd_ppo,'k')
plt.legend(['PPO',"Random","Greedy"], loc='lower right')
plt.xlabel("step")
plt.ylabel("score")
plt.title("Performance")
plt.savefig("step_PPO")

plt.figure()
plt.plot(step,rl_dqn,'r')
plt.plot(step,rd_dqn,'gray')
plt.plot(step,gd_dqn,'k')
plt.legend(['DQN',"Random","Greedy"], loc='lower right')
plt.xlabel("step")
plt.ylabel("score")
plt.title("Performance")
plt.savefig("step_DQN")

plt.figure()
plt.plot(step,rl_a3c,'b')
plt.plot(step,rd_a3c,'gray')
plt.plot(step,gd_a3c,'k')
plt.legend(['A3C',"Random","Greedy"], loc='lower right')
plt.xlabel("step")
plt.ylabel("score")
plt.title("Performance")
plt.savefig("step_A3C")

y_ppo=(rl_ppo-rd_ppo)/step
y_dqn=(rl_dqn-rd_dqn)/step
y_a3c=(rl_a3c-rd_a3c)/step

plt.figure()
plt.plot(step,y_ppo,'g')
plt.plot(step,y_dqn,'r')
plt.plot(step,y_a3c,'b')
plt.xlabel("step")
plt.ylabel("score / step")
plt.legend(['PPO',"DQN", 'A3C'], loc='lower right')
plt.title("Relative performance step")
plt.savefig("step rp")

y_ppo=(rl_ppo-rd_ppo)/(gd_ppo-rd_ppo)*100
y_dqn=(rl_dqn-rd_dqn)/(gd_dqn-rd_dqn)*100
y_a3c=(rl_a3c-rd_a3c)/(gd_a3c-rd_a3c)*100

plt.figure()
plt.plot(step,y_ppo,'g')
plt.plot(step,y_dqn,'r')
plt.plot(step,y_a3c,'b')
plt.xlabel("step")
plt.ylabel("score(%)")
plt.legend(['PPO',"DQN", 'A3C'], loc='lower right')
plt.title("Relative performance (%)")
plt.savefig("step%")