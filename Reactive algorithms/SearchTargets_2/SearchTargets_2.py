from Agent import *
from Sensor import *
from Target import *
from random import *
from math import *
from numpy import *
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
from itertools import product
import datetime as dt

import cv2
import os

newpath = r'C:\Users\barouch\Desktop\SearchVideo' 
if not os.path.exists(newpath):
    os.makedirs(newpath)
file=open("testfile.txt","w+")

seed(7)



N=10 # Matrix size
T=400

pr_static=100

#setting sensors
s0=Sensor(1.0,15.0,1.0)
#s1=Sensor(1.0,10.0,1.0)
#s2=Sensor(1.0,10.0,1.0)
#s3=Sensor(1.0,10.0,1.0)


#setting agents
#a=[Agent(randint(0,N-1),randint(0,N-1)) for i in range(NumberOfAgents)]
#a=[Agent(5,5),Agent(8,8),Agent(62,62)]#,Agent(4,4),Agent(10,10)]#,Agent(10,7),Agent(4,4),Agent(10,10)]
a0=Agent(x_pos= 10,y_pos= 4, sensors=[s0])
#a1=Agent(x_pos= 3,y_pos= 3, sensors=[s2,s3])

agents_arr=[a0]
agents_original_pos=[[agent.x_pos, agent.y_pos] for agent in agents_arr]

NumberOfAgents=len(agents_arr)

for m in range(NumberOfAgents):
       print(m)
       print(agents_arr[m].x_pos)
       print(agents_arr[m].y_pos)


#setting targets
#Targets=[Target(randint(0,N-1),randint(0,N-1)) for i in range(NumberOfTargets)]
targets_arr=[Target(4,0),Target(0,9)]
NumberOfTargets=len(targets_arr)
#seed(5)
#Target event


P_sensor=[[[[[0 for t in range(T)] for k in range(agents_arr[m].NumberOfSensors())] for m in range(NumberOfAgents)] for j in range(N)] for i in range(N)]
P_agent=[[[[0 for t in range(T)]  for m in range(NumberOfAgents)] for j in range(N)] for i in range(N)]
P_global= [[[0 for t in range(T)] for j in range(N)] for i in range(N)]
Entropy_mat_arr=[[[0 for t in range(T)] for j in range(N)] for i in range(N)]
Entropy_arr=[0 for t in range(T)]
P_TA=1.0
alpha=0.1



for i in range(N):
    for j in range(N):
        for m in range(NumberOfAgents):
            for k in range(agents_arr[m].NumberOfSensors()):
                P_sensor[i][j][m][k][0]=1/(N*N)

def calc_P_agent(t):
    mul1=[[[1.0 for m in range(NumberOfAgents)] for j in range(N)] for i in range(N)]
    mul2=[[[1.0 for m in range(NumberOfAgents)] for j in range(N)] for i in range(N)]

    for i in range(N):
        for j in range(N):
            for m in range(NumberOfAgents):
                for k in range(agents_arr[m].NumberOfSensors()):
                    mul1[i][j][m]=mul1[i][j][m]*P_sensor[i][j][m][k][t]
                    mul2[i][j][m]=mul2[i][j][m]*(1-P_sensor[i][j][m][k][t])
                P_agent[i][j][m][t]=mul1[i][j][m]/(mul1[i][j][m]+mul2[i][j][m])
def calc_P_global(t):
    mul1=[[1.0 for j in range(N)] for i in range(N)]
    mul2=[[1.0 for j in range(N)] for i in range(N)]

    for i in range(N):
        for j in range(N):
            for m in range(NumberOfAgents):
                mul1[i][j]=mul1[i][j]*P_agent[i][j][m][t]
                mul2[i][j]=mul2[i][j]*(1-P_agent[i][j][m][t])
            P_global[i][j][t]=mul1[i][j]/(mul1[i][j]+mul2[i][j])

def calc_entropy(t):
    result=0
    for i in range(N):
        for j in range(N):
            Entropy_mat_arr[i][j][t]=P_global[i][j][t]*math.log2(1/P_global[i][j][t])+(1-P_global[i][j][t])*math.log2(1/(1-P_global[i][j][t]))
            result+=Entropy_mat_arr[i][j][t]
    Entropy_arr[t]=result

def isItTarget(x,y):
    for target in targets_arr:
        if(target.x_pos==x and target.y_pos==y):
            return True
    return False

def initialize_agents_locations():
    for m in range(NumberOfAgents):
        agents_arr[m].x_pos=agents_original_pos[m][0]
        agents_arr[m].y_pos=agents_original_pos[m][1]


def dynamic_targets_movement():
    for tar in range(NumberOfTargets):
        rand=uniform(0,1)
        if(rand<pr_static/100):
            continue
        else:
            direction_rand=randint(0,7)
            if(direction_rand>=4):
                direction_rand+=1
            dir_x=direction_rand%3 -1
            dir_y=direction_rand//3 -1

            if(targets_arr[tar].x_pos+dir_x<0 or targets_arr[tar].x_pos+dir_x>=N):
                continue
            if(targets_arr[tar].y_pos+dir_y<0 or targets_arr[tar].y_pos+dir_y>=N):
                continue
            targets_arr[tar].x_pos+=dir_x
            targets_arr[tar].y_pos+=dir_y

def update_sensor_probability(t):
    dynamic_targets_movement()


    temp_mat=[[[[0 for m in range(NumberOfAgents)]for k in range(agents_arr[0].NumberOfSensors())] for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
        #for every every cell:
            for m in range(NumberOfAgents):
                for k in range(agents_arr[m].NumberOfSensors()):
                    directions_sum=0
                    count=0
                    for ind in [(i-1,j-1),(i-1,j),(i-1,j+1),(i,j-1),(i,j+1),(i+1,j-1),(i+1,j),(i+1,j+1)]:
                        if(ind[0]>=0 and ind[0]<=N-1 and ind[1]>=0 and ind[1]<=N-1):
                            count+=1
                            directions_sum+=P_sensor[ind[0]][ind[1]][m][k][t-1]
                    directions_sum+=(8-count)*P_sensor[i][j][m][k][t-1]
                    temp_mat[i][j][k][m]=P_sensor[i][j][m][k][t-1]*pr_static/100+directions_sum*(1-pr_static/100)/8

    for i in range(N):
        for j in range(N):
        #for every every cell:
            for k in range(agents_arr[m].NumberOfSensors()):
                for m in range(NumberOfAgents):
                    P_sensor[i][j][m][k][t-1]=temp_mat[i][j][k][m]



    for i in range(N):
        for j in range(N):
            for m in range(NumberOfAgents):
                r=((agents_arr[m].x_pos-i)**2+(agents_arr[m].y_pos-j)**2)**0.5
                for k in range(agents_arr[m].NumberOfSensors()):
                    exponent=math.exp(-r/agents_arr[m].sensors[k].pow)
                    if(isItTarget(i,j)==True):
                        P_sensor[i][j][m][k][t]=((P_sensor[i][j][m][k][t-1]*P_TA*exponent)/(P_sensor[i][j][m][k][t-1]*(1-alpha)+alpha))+((P_sensor[i][j][m][k][t-1]*(1-exponent)**2)/(P_sensor[i][j][m][k][t-1]*(1-P_TA*exponent)+(1-P_sensor[i][j][m][k][t-1])*(1-alpha*P_TA*exponent)))
                    else:
                        P_sensor[i][j][m][k][t]=((P_sensor[i][j][m][k][t-1]*alpha*P_TA*exponent)/(P_sensor[i][j][m][k][t-1]*(1-alpha)+alpha))+((P_sensor[i][j][m][k][t-1]*(1-exponent)*(1-alpha*P_TA*exponent))/(P_sensor[i][j][m][k][t-1]*(1-P_TA*exponent)+(1-P_sensor[i][j][m][k][t-1])*(1-alpha*P_TA*exponent)))


calc_P_agent(t=0)
calc_P_global(t=0)
calc_entropy(t=0)



def move_agent(agent, direction):
    direction=int(direction)
    if(direction==0):
        return
    if(direction==1):
        agent.y_pos=max(0,agent.y_pos-1)
        return
    if(direction==2):
        agent.x_pos=min(N-1,agent.x_pos+1)
        agent.y_pos=max(0,agent.y_pos-1)
        return
    if(direction==3):
        agent.x_pos=min(N-1,agent.x_pos+1)
        return
    if(direction==4):
        agent.x_pos=min(N-1,agent.x_pos+1)
        agent.y_pos=min(N-1,agent.y_pos+1)
        return
    if(direction==5):
        agent.y_pos=min(N-1,agent.y_pos+1)
        return
    if(direction==6):
        agent.x_pos=max(0,agent.x_pos-1)
        agent.y_pos=min(N-1,agent.y_pos+1)
        return
    if(direction==7):
        agent.x_pos=max(0,agent.x_pos-1)
        return
    if(direction==8):
        agent.x_pos=max(0,agent.x_pos-1)
        agent.y_pos=max(0,agent.y_pos-1)
        return



def agent_next_pos(agent, direction):
    direction=int(direction)
    if(direction==0):
        return agent.x_pos, agent.y_pos
    if(direction==1):
        return agent.x_pos, max(0,agent.y_pos-1)
    if(direction==2):
        return min(N-1,agent.x_pos+1), max(0,agent.y_pos-1)
    if(direction==3):
        return min(N-1,agent.x_pos+1), agent.y_pos
    if(direction==4):
        return min(N-1,agent.x_pos+1), min(N-1,agent.y_pos+1)
    if(direction==5):
        return agent.x_pos, min(N-1,agent.y_pos+1)
    if(direction==6):
        return max(0,agent.x_pos-1), min(N-1,agent.y_pos+1)
    if(direction==7):
        return max(0,agent.x_pos-1), agent.y_pos
    if(direction==8):
        return max(0,agent.x_pos-1), max(0,agent.y_pos-1)


def number_cells_above_p_global(t,p_limit):
    amount=0
    for i in range(N):
        for j in range(N):
            if(P_global[i][j][t]>=p_limit):
                amount+=1
    return amount


def run_over_all_paths():
    directions='012345678'
    entropy_limit=0.3
    p_limit=0.95
    finish_flag=False
    finish_path_agents=0
    finish_path_length=0
    for path_length in range(1,T):
    
        t1=dt.datetime.now()
        for path_all in product(directions,repeat=NumberOfAgents*path_length):
            path_agent=[path_all[m*path_length:(m+1)*path_length] for m in range(NumberOfAgents)]
            a_pos_arr=[]
            #go over all the moves in the path:
            for move_index in range(path_length):
                t=move_index+1
                #move all the agents one step:
                for m in range(NumberOfAgents):
                    move_agent(agents_arr[m],path_agent[m][move_index])
                #update sensor probability:
                #t3=dt.datetime.now()
                update_sensor_probability(t)
                #t4=dt.datetime.now()
                #print(t4-t3)
                calc_P_agent(t)
                calc_P_global(t)
                calc_entropy(t)
            #initialize the agents original positions:
            initialize_agents_locations()

            #check if entropy crossed the limit:
            #if(Entropy_arr[path_length]<=entropy_limit):

            #check how many cells have probability above p_limit
            if(number_cells_above_p_global(t,p_limit)>=NumberOfTargets):
                finish_flag=True
                finish_path_agents=path_agent
                finish_path_length=path_length
                break
            #print(path_agent,P_global[0][9][t])
            #print(path_agent,Entropy_arr[:path_length+1])
            #print(path_agent,Entropy_mat_arr[0][0][:path_length+1])


        t2=dt.datetime.now()
        print(path_length,t2-t1)

        if(finish_flag==True):
            break


    print('finished')
    print('finish_path_agents:',finish_path_agents)
    print('entropy:',Entropy_arr[:finish_path_length+1])
    print('prob 0,0 over time:',P_global[0][9][:finish_path_length+1])

def cell_to_direction(x,y,x_next,y_next):
    x=int(x)
    y=int(y)
    x_next=int(x_next)
    y_next=int(y_next)

    x_rel=x_next-x
    y_rel=y_next-y
    if (x_rel==0 and y_rel==0):
        return '0'
    if (x_rel==0 and y_rel<0):
        return '1'
    if (x_rel>0 and y_rel<0):
        return '2'
    if (x_rel>0 and y_rel==0):
        return '3'
    if (x_rel>0 and y_rel>0):
        return '4'
    if (x_rel==0 and y_rel>0):
        return '5'
    if (x_rel<0 and y_rel>0):
        return '6'
    if (x_rel<0 and y_rel==0):
        return '7'
    if (x_rel<0 and y_rel<0):
        return '8'



targets_agents_pos=[[(255,255,255) for _ in range(N)] for _ in range(N)]
time_arr=[-1 for _ in range(NumberOfTargets)]
def GeneratePlots(t,isSharing):
    gs = gridspec.GridSpec(2, 2)
    plt.ion()
    
    plt.subplot(gs[0,0])
    if(isSharing==True):
        temp=[[0 for _ in range(N)] for _ in range(N)]
        for i in range(N):
            for j in range(N):
                    temp[i][j]=P_global[i][j][t]
        plt.imshow(temp,cmap='gray')
        plt.title("Probabilities matrix at t="+ str(t))
    else:
        temp=[[0 for _ in range(N)] for _ in range(N)]
        for i in range(N):
            for j in range(N):
                for m in range(NumberOfAgents):
                    temp[i][j]=max(temp[i][j],P_agent[i][j][m][t])
        plt.imshow(temp,cmap='gray')
        plt.title("Agents prob matrix at t="+ str(t))

    
    #plt.subplot(gs[0,1])
    #TEV_FA=[[0 for k in range(N)] for j in range(N)] 
    #for i in range(N):
    #    for j in range(N):
    #        for k in range(len(s)):
    #            if(FA[t][i][j][k]==1):
    #                TEV_FA[i][j]=1
    #for tar in range(NumberOfTargets):
    #    for k in range(len(s)):
    #        if(Tev[t][tar][k]==1):
    #            TEV_FA[Targets[tar].x_pos][Targets[tar].y_pos]=1
    #
    #plt.imshow(TEV_FA,cmap='gray')
    #plt.title("Events at t="+ str(t))
    
    
    
    plt.subplot(gs[1,0])
    for m in range(NumberOfAgents):
        targets_agents_pos[agents_arr[m].x_pos][agents_arr[m].y_pos]=(0,255,0) # green
    
    for tar in range(NumberOfTargets):
        targets_agents_pos[targets_arr[tar].x_pos][targets_arr[tar].y_pos]=(255,0,0) # red
    
    plt.imshow(targets_agents_pos)
    plt.title("Agents-Blue, Targets-Red ")
    for tar in range(NumberOfTargets):
        targets_agents_pos[targets_arr[tar].x_pos][targets_arr[tar].y_pos]=(255,255,255) # reset red
    
    
    
    #time figure
    plt.subplot(gs[1,1])
    plt.ylim((0,T))
    plt.bar(range(NumberOfTargets),time_arr,color="blue")
    plt.xticks(range(NumberOfTargets),["("+str(targets_arr[tar].x_pos)+","+str(targets_arr[tar].y_pos)+")" for tar in range(NumberOfTargets)])
    for tar in range(NumberOfTargets):
        if(time_arr[tar]!=-1):
            plt.text(tar,time_arr[tar]+3,"t="+str(time_arr[tar]))
    plt.xlabel("Targets positions")
    plt.ylabel("Time")
    plt.title("Time to targets")
    
    
    
    
    plt.subplots_adjust(wspace=0.5,hspace=0.5)
    plt.pause(0.01)
    #plt.show()
    
    
    plt.draw()  
    plt.savefig(newpath+'/fig'+str(t)+'.png', bbox_inches='tight')
    plt.clf()
    
    for m in range(NumberOfAgents):
        targets_agents_pos[agents_arr[m].x_pos][agents_arr[m].y_pos]=(0,0,255)



def algorithm1_EIG_global_local_action():
    directions='012345678'
    p_limit=0.95
    finish_path_agents=0 #[agent0_path, agent1_path,...]
    finish_path_length=0

    for t in range(1,T):
        best_direction_agents=[0 for i in range(NumberOfAgents)]
        for m in range(NumberOfAgents):
            max_EIG_agent=0
            for direction in directions:
                EIG_agent=0
                for i in range(N):
                    for j in range(N):
                        r_prev=((agents_arr[m].x_pos-i)**2+(agents_arr[m].y_pos-j)**2)**0.5
                        r_next=((agent_next_pos(agents_arr[m],direction)[0]-i)**2+(agent_next_pos(agents_arr[m],direction)[1]-j)**2)**0.5
                        for k in range(agents_arr[m].NumberOfSensors()):
                            estimated_p_sensor=P_global[i][j][t-1]*P_TA*math.exp(-r_next/agents_arr[m].sensors[k].pow)
                            estimated_p_prime_sensor=P_global[i][j][t-1]*P_TA*math.exp(-r_prev/agents_arr[m].sensors[k].pow)
                            EIG_agent+=estimated_p_sensor*math.log2(estimated_p_sensor/estimated_p_prime_sensor)
                if(EIG_agent>max_EIG_agent):
                    max_EIG_agent=EIG_agent
                    best_direction_agents[m]=direction
        #at this point we know the best directions for all the agents
        for m in range(NumberOfAgents):
            move_agent(agents_arr[m],best_direction_agents[m])

        update_sensor_probability(t)
        #t4=dt.datetime.now()
        #print(t4-t3)
        calc_P_agent(t)
        calc_P_global(t)
        calc_entropy(t)

        if(finish_path_length==0):
            finish_path_agents=[[best_direction_agents[m]] for m in range(NumberOfAgents)]
            finish_path_length=1
        else:
            finish_path_length+=1
            for m in range(NumberOfAgents):
                finish_path_agents[m].append(best_direction_agents[m])



        for tar in range(NumberOfTargets):
            if(P_global[targets_arr[tar].x_pos][targets_arr[tar].y_pos][t]>p_limit and time_arr[tar]<0):
                time_arr[tar]=t
  
        GeneratePlots(t,isSharing=True)


        if(number_cells_above_p_global(t,p_limit)>=NumberOfTargets):
            break


    print('finished')
    print('finish_path_agents:',finish_path_agents)
    print('entropy:',Entropy_arr[:finish_path_length+1])
    print('prob 0,0 over time:',P_global[0][0][:finish_path_length+1])

def algorithm2_EIG_agent_local_action():
    directions='012345678'
    p_limit=0.95
    finish_path_agents=0 #[agent0_path, agent1_path,...]
    finish_path_length=0

    for t in range(1,T):
        best_direction_agents=[0 for i in range(NumberOfAgents)]
        for m in range(NumberOfAgents):
            max_EIG_agent=0
            for direction in directions:
                EIG_agent=0
                for i in range(N):
                    for j in range(N):
                        r_prev=((agents_arr[m].x_pos-i)**2+(agents_arr[m].y_pos-j)**2)**0.5
                        r_next=((agent_next_pos(agents_arr[m],direction)[0]-i)**2+(agent_next_pos(agents_arr[m],direction)[1]-j)**2)**0.5
                        for k in range(agents_arr[m].NumberOfSensors()):
                            estimated_p_sensor=P_agent[i][j][m][t-1]*P_TA*math.exp(-r_next/agents_arr[m].sensors[k].pow)
                            estimated_p_prime_sensor=P_agent[i][j][m][t-1]*P_TA*math.exp(-r_prev/agents_arr[m].sensors[k].pow)
                            EIG_agent+=estimated_p_sensor*math.log2(estimated_p_sensor/estimated_p_prime_sensor)
                if(EIG_agent>max_EIG_agent):
                    max_EIG_agent=EIG_agent
                    best_direction_agents[m]=direction
        #at this point we know the best directions for all the agents
        for m in range(NumberOfAgents):
            move_agent(agents_arr[m],best_direction_agents[m])
        update_sensor_probability(t)
        #t4=dt.datetime.now()
        #print(t4-t3)
        calc_P_agent(t)
        calc_P_global(t)
        calc_entropy(t)

        if(finish_path_length==0):
            finish_path_agents=[[best_direction_agents[m]] for m in range(NumberOfAgents)]
            finish_path_length=1
        else:
            finish_path_length+=1
            for m in range(NumberOfAgents):
                finish_path_agents[m].append(best_direction_agents[m])

        
        for tar in range(NumberOfTargets):
            if(P_global[targets_arr[tar].x_pos][targets_arr[tar].y_pos][t]>p_limit and time_arr[tar]<0):
                time_arr[tar]=t

        GeneratePlots(t,isSharing=True)        
                
        if(number_cells_above_p_global(t,p_limit)>=NumberOfTargets):
            break


    print('finished')
    print('finish_path_agents:',finish_path_agents)
    print('entropy:',Entropy_arr[:finish_path_length+1])
    print('prob 0,0 over time:',P_global[0][0][:finish_path_length+1])

def algorithm3_EIG_global_global_action():
    directions='012345678'
    p_limit=0.95
    finish_path_agents=0 #[agent0_path, agent1_path,...]
    finish_path_length=0

    for t in range(1,T):
        best_direction_agents=[0 for i in range(NumberOfAgents)]
        max_EIG_global=0
        for direction_agents in product(directions,repeat=NumberOfAgents):
            EIG_global=0
            for m in range(NumberOfAgents):
                direction=direction_agents[m]
                for i in range(N):
                    for j in range(N):
                        r_prev=((agents_arr[m].x_pos-i)**2+(agents_arr[m].y_pos-j)**2)**0.5
                        r_next=((agent_next_pos(agents_arr[m],direction)[0]-i)**2+(agent_next_pos(agents_arr[m],direction)[1]-j)**2)**0.5
                        for k in range(agents_arr[m].NumberOfSensors()):
                            estimated_p_sensor=P_global[i][j][t-1]*P_TA*math.exp(-r_next/agents_arr[m].sensors[k].pow)
                            estimated_p_prime_sensor=P_global[i][j][t-1]*P_TA*math.exp(-r_prev/agents_arr[m].sensors[k].pow)
                            EIG_global+=estimated_p_sensor*math.log2(estimated_p_sensor/estimated_p_prime_sensor)
            if(EIG_global>max_EIG_global):
                max_EIG_global=EIG_global
                best_direction_agents=direction_agents
        #at this point we know the best directions for all the agents
        for m in range(NumberOfAgents):
            move_agent(agents_arr[m],best_direction_agents[m])
        update_sensor_probability(t)
        #t4=dt.datetime.now()
        #print(t4-t3)
        calc_P_agent(t)
        calc_P_global(t)
        calc_entropy(t)

        if(finish_path_length==0):
            finish_path_agents=[[best_direction_agents[m]] for m in range(NumberOfAgents)]
            finish_path_length=1
        else:
            finish_path_length+=1
            for m in range(NumberOfAgents):
                finish_path_agents[m].append(best_direction_agents[m])

        for tar in range(NumberOfTargets):
            if(P_global[targets_arr[tar].x_pos][targets_arr[tar].y_pos][t]>p_limit and time_arr[tar]<0):
                time_arr[tar]=t

        GeneratePlots(t,isSharing=True)
                
        if(number_cells_above_p_global(t,p_limit)>=NumberOfTargets):
            break


    print('finished')
    print('finish_path_agents:',finish_path_agents)
    print('entropy:',Entropy_arr[:finish_path_length+1])
    print('prob 0,0 over time:',P_global[0][0][:finish_path_length+1])


def algorithm4_COV_global_local_action():
    directions='012345678'
    p_limit=0.95
    finish_path_agents=0 #[agent0_path, agent1_path,...]
    finish_path_length=0

    for t in range(1,T):
        best_direction_agents=[0 for i in range(NumberOfAgents)]
        for m in range(NumberOfAgents):
            max_EIG_agent=0
            for i_next in range(N):
                for j_next in range(N):
                    EIG_agent=0
                    for i in range(N):
                        for j in range(N):
                            r_prev=((agents_arr[m].x_pos-i)**2+(agents_arr[m].y_pos-j)**2)**0.5
                            r_next=((i_next-i)**2+(j_next-j)**2)**0.5
                            for k in range(agents_arr[m].NumberOfSensors()):
                                estimated_p_sensor=P_global[i][j][t-1]*P_TA*math.exp(-r_next/agents_arr[m].sensors[k].pow)
                                estimated_p_prime_sensor=P_global[i][j][t-1]*P_TA*math.exp(-r_prev/agents_arr[m].sensors[k].pow)
                                EIG_agent+=estimated_p_sensor*math.log2(estimated_p_sensor/estimated_p_prime_sensor)
                    if(EIG_agent>max_EIG_agent):
                        max_EIG_agent=EIG_agent
                        best_direction_agents[m]=cell_to_direction(agents_arr[m].x_pos,agents_arr[m].y_pos,i_next,j_next)
                        print('t:',t,'agent:',m, '    i,j:',i_next,j_next, '     EIG:',EIG_agent)
        #at this point we know the best directions for all the agents
        for m in range(NumberOfAgents):
            move_agent(agents_arr[m],best_direction_agents[m])
        update_sensor_probability(t)
        #t4=dt.datetime.now()
        #print(t4-t3)
        calc_P_agent(t)
        calc_P_global(t)
        calc_entropy(t)

        if(finish_path_length==0):
            finish_path_agents=[[best_direction_agents[m]] for m in range(NumberOfAgents)]
            finish_path_length=1
        else:
            finish_path_length+=1
            for m in range(NumberOfAgents):
                finish_path_agents[m].append(best_direction_agents[m])

        for tar in range(NumberOfTargets):
            if(P_global[targets_arr[tar].x_pos][targets_arr[tar].y_pos][t]>p_limit and time_arr[tar]<0):
                time_arr[tar]=t

        GeneratePlots(t,isSharing=True)

        if(number_cells_above_p_global(t,p_limit)>=NumberOfTargets):
            break


    print('finished')
    print('finish_path_agents:',finish_path_agents)
    print('entropy:',Entropy_arr[:finish_path_length+1])
    print('prob 0,0 over time:',P_global[0][0][:finish_path_length+1])

def algorithm5_COV_agent_local_action():
    directions='012345678'
    p_limit=0.95
    finish_path_agents=0 #[agent0_path, agent1_path,...]
    finish_path_length=0

    for t in range(1,T):
        best_direction_agents=[0 for i in range(NumberOfAgents)]
        for m in range(NumberOfAgents):
            max_EIG_agent=0
            for i_next in range(N):
                for j_next in range(N):
                    EIG_agent=0
                    for i in range(N):
                        for j in range(N):
                            r_prev=((agents_arr[m].x_pos-i)**2+(agents_arr[m].y_pos-j)**2)**0.5
                            r_next=((i_next-i)**2+(j_next-j)**2)**0.5
                            for k in range(agents_arr[m].NumberOfSensors()):
                                estimated_p_sensor=P_agent[i][j][m][t-1]*P_TA*math.exp(-r_next/agents_arr[m].sensors[k].pow)
                                estimated_p_prime_sensor=P_agent[i][j][m][t-1]*P_TA*math.exp(-r_prev/agents_arr[m].sensors[k].pow)
                                EIG_agent+=estimated_p_sensor*math.log2(estimated_p_sensor/estimated_p_prime_sensor)
                    if(EIG_agent>max_EIG_agent):
                        max_EIG_agent=EIG_agent
                        best_direction_agents[m]=cell_to_direction(agents_arr[m].x_pos,agents_arr[m].y_pos,i_next,j_next)
                        print('t:',t,'agent:',m, '    i,j:',i_next,j_next, '     EIG:',EIG_agent)
        #at this point we know the best directions for all the agents
        for m in range(NumberOfAgents):
            move_agent(agents_arr[m],best_direction_agents[m])
        update_sensor_probability(t)
        #t4=dt.datetime.now()
        #print(t4-t3)
        calc_P_agent(t)
        calc_P_global(t)
        calc_entropy(t)

        if(finish_path_length==0):
            finish_path_agents=[[best_direction_agents[m]] for m in range(NumberOfAgents)]
            finish_path_length=1
        else:
            finish_path_length+=1
            for m in range(NumberOfAgents):
                finish_path_agents[m].append(best_direction_agents[m])
        for tar in range(NumberOfTargets):
            if(P_global[targets_arr[tar].x_pos][targets_arr[tar].y_pos][t]>p_limit and time_arr[tar]<0):
                time_arr[tar]=t

        GeneratePlots(t,isSharing=True)
        if(number_cells_above_p_global(t,p_limit)>=NumberOfTargets):
            break


    print('finished')
    print('finish_path_agents:',finish_path_agents)
    print('entropy:',Entropy_arr[:finish_path_length+1])
    print('prob 0,0 over time:',P_global[0][0][:finish_path_length+1])

def algorithm6_COV_global_global_action():
    directions='012345678'
    p_limit=0.95
    finish_path_agents=0 #[agent0_path, agent1_path,...]
    finish_path_length=0

    for t in range(1,T):
        best_direction_agents=[0 for i in range(NumberOfAgents)]
        max_EIG_global=0
        for i_next_agents in product(range(N),repeat=NumberOfAgents):
            for j_next_agents in product(range(N),repeat=NumberOfAgents):
                EIG_global=0
                for m in range(NumberOfAgents):
                    for i in range(N):
                        for j in range(N):
                            r_prev=((agents_arr[m].x_pos-i)**2+(agents_arr[m].y_pos-j)**2)**0.5
                            r_next=((i_next_agents[m]-i)**2+(j_next_agents[m]-j)**2)**0.5
                            
                            for k in range(agents_arr[m].NumberOfSensors()):
                                estimated_p_sensor=NumberOfTargets*P_TA*math.exp(-r_next/agents_arr[m].sensors[k].pow)
                                estimated_p_prime_sensor=P_global[i][j][t-1]*P_TA*math.exp(-r_prev/agents_arr[m].sensors[k].pow)
                                EIG_global+=estimated_p_sensor*math.log2(estimated_p_sensor/estimated_p_prime_sensor)
                                
                if(EIG_global>max_EIG_global):
                    max_EIG_global=EIG_global
                    best_direction_agents=[cell_to_direction(agents_arr[m].x_pos,agents_arr[m].y_pos,i_next_agents[m],j_next_agents[m]) for m in range(NumberOfAgents)]
        #at this point we know the best directions for all the agents
        for m in range(NumberOfAgents):
            move_agent(agents_arr[m],best_direction_agents[m])
        update_sensor_probability(t)
        #t4=dt.datetime.now()
        #print(t4-t3)
        calc_P_agent(t)
        calc_P_global(t)
        calc_entropy(t)

        if(finish_path_length==0):
            finish_path_agents=[[best_direction_agents[m]] for m in range(NumberOfAgents)]
            finish_path_length=1
        else:
            finish_path_length+=1
            for m in range(NumberOfAgents):
                finish_path_agents[m].append(best_direction_agents[m])
        for tar in range(NumberOfTargets):
            if(P_global[targets_arr[tar].x_pos][targets_arr[tar].y_pos][t]>p_limit and time_arr[tar]<0):
                time_arr[tar]=t

        GeneratePlots(t,isSharing=True)
        if(number_cells_above_p_global(t,p_limit)>=NumberOfTargets):
            break
        


    print('finished')
    print('finish_path_agents:',finish_path_agents)
    print('entropy:',Entropy_arr[:finish_path_length+1])
    print('prob 0,0 over time:',P_global[0][0][:finish_path_length+1])




#run_over_all_paths()
algorithm1_EIG_global_local_action()

import time
time.sleep(9999)




