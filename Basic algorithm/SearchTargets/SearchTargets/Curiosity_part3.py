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

import cv2
import os
newpath = r'C:\Users\barouch\Desktop\SearchVideo' 
if not os.path.exists(newpath):
    os.makedirs(newpath)
file=open("testfile.txt","w+")

seed(7)



N=20 # Matrix size
T=400
NumberOfTargets=3
NumberOfAgents=1


#setting sensors
s0=Sensor(1.0,10.0,1.0)
s1=Sensor(1.0,10.0,1.0)
s=[s0]
#agent_sensor_mat=[[s0,s0],[s1,s1]]
#agent_sensor_mat[3][2]- the agent with index 3, sensor index 2


#setting agents
#a=[Agent(randint(0,N-1),randint(0,N-1)) for i in range(NumberOfAgents)]
#a=[Agent(5,5),Agent(8,8),Agent(62,62)]#,Agent(4,4),Agent(10,10)]#,Agent(10,7),Agent(4,4),Agent(10,10)]
#a=[Agent(25,3),Agent(20,10)]
a=[Agent(9,9)]
for m in range(NumberOfAgents):
       print(m)
       print(a[m].x_pos)
       print(a[m].y_pos)

#setting targets
#Targets=[Target(randint(0,N-1),randint(0,N-1)) for i in range(NumberOfTargets)]
#Targets=[Target(4,34),Target(6,23),Target(37,3),Target(32,13),Target(2,5)]
Targets=[Target(2,2),Target(9,2),Target(2,16)]

#seed(5)
#Target event
Estimated_TE=[1]; #Estimated time for a target event per sensor

TargetsParameterGenerator=[[1] for i in range(NumberOfTargets)]
#TargetsParameterGenerator=[[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]]
def TargetsEventsGenerator(c,target, sensor): #c= TargetsParameterGenerator
    if(randint(1,c)==1):
        return 1
    else:
        return 0


#False Alarm
Estimated_FA=[4]; #Estimated False alarm rate per sensor

sum_max_avg_reward=0

FalseAlarmParameterGenerator=[4];
def FalseAlarmEventsGenerator(c): 
    if(randint(1,c)==1):
        return 1
    else:
        return 0



#setting the probability matrix at start

P=[[[[[0 for k in range(T)] for j in range(len(s))] for i in range(NumberOfAgents)] for i in range(N)]for i in range(N)]
P_cell=[[[0 for k in range(N)] for j in range(N)] for i in range(T)]
p_sensor=[0 for k in range(len(s))]

for i in range(N):
    for j in range(N):
        for m in range(NumberOfAgents):
            for k in range(len(s)):
                P[i][j][m][k][0]=0.01#NumberOfTargets/(N*N)
                #P_cell[0][i][j]=NumberOfTargets/(N*N)

def calcProbability(A,R,power):
    if(uniform(0,1)<A*e**(-R/power)):
        return 1
    else:
        return 0

def sensorsIntegration(p1,p2):
    temp= (p1*p2)/(p1*p2+(1-p1)*(1-p2))
    return temp


def updateProbability(t,pos_x,pos_y,agent,sensor):
    R=((pos_x-a[agent].x_pos)**2+(pos_y-a[agent].y_pos)**2)**0.5
    expCalc=e**(-R/s[sensor].pow)

    if(Rev[t][pos_x][pos_y][sensor][agent]==1):
        return (s[sensor].A*expCalc)*P[pos_x][pos_y][agent][sensor][t-1]*(1.0/Estimated_TE[sensor])/((s[sensor].A*expCalc)*P[pos_x][pos_y][agent][sensor][t-1]*(1.0/Estimated_TE[sensor])+(1.0/Estimated_FA[sensor])*s[sensor].A*expCalc*(1-P[pos_x][pos_y][agent][sensor][t-1]))
    else:
        return (1-(1.0/Estimated_TE[sensor])*s[sensor].A*expCalc)*P[pos_x][pos_y][agent][sensor][t-1]/(((1-(1.0/Estimated_TE[sensor])*s[sensor].A*e**(-R/s[sensor].pow))*P[pos_x][pos_y][agent][sensor][t-1])+(Estimated_FA[sensor]-s[sensor].A*expCalc)*(1-P[pos_x][pos_y][agent][sensor][t-1])/Estimated_FA[sensor])

def GenerateVideo(T, newpath):
    video_name = 'SearchVideo.avi'
    images = [0 for t in range(1,T)]
    for t in range(1,T):
        images[t-1] = cv2.imread(newpath+'/fig'+str(t)+'.png',0)
    frame = cv2.imread(newpath+'/fig1.png',0)
    height, width = frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    FPS=1
    video = cv2.VideoWriter(newpath+'/'+video_name, fourcc, FPS, (width,height))
    cv2.VideoWriter()
    
    for t in range(1,T):
        video.write(images[t-1])
    
    
    video.release()
    cv2.destroyAllWindows()

Tev=[[[0 for k in range(len(s))] for j in range(NumberOfTargets)] for i in range(T)]
Rev=[[[[[0 for k in range(NumberOfAgents)] for j in range(len(s))] for i in range(N)] for i in range(N)]for i in range(T)]
FA=[[[[0 for k in range(len(s))] for j in range(N)] for i in range(N)] for i in range(T)]
P_agent=[[[[0 for k in range(T)] for j in range(NumberOfAgents)] for i in range(N)] for i in range(N)]

P_range=[[[[0 for k in range(N)] for j in range(N)] for i in range(NumberOfAgents)] for m in range(T)]
time_arr=[0 for k in range(NumberOfTargets)]
p_ci_a=[0 for k in range(len(s))]

targets_agents_pos=[[(255,255,255) for k in range(N)] for j in range(N)] #Matrix for displaying agents and targets positions


def pr_calc(t, sharing):
    for m in range(NumberOfAgents):
        for i in range(N):
            for j in range(N):
                P_range[t][m][i][j]=0
                for k in range(len(s)):
                    R=((i-a[m].x_pos)**2+(j-a[m].y_pos)**2)**0.5
                    if (R==0):
                        R=0.01
                    if(sharing==True):
                        P_range[t][m][i][j]+=(P_cell[t][i][j])*(1.0/Estimated_TE[k])*s[k].A*exp(-R/s[k].pow)
                        #P_range[t][m][i][j]+=(P_cell[t][i][j])/(R**0.5)
                    else:
                        P_range[t][m][i][j]+=(P_agent[i][j][m][t])*(1.0/Estimated_TE[k])*s[k].A*exp(-R/s[k].pow)
                        #P_range[t][m][i][j]+=(P_agent[i][j][m][t])/(R**0.5)
                P_range[t][m][i][j]=P_range[t][m][i][j]/len(s)

def P_range_max(t):
    pr_max_indexes=[(-1,-1) for m in range(NumberOfAgents)]
    P_range_max_rec(pr_max_indexes,t)
    return pr_max_indexes


def P_range_max_rec(pr_max_indexes,t):
    if(((-1,-1) in pr_max_indexes)): #pr_max_indexes is still not complete
        index=-1
        for m in range(NumberOfAgents):
            if(pr_max_indexes[m]==(-1,-1)): #we still didn't attach indexes to this agent
                if(index==-1):
                    index=m
                for i in range(N):
                    for j in range(N):
                        if(P_range[t][m][i][j]>P_range[t][index][pr_max_indexes[index][0]][pr_max_indexes[index][1]]):
                              pr_max_indexes[index]=(-1,-1)
                              pr_max_indexes[m]=(i,j)                           
                              index=m

        temp=[P_range[t][m][pr_max_indexes[index][0]][pr_max_indexes[index][1]] for m in range(NumberOfAgents)]

        for m in range(NumberOfAgents):
            P_range[t][m][pr_max_indexes[index][0]][pr_max_indexes[index][1]]=0

        P_range_max_rec(pr_max_indexes,t) #recursive call

        for m in range(NumberOfAgents):
            P_range[t][m][pr_max_indexes[index][0]][pr_max_indexes[index][1]]=temp[m]



def move_decision(t,pr_max_indexes):
    for k in range(NumberOfAgents): #move each agent one step towards its selected cell
        #change x position:
        if(a[k].x_pos!=pr_max_indexes[k][0]):
            if(a[k].x_pos>pr_max_indexes[k][0]):
                a[k].x_pos-=1
            else:
                a[k].x_pos+=1

        #change y position:
        if(a[k].y_pos!=pr_max_indexes[k][1]):
            if(a[k].y_pos>pr_max_indexes[k][1]):
                a[k].y_pos-=1
            else:
                a[k].y_pos+=1
    #update targets probability
    isInTargetCell=0
    for k in range(NumberOfAgents):
        for tar in range(NumberOfTargets):
            if(Targets[tar].x_pos==a[k].x_pos and Targets[tar].y_pos==a[k].y_pos):
                #P_cell[t][a[k].x_pos][a[k].y_pos]=1
                for sensor in range(len(s)):
                    for agent in range(NumberOfAgents):
                        P[a[k].x_pos][a[k].y_pos][agent][sensor][t]=1
                isInTargetCell=1
                if(time_arr[tar]==0):
                    time_arr[tar]=t
        if(isInTargetCell==0): #agent is not located in a target cell
             #P_cell[t][a[k].x_pos][a[k].y_pos]=0
             for sensor in range(len(s)):
                   for agent in range(NumberOfAgents):
                       P[a[k].x_pos][a[k].y_pos][agent][sensor][t]=0
        isInTargetCell=0 #reset for next iteration
  
def sensor_power_modification(t,p_min,p_max,modFactor):
     for k in range(len(s)):      
         CounterAbove=0;
         CounterBelow=0;
         for i in range(N):
             for j in range(N):
                 for m in range(NumberOfAgents):
                    if(p_max<P[i][j][m][k][t]):
                        CounterAbove+=1
                        break;
         if(CounterAbove>=NumberOfTargets):
            if(s[k].pow>0.1):
               s[k].pow=s[k].pow*(1-modFactor)
            continue;
         for i in range(N):
             for j in range(N):
                 for m in range(NumberOfAgents):
                    if(p_min<P[i][j][m][k][t]):
                        CounterBelow+=1
                        break;
         if(CounterBelow<NumberOfTargets):
            s[k].pow=s[k].pow*(1+modFactor) 
            continue;

def GeneratePlots(t,isSharing):
    gs = gridspec.GridSpec(2, 2)
    plt.ion()
    
    plt.subplot(gs[0,0])
    if(isSharing==True):
        plt.imshow(P_cell[t],cmap='gray')
        plt.title("Probabilities matrix at t="+ str(t))
    else:
        temp=[[0 for k in range(N)] for j in range(N)]
        for i in range(N):
            for j in range(N):
                for m in range(NumberOfAgents):
                    temp[i][j]=max(temp[i][j],P_agent[i][j][m][t])
        plt.imshow(temp,cmap='gray')
        plt.title("Agents prob matrix at t="+ str(t))

    
    plt.subplot(gs[0,1])
    TEV_FA=[[0 for k in range(N)] for j in range(N)] 
    for i in range(N):
        for j in range(N):
            for k in range(len(s)):
                if(FA[t][i][j][k]==1):
                    TEV_FA[i][j]=1
    for tar in range(NumberOfTargets):
        for k in range(len(s)):
            if(Tev[t][tar][k]==1):
                TEV_FA[Targets[tar].x_pos][Targets[tar].y_pos]=1
    
    plt.imshow(TEV_FA,cmap='gray')
    plt.title("Events at t="+ str(t))
    
    
    
    plt.subplot(gs[1,0])
    for m in range(NumberOfAgents):
        targets_agents_pos[a[m].x_pos][a[m].y_pos]=(0,255,0) # green
    
    for tar in range(NumberOfTargets):
        targets_agents_pos[Targets[tar].x_pos][Targets[tar].y_pos]=(255,0,0) # red
    
    plt.imshow(targets_agents_pos)
    plt.title("Agents-Blue, Targets-Red ")
    
    
    
    
    #time figure
    plt.subplot(gs[1,1])
    plt.ylim((0,T))
    plt.bar(range(NumberOfTargets),time_arr,color="blue")
    plt.xticks(range(NumberOfTargets),["("+str(Targets[tar].x_pos)+","+str(Targets[tar].y_pos)+")" for tar in range(NumberOfTargets)])
    for tar in range(NumberOfTargets):
        if(time_arr[tar]!=0):
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
        targets_agents_pos[a[m].x_pos][a[m].y_pos]=(0,0,255)

def Rev_Func(t):
    for tar in range(NumberOfTargets):
        for k in range(len(s)):
            Tev[t][tar][k]=TargetsEventsGenerator(TargetsParameterGenerator[tar][k],Targets[tar],s[k])
            if(Tev[t][tar][k]==1):
                for m in range(NumberOfAgents):
                    R=((Targets[tar].x_pos-a[m].x_pos)**2+(Targets[tar].y_pos-a[m].y_pos)**2)**0.5
                    Rev[t][Targets[tar].x_pos][Targets[tar].y_pos][k][m]=calcProbability(s[k].A,R,s[k].pow)
            else:
                for m in range(NumberOfAgents):
                    Rev[t][Targets[tar].x_pos][Targets[tar].y_pos][k][m]=0
    for i in range(N):
        for j in range(N):
            for k in range(len(s)):
                FA[t][i][j][k]=FalseAlarmEventsGenerator(FalseAlarmParameterGenerator[k]);
                if(FA[t][i][j][k]==1):
                    for m in range(NumberOfAgents):
                        R=((i-a[m].x_pos)**2+(j-a[m].y_pos)**2)**0.5
                        Rev[t][i][j][k][m]=calcProbability(s[k].A,R,s[k].pow)
                elif(Rev[t][i][j][k][m]!=1):
                    Rev[t][i][j][k][m]=0

def Sensor_Probability(t):
    for i in range(N):
        for j in range(N):
        #for every every cell:
            for k in range(len(s)):
                for m in range(NumberOfAgents):
                    P[i][j][m][k][t]=updateProbability(t,i,j,m,k)

def Update_Estimated_FA(t):
    for k in range(len(s)):#for each type of sensor
        avg=0.0
        for m in range(NumberOfAgents):
            sumREV=0.0
            sumExpLambda=0.0
            #sumTrueEvents=0.0
            for i in range(N):
                for j in range(N):
                    sumREV+=Rev[t][i][j][k][m]
                    R=((i-a[m].x_pos)**2+(j-a[m].y_pos)**2)**0.5
                    sumExpLambda+=s[k].A*e**(-R/s[k].pow)
                    #sumTrueEvents+=P[i][j][m][k][t]*s[k].A*exp(-R/s[k].pow)
            if(sumREV==0):
                sumREV=0.01
            avg+=sumExpLambda/(sumREV)#-sumTrueEvents)
        Estimated_FA[k]=avg/NumberOfAgents


def Agent_Probability(t):
    for i in range(N):
        for j in range(N):
            #sensors integration per agent:
            for m in range(NumberOfAgents):
                P_agent[i][j][m][t]=P[i][j][m][0][t]
                for k in range(1,len(s)):
                    P_agent[i][j][m][t]=sensorsIntegration(P_agent[i][j][m][t],P[i][j][m][k][t])

def Update_Agent_Probability(t, p_limit):
    for i in range(N):
        for j in range(N):
            PmaxValue=0;
            PmaxIndex=0;
            for m in range(NumberOfAgents):
                if(PmaxValue<P_agent[i][j][m][t]):
                    PmaxValue=P_agent[i][j][m][t]
                    PmaxIndex=m;
            if(PmaxValue>p_limit):
                 for m in range(NumberOfAgents):
                     if(m!=PmaxIndex):
                         P_agent[i][j][m][t]=sensorsIntegration(P_agent[i][j][m][t],PmaxValue**0.1)

def Update_Sensor_Probability(t, p_limit):
    for i in range(N):
        for j in range(N):
            for k in range(len(s)):
                PmaxValue=0;
                PmaxAgent=0;
                for m in range(NumberOfAgents):               
                    if(PmaxValue<P[i][j][m][k][t]):
                        PmaxValue=P[i][j][m][k][t]
                        PmaxAgent=m;
                if(PmaxValue>p_limit):
                     print("index-%d,%d, agent-%d,%d" %(i,j,a[PmaxAgent].x_pos,a[PmaxAgent].y_pos))
                     for m in range(NumberOfAgents):
                            if(m!=PmaxAgent):
                                P[i][j][m][k][t]=sensorsIntegration(P[i][j][m][k][t],PmaxValue)


def Cell_Probability(t):
    for i in range(N):
        for j in range(N):
            #agents integration
            P_cell[t][i][j]= P_agent[i][j][0][t]
            for m in range(1,NumberOfAgents):
                P_cell[t][i][j]=sensorsIntegration(P_cell[t][i][j],P_agent[i][j][m][t])

def Algorithm1_RT_NetworkSharing(t):
    Rev_Func(t) #False alarm and true events generators + recieve event function 
    Sensor_Probability(t) #calculating the probability of a target for each sensor
    Update_Estimated_FA(t)
    Agent_Probability(t) #sensors' probabilities integration
    Cell_Probability(t) #agents' probabilities integration
    pr_calc(t,True)    #setting p_range, True-network sharing
    pr_max_indexes=P_range_max(t)  #getting the indexes of the max p_range per agent
    move_decision(t,pr_max_indexes)
    #sensor_power_modification(t,0.5,0.8,0.2)
    GeneratePlots(t,True) #Prints the plots

def Algorithm2_RT_No_NetworkSharing(t):
    Rev_Func(t) #False alarm and true events generators + recieve event function 
    
    Sensor_Probability(t) #calculating the probability of a target for each sensor
     
    Update_Estimated_FA(t)

    Agent_Probability(t) #sensors' probabilities integration

    pr_calc(t,False)    #setting p_range, False-no network sharing
    pr_max_indexes=P_range_max(t)  #getting the indexes of the max p_range per agent
    move_decision(t,pr_max_indexes)
    GeneratePlots(t,False) #Prints the plots

def Algorithm3_RT_PartialNetworkSharing(t):

    Rev_Func(t) #False alarm and true events generators + recieve event function 

    Sensor_Probability(t) #calculating the probability of a target for each sensor
    
    for m in range(NumberOfAgents):
        for k in range (len(s)):
            print("%f" %(P[2][7][m][k][t]))

    Update_Sensor_Probability(t,0.6)

    Update_Estimated_FA(t)
    
    Agent_Probability(t) #sensors' probabilities integration

    #Update_Agent_Probability(t,0.5)

    pr_calc(t,False)    #setting p_range, True-network sharing
    
    pr_max_indexes=P_range_max(t)  #getting the indexes of the max p_range per agent

    move_decision(t,pr_max_indexes)

    GeneratePlots(t,False) #Prints the plots

def Algorithm4_RT_LowerBound(t):
    for i in range(N):
        for j in range(N):
            P_cell[t][i][j]= 0
    for tar in range(NumberOfTargets):
        P_cell[t][Targets[tar].x_pos][Targets[tar].y_pos]=1

    pr_calc(t,True)    #setting p_range, True-network sharing
    pr_max_indexes=P_range_max(t)  #getting the indexes of the max p_range per agent
    move_decision(t,pr_max_indexes)
    #sensor_power_modification(t,0.5,0.8,0.2)
    GeneratePlots(t,True) #Prints the plots

def calculate_EIG_global(t,agent_location_x,agent_location_y,new_agent_location_x,new_agent_location_y):
    #P_cell[t]
    if(new_agent_location_x<0 or new_agent_location_x>=N):
        return -100000
    if(new_agent_location_y<0 or new_agent_location_y>=N):
        return -100000
    if(new_agent_location_y==agent_location_y and new_agent_location_y==agent_location_x):
        return 0
    EIG=0
    for i in range(N):
        for j in range(N):
            for k in range(len(s)):
                flag_visited=0
                for tar in range(NumberOfTargets):
                    if(time_arr[tar]!=0 and i==Targets[tar].x_pos and j==Targets[tar].y_pos):
                        flag_visited=1
                if(flag_visited==1):
                    continue
                R_new=((i-new_agent_location_x)**2+(j-new_agent_location_y)**2)**0.5
                R_prev=((i-agent_location_x)**2+(j-agent_location_y)**2)**0.5
                p_ci_a=P_cell[t][i][j]*(1.0/Estimated_TE[k])*s[k].A*exp(-R_new/s[k].pow)
                p_ci=P_cell[t][i][j]*(1.0/Estimated_TE[k])*s[k].A*exp(-R_prev/s[k].pow)
                #print(log(p_ci_a/p_ci,10))
                p_ci_a=max(p_ci_a,0.00001)
                p_ci=max(p_ci,0.00001)
                base=2
                EIG+=p_ci_a*math.log(p_ci_a/p_ci,base)

    return EIG


def calculate_EIG_global2(t,agent_location_x,agent_location_y,new_agent_location_x,new_agent_location_y):
    #P_cell[t]
    if(new_agent_location_x<0 or new_agent_location_x>=N):
        return -100000
    if(new_agent_location_y<0 or new_agent_location_y>=N):
        return -100000
    if(new_agent_location_y==agent_location_y and new_agent_location_y==agent_location_x):
        return 0
    EIG=0
    for i in range(N):
        for j in range(N):
            p_ci_a=0
            p_ci=0
            for k in range(len(s)):
                flag_visited=0
                for tar in range(NumberOfTargets):
                    if(time_arr[tar]!=0 and i==Targets[tar].x_pos and j==Targets[tar].y_pos):
                        flag_visited=1
                if(flag_visited==1):
                    continue
                R_new=((i-new_agent_location_x)**2+(j-new_agent_location_y)**2)**0.5
                R_prev=((i-agent_location_x)**2+(j-agent_location_y)**2)**0.5
                p_ci_a+=P_cell[t][i][j]*(1.0/Estimated_TE[k])*s[k].A*exp(-R_new/s[k].pow)
                p_ci+=P_cell[t][i][j]*(1.0/Estimated_TE[k])*s[k].A*exp(-R_prev/s[k].pow)
                #print(log(p_ci_a/p_ci,10))
                
            base=2
            p_ci_a=max(p_ci_a,10**(-100))
            p_ci=max(p_ci,10**(-100))
            EIG+=p_ci_a*math.log(p_ci_a/p_ci,base)
            #EIG+=p_ci_a*math.log(p_ci_a/p_ci,base)+(1-p_ci_a)*math.log(((1-p_ci_a)/(1-p_ci)),base)
    return EIG

def calculate_EIG_globa10(t,agent_location_x,agent_location_y,new_agent_location_x,new_agent_location_y):
    #P_cell[t]
    if(new_agent_location_x<0 or new_agent_location_x>=N):
        return -100000
    if(new_agent_location_y<0 or new_agent_location_y>=N):
        return -100000
    if(new_agent_location_y==agent_location_y and new_agent_location_y==agent_location_x):
        return 0
    EIG=0
    for i in range(N):
        for j in range(N):
            for k in range(len(s)):
                flag_visited=0
                for tar in range(NumberOfTargets):
                    if(time_arr[tar]!=0 and i==Targets[tar].x_pos and j==Targets[tar].y_pos):
                        flag_visited=1
                if(flag_visited==1):
                    continue
                EIG_sensor=0
                R_new=((i-new_agent_location_x)**2+(j-new_agent_location_y)**2)**0.5
                R_prev=((i-agent_location_x)**2+(j-agent_location_y)**2)**0.5
                p_x1=P[i][j][m][k][t]*(1.0/Estimated_TE[k])*s[k].A*exp(-R_new/s[k].pow)#+(1-P[i][j][m][k][t])*(1.0/Estimated_FA[k])*s[k].A*exp(-R_new/s[k].pow)
                p_x0=1-p_x1
                p_s_x1=P[i][j][m][k][t]*(1.0/Estimated_TE[k])/(P[i][j][m][k][t]*(1.0/Estimated_TE[k])+(1-P[i][j][m][k][t])*1.0/Estimated_FA[k])
                p_s_x0=P[i][j][m][k][t]*1.0/Estimated_TE[k]*s[k].A*exp(-R_new/s[k].pow)/(P[i][j][m][k][t]*1.0/Estimated_TE[k]*s[k].A*exp(-R_new/s[k].pow)+(1-P[i][j][m][k][t])*(1+1.0/Estimated_FA[k]*s[k].A*exp(-R_new/s[k].pow)))
               
                p_s_x1=max(p_s_x1,10**(-100))
                p_s_x0=max(p_s_x0,10**(-100))
                if (P[i][j][m][k][t]>0.000000001 and P[i][j][m][k][t]<1.0):
                     KL_x1=p_s_x1*math.log(p_s_x1/P[i][j][m][k][t],2)+(1-p_s_x1)*math.log((1-p_s_x1)/(1-P[i][j][m][k][t]),2)
                     KL_x0=p_s_x0*math.log(p_s_x0/P[i][j][m][k][t],2)+(1-p_s_x0)*math.log((1-p_s_x0)/(1-P[i][j][m][k][t]),2)
                else:
                     KL_x1=0
                     KL_x0=0
                EIG_sensor=p_x1*KL_x1+p_x0*KL_x0
                EIG+=EIG_sensor
                           
    print("EIG")
    print(EIG)
    return EIG

def calculate_EIG_MAP_global2(t,agent_location_x,agent_location_y,new_agent_location_x,new_agent_location_y,agent):
    #P_cell[t]
    if(new_agent_location_x<0 or new_agent_location_x>=N):
        return -100000
    if(new_agent_location_y<0 or new_agent_location_y>=N):
        return -100000
    if(new_agent_location_y==agent_location_y and new_agent_location_y==agent_location_x):
        return 0
    EIG=0
    p_sensor=0
    for i in range(N):
        for j in range(N):
            p_ci_a=0
            p_ci=0
            for k in range(len(s)):
                flag_visited=0
                for tar in range(NumberOfTargets):
                    if(time_arr[tar]!=0 and i==Targets[tar].x_pos and j==Targets[tar].y_pos):
                        flag_visited=1
                if(flag_visited==1):
                    continue
                R_new=((i-new_agent_location_x)**2+(j-new_agent_location_y)**2)**0.5
                R_old=((i-agent_location_x)**2+(j-agent_location_y)**2)**0.5
                R_new=max(R_new,0.001)
                R_old=max(R_old,0.001)
                
                P[i][j][agent][k][t]=max(P[i][j][agent][k][t],10**(-10))
                
                expCalc=e**(-R_new/s[k].pow)
                p_ci_a=P_cell[t][i][j]*(1.0/Estimated_TE[k])*s[k].A*expCalc/(((s[k].A*expCalc)*P[i][j][agent][k][t]*(1.0/Estimated_TE[k])+(1.0/Estimated_FA[k])*s[k].A*expCalc*(1-P[i][j][agent][k][t]))+(((1-(1.0/Estimated_TE[k])*s[k].A*expCalc)*P[i][j][agent][k][t])+(Estimated_FA[k]-s[k].A*expCalc)*(1-P[i][j][agent][k][t])/Estimated_FA[k]))
                #p_ci_a=P_cell[t][i][j]*(1.0/Estimated_TE[k])*s[k].A*expCalc#+(1.0/Estimated_FA[k])*s[k].A*expCalc*(1-P_cell[t][i][j])
                #p_pos=(s[k].A*expCalc)*P[i][j][agent][k][t]*(1.0/Estimated_TE[k])/((s[k].A*expCalc)*P[i][j][agent][k][t]*(1.0/Estimated_TE[k])+(1.0/Estimated_FA[k])*s[k].A*expCalc*(1-P[i][j][agent][k][t]))
                #p_neg=(1-(1.0/Estimated_TE[k])*s[k].A*expCalc)*P[i][j][agent][k][t]/(((1-(1.0/Estimated_TE[k])*s[k].A*expCalc)*P[i][j][agent][k][t])+(Estimated_FA[k]-s[k].A*expCalc)*(1-P[i][j][agent][k][t])/Estimated_FA[k])
                P[i][j][agent][k][t+2]=p_ci_a#*p_pos+(1-p_ci_a)*p_neg
               
                               
                expCalc=e**(-R_old/s[k].pow)
                p_ci_a=P_cell[t][i][j]*(1.0/Estimated_TE[k])*s[k].A*expCalc/(((s[k].A*expCalc)*P[i][j][agent][k][t]*(1.0/Estimated_TE[k])+(1.0/Estimated_FA[k])*s[k].A*expCalc*(1-P[i][j][agent][k][t]))+(((1-(1.0/Estimated_TE[k])*s[k].A*expCalc)*P[i][j][agent][k][t])+(Estimated_FA[k]-s[k].A*expCalc)*(1-P[i][j][agent][k][t])/Estimated_FA[k]))
                #p_ci_a=P_cell[t][i][j]*(1.0/Estimated_TE[k])*s[k].A*expCalc#+(1.0/Estimated_FA[k])*s[k].A*expCalc*(1-P_cell[t][i][j])
                #p_pos=(s[k].A*expCalc)*P[i][j][agent][k][t]*(1.0/Estimated_TE[k])/((s[k].A*expCalc)*P[i][j][agent][k][t]*(1.0/Estimated_TE[k])+(1.0/Estimated_FA[k])*s[k].A*expCalc*(1-P[i][j][agent][k][t]))
                #p_neg=(1-(1.0/Estimated_TE[k])*s[k].A*expCalc)*P[i][j][agent][k][t]/(((1-(1.0/Estimated_TE[k])*s[k].A*expCalc)*P[i][j][agent][k][t])+(Estimated_FA[k]-s[k].A*expCalc)*(1-P[i][j][agent][k][t])/Estimated_FA[k])
                P[i][j][agent][k][t+1]=p_ci_a#*p_pos+(1-p_ci_a)*p_neg

                #if (i==37 and j==3):
                     #print(str("P[37][3][agent][k][t+1] ")+str(agent)+str(" ")+str(P[i][j][agent][k][t+1]))
                #if (i==32 and j==13):
                     #print(str("P[32][13][agent][k][t+1] ")+str(agent)+str(" ")+str(P[i][j][agent][k][t+1]))

                if (P[i][j][agent][k][t+2]>0.00000001 and P[i][j][agent][k][t+1]>0.00000001):
                    EIG+=P[i][j][agent][k][t+2]*math.log(P[i][j][agent][k][t+2]/P[i][j][agent][k][t+1],2)      
            
               
            #P_agent[i][j][agent][t+1]=P[i][j][agent][0][t+1]
            #for k in range(1,len(s)):
                 #P_agent[i][j][agent][t+1]=sensorsIntegration(P_agent[i][j][agent][t+1],P[i][j][agent][k][t+1])
            #P_agent[i][j][agent][t+2]=P[i][j][agent][0][t+2]
            #for k in range(1,len(s)):
                 #P_agent[i][j][agent][t+2]=sensorsIntegration(P_agent[i][j][agent][t+2],P[i][j][agent][k][t+2])
            
            #if (P_agent[i][j][agent][t+1]>0.0000001 and P_agent[i][j][agent][t+2]>0.0000001):
                 #EIG+=P_agent[i][j][agent][t+2]*math.log(P_agent[i][j][agent][t+2]/P_agent[i][j][agent][t+1],2) 
            #for m in range(NumberOfAgents):
                #if (m!=agent):
                     #P_agent[i][j][m][t+1]=P_agent[i][j][m][t]
            #P_cell[t+1][i][j]= P_agent[i][j][0][t+1]
            #for m in range(1,NumberOfAgents):
                #if (P_cell[t+1][i][j]>0.0000001 and P_agent[i][j][m][t+1]>0.0000001):
                   #P_cell[t+1][i][j]=sensorsIntegration(P_cell[t+1][i][j],P_agent[i][j][m][t+1]) 
            #if (P_cell[t+1][i][j]>0.0000001 and P_cell[t][i][j]>0.0000001):
                #EIG+=P_cell[t+1][i][j]*math.log(P_cell[t+1][i][j]/P_cell[t][i][j],2) 
            P[i][j][agent][k][t+1]=0 
            P[i][j][agent][k][t+2]=0           
            P_agent[i][j][agent][t+1]=0
            P_agent[i][j][agent][t+2]=0
            #P_cell[t+1][i][j]=0
    #print(EIG)        
    return EIG


def calculate_EIG_agent(t,m,agent_location_x,agent_location_y,new_agent_location_x,new_agent_location_y):
    #P_cell[t]
    if(new_agent_location_x<0 or new_agent_location_x>=N):
        return -100000
    if(new_agent_location_y<0 or new_agent_location_y>=N):
        return -100000
    if(new_agent_location_y==agent_location_y and new_agent_location_y==agent_location_x):
        return 0
    EIG=0
    for i in range(N):
        for j in range(N):
            for k in range(len(s)):
                flag_visited=0
                for tar in range(NumberOfTargets):
                    if(time_arr[tar]!=0 and i==Targets[tar].x_pos and j==Targets[tar].y_pos):
                        flag_visited=1
                if(flag_visited==1):
                    continue
                R_new=((i-new_agent_location_x)**2+(j-new_agent_location_y)**2)**0.5
                R_prev=((i-agent_location_x)**2+(j-agent_location_y)**2)**0.5
                p_ci_a=P_agent[i][j][m][t]*(1.0/Estimated_TE[k])*s[k].A*exp(-R_new/s[k].pow)
                p_ci=P_agent[i][j][m][t]*(1.0/Estimated_TE[k])*s[k].A*exp(-R_prev/s[k].pow)
                #print(log(p_ci_a/p_ci,10))
                p_ci_a=max(p_ci_a,0.00001)
                p_ci=max(p_ci,0.00001)
                base=2
                EIG+=p_ci_a*math.log(p_ci_a/p_ci,base)

    return EIG


def calculate_EIG_agent2(t,m,agent_location_x,agent_location_y,new_agent_location_x,new_agent_location_y):
    #P_cell[t]
    if(new_agent_location_x<0 or new_agent_location_x>=N):
        return -100000
    if(new_agent_location_y<0 or new_agent_location_y>=N):
        return -100000
    if(new_agent_location_y==agent_location_y and new_agent_location_y==agent_location_x):
        return 0
    EIG=0
    for i in range(N):
        for j in range(N):
            p_ci_a=0
            p_ci=0
            for k in range(len(s)):
                flag_visited=0
                for tar in range(NumberOfTargets):
                    if(time_arr[tar]!=0 and i==Targets[tar].x_pos and j==Targets[tar].y_pos):
                        flag_visited=1
                if(flag_visited==1):
                    continue
                R_new=((i-new_agent_location_x)**2+(j-new_agent_location_y)**2)**0.5
                R_prev=((i-agent_location_x)**2+(j-agent_location_y)**2)**0.5
                p_ci_a+=P_agent[i][j][m][t]*(1.0/Estimated_TE[k])*s[k].A*exp(-R_new/s[k].pow)
                p_ci+=P_agent[i][j][m][t]*(1.0/Estimated_TE[k])*s[k].A*exp(-R_prev/s[k].pow)
                #print(log(p_ci_a/p_ci,10))
                
            base=2
            p_ci_a=max(p_ci_a,10**(-100))
            p_ci=max(p_ci,10**(-100))
            EIG+=p_ci_a*math.log(p_ci_a/p_ci,base)

    return EIG

def move_decision_EIG_global(t):
    for m in range(NumberOfAgents):
        max_EIG=0
        max_new_agent_location_x=a[m].x_pos
        max_new_agent_location_y=a[m].y_pos
        for x_offset in [-1,0,1]:
            for y_offset in [-1,0,1]:
                new_agent_location_x=a[m].x_pos+x_offset
                new_agent_location_y=a[m].y_pos+y_offset
                #print(max_new_agent_location_x)
                if(calculate_EIG_global2(t,a[m].x_pos,a[m].y_pos,new_agent_location_x,new_agent_location_y)>max_EIG):
                    max_EIG=calculate_EIG_global2(t,a[m].x_pos,a[m].y_pos,new_agent_location_x,new_agent_location_y)
                    max_new_agent_location_x=new_agent_location_x
                    max_new_agent_location_y=new_agent_location_y
                    #print(" ")
                    #print(a[m].x_pos)

        #print(max_new_agent_location_x-a[m].x_pos)
        #print(a[m].x_pos)
        #print(max_new_agent_location_x)
        if max_EIG>0.000000000001:
           a[m].x_pos=max_new_agent_location_x
           a[m].y_pos=max_new_agent_location_y

def move_decision10_EIG_global(t):
    for m in range(NumberOfAgents):
        max_EIG=0
        max_new_agent_location_x=a[m].x_pos
        max_new_agent_location_y=a[m].y_pos
        for x_offset in [-1,0,1]:
            for y_offset in [-1,0,1]:
                new_agent_location_x=a[m].x_pos+x_offset
                new_agent_location_y=a[m].y_pos+y_offset
                #print(max_new_agent_location_x)
                if(calculate_EIG_globa10(t,a[m].x_pos,a[m].y_pos,new_agent_location_x,new_agent_location_y)>max_EIG):
                    max_EIG=calculate_EIG_globa10(t,a[m].x_pos,a[m].y_pos,new_agent_location_x,new_agent_location_y)
                    max_new_agent_location_x=new_agent_location_x
                    max_new_agent_location_y=new_agent_location_y
                    #print(" ")
                    #print(a[m].x_pos)

        #print(max_new_agent_location_x-a[m].x_pos)
        #print(a[m].x_pos)
        #print(max_new_agent_location_x)
        print("max_EIG")
        print(max_EIG)
        if max_EIG>0.000000000001:
           a[m].x_pos=max_new_agent_location_x
           a[m].y_pos=max_new_agent_location_y


def move_decision_EIG_MAP_global(t):
    for m in range(NumberOfAgents):
        max_EIG=0
        max_new_agent_location_x=a[m].x_pos
        max_new_agent_location_y=a[m].y_pos
        for x_offset in [-1,0,1]:
            for y_offset in [-1,0,1]:
                new_agent_location_x=a[m].x_pos+x_offset
                new_agent_location_y=a[m].y_pos+y_offset
                #print(max_new_agent_location_x)
                if(calculate_EIG_MAP_global2(t,a[m].x_pos,a[m].y_pos,new_agent_location_x,new_agent_location_y,m)>max_EIG):
                    max_EIG=calculate_EIG_MAP_global2(t,a[m].x_pos,a[m].y_pos,new_agent_location_x,new_agent_location_y,m)
                    max_new_agent_location_x=new_agent_location_x
                    max_new_agent_location_y=new_agent_location_y
                    #print(" ")
                    #print(a[m].x_pos)

        print(max_EIG)

        #print(max_new_agent_location_x-a[m].x_pos)
        #print(a[m].x_pos)
        #print(max_new_agent_location_x)
        if max_EIG>0.0000000000000000002:
           a[m].x_pos=max_new_agent_location_x
           a[m].y_pos=max_new_agent_location_y   

def move_decision_EIG_agent(t):
    for m in range(NumberOfAgents):
        max_EIG=0
        max_new_agent_location_x=a[m].x_pos
        max_new_agent_location_y=a[m].y_pos
        for x_offset in [-1,0,1]:
            for y_offset in [-1,0,1]:
                new_agent_location_x=a[m].x_pos+x_offset
                new_agent_location_y=a[m].y_pos+y_offset
                #print(max_new_agent_location_x)
                if(calculate_EIG_agent2(t,m,a[m].x_pos,a[m].y_pos,new_agent_location_x,new_agent_location_y)>max_EIG):
                    max_EIG=calculate_EIG_agent2(t,m,a[m].x_pos,a[m].y_pos,new_agent_location_x,new_agent_location_y)
                    max_new_agent_location_x=new_agent_location_x
                    max_new_agent_location_y=new_agent_location_y
                    #print(" ")
                    #print(a[m].x_pos)

        #print(max_new_agent_location_x-a[m].x_pos)
        #print(a[m].x_pos)
        #print(max_new_agent_location_x)
        if max_EIG>0.00000001:
          a[m].x_pos=max_new_agent_location_x
          a[m].y_pos=max_new_agent_location_y


def all_offsets(n):
    return list(itertools.product([-1,0,1], repeat=2*n))

def calculate_EIG_global_united(t,offset_product):
    #P_cell[t]
    for m in range(NumberOfAgents):
        if(a[m].x_pos+offset_product[2*m]<0 or a[m].x_pos+offset_product[2*m]>=N):
            return -100000
        if(a[m].y_pos+offset_product[2*m+1]<0 or a[m].y_pos+offset_product[2*m+1]>=N):
            return -100000
    EIG=0
    
    for i in range(N):
        for j in range(N):
            p_ci_a=0
            p_ci=0
            for m in range(NumberOfAgents):
                for k in range(len(s)):
                    flag_visited=0
                    for tar in range(NumberOfTargets):
                        if(time_arr[tar]!=0 and i==Targets[tar].x_pos and j==Targets[tar].y_pos):
                            flag_visited=1
                    if(flag_visited==1):
                        continue
                    agent_location_x=a[m].x_pos
                    agent_location_y=a[m].y_pos
                    new_agent_location_x=agent_location_x+offset_product[2*m]
                    new_agent_location_y=agent_location_y+offset_product[2*m+1]
                    R_new=((i-new_agent_location_x)**2+(j-new_agent_location_y)**2)**0.5
                    R_prev=((i-agent_location_x)**2+(j-agent_location_y)**2)**0.5
                    p_ci_a+=P_cell[t][i][j]*(1.0/Estimated_TE[k])*s[k].A*exp(-R_new/s[k].pow)
                    p_ci+=P_cell[t][i][j]*(1.0/Estimated_TE[k])*s[k].A*exp(-R_prev/s[k].pow)
                    #print(log(p_ci_a/p_ci,10))
                
            base=2
            p_ci_a=max(p_ci_a,0.000000000001)
            p_ci=max(p_ci,0.000000000001)
            EIG+=p_ci_a*math.log(p_ci_a/p_ci,base)

    return EIG

def calculate_EIG_global_united10(t,offset_product):
    #P_cell[t]
    for m in range(NumberOfAgents):
        if(a[m].x_pos+offset_product[2*m]<0 or a[m].x_pos+offset_product[2*m]>=N):
            return -100000
        if(a[m].y_pos+offset_product[2*m+1]<0 or a[m].y_pos+offset_product[2*m+1]>=N):
            return -100000
    EIG=0
    
    for i in range(N):
        for j in range(N):
            p_ci_a=0
            p_ci=0
            for m in range(NumberOfAgents):
                for k in range(len(s)):
                    flag_visited=0
                    for tar in range(NumberOfTargets):
                        if(time_arr[tar]!=0 and i==Targets[tar].x_pos and j==Targets[tar].y_pos):
                            flag_visited=1
                    if(flag_visited==1):
                        continue
                    agent_location_x=a[m].x_pos
                    agent_location_y=a[m].y_pos
                    new_agent_location_x=agent_location_x+offset_product[2*m]
                    new_agent_location_y=agent_location_y+offset_product[2*m+1]
                    R_new=((i-new_agent_location_x)**2+(j-new_agent_location_y)**2)**0.5
                    R_prev=((i-agent_location_x)**2+(j-agent_location_y)**2)**0.5
                    p_x1= P[i][j][m][k][t]*(1.0/Estimated_TE[k])*s[k].A*exp(-R_new/s[k].pow)+(1-P[i][j][m][k][t])*(1.0/Estimated_FA[k])*s[k].A*exp(-R_new/s[k].pow)
                    p_x0=1-p_x1
                    p_s_x1=P[i][j][m][k][t]*(1.0/Estimated_TE[k])/[P[i][j][m][k][t]*(1.0/Estimated_TE[k])+(1-P[i][j][m][k][t])*1.0/Estimated_FA[k]]
                    p_s_x0=P[i][j][m][k][t]*1.0/Estimated_TE[k]*s[k].A*exp(-R_new/s[k].pow)/[P[i][j][m][k][t]*1.0/Estimated_TE[k]*s[k].A*exp(-R_new/s[k].pow)+(1-[P[i][j][m][k][t]])*(1+1.0/Estimated_FA[k]*s[k].A*exp(-R_new/s[k].pow))]
               
                    p_s_x1=max(p_s_x1,10**(-100))
                    p_s_x0=max(p_s_x0,10**(-100))
                
                    KL_x1=p_s_x1*math.log(p_s_x1/P[i][j][m][k][t],2)+(1-p_s_x1)*math.log((1-p_s_x1)/(1-P[i][j][m][k][t]),2)
                    KL_x0=p_s_x0*math.log(p_s_x0/P[i][j][m][k][t],2)+(1-p_s_x0)*math.log((1-p_s_x0)/(1-P[i][j][m][k][t]),2)
                    EIG_sensor=p_x1*KL_x1+p_x0*KL_x0
                    EIG+=EIG_sensor
                
            
    return EIG

def move_decision_EIG_global_united(t):
    max_offset_product=0
    max_EIG=0
    for offset_product in all_offsets(NumberOfAgents):
        #print(max_new_agent_location_x)
        if(calculate_EIG_global_united(t,offset_product)>max_EIG):
            max_EIG=calculate_EIG_global_united(t,offset_product)
            max_offset_product=offset_product

        #print(max_new_agent_location_x-a[m].x_pos)
        #print(a[m].x_pos)
        #print(max_new_agent_location_x)
    if(max_offset_product==0):
        return
    for m in range(NumberOfAgents):
        print(max_offset_product)
        a[m].x_pos=a[m].x_pos+max_offset_product[2*m]
        a[m].y_pos=a[m].y_pos+max_offset_product[2*m+1]

def move_decision_EIG_global_united10(t):
    max_offset_product=0
    max_EIG=0
    for offset_product in all_offsets(NumberOfAgents):
        #print(max_new_agent_location_x)
        if(calculate_EIG_global_united10(t,offset_product)>max_EIG):
            max_EIG=calculate_EIG_global_united10(t,offset_product)
            max_offset_product=offset_product

        #print(max_new_agent_location_x-a[m].x_pos)
        #print(a[m].x_pos)
        #print(max_new_agent_location_x)
    if(max_offset_product==0):
        return
    for m in range(NumberOfAgents):
        print(max_offset_product)
        a[m].x_pos=a[m].x_pos+max_offset_product[2*m]
        a[m].y_pos=a[m].y_pos+max_offset_product[2*m+1]

def update_time(t,prob_limit):
    for tar in range(NumberOfTargets):
        if(P_cell[t][Targets[tar].x_pos][Targets[tar].y_pos]>prob_limit and time_arr[tar]==0):
            time_arr[tar]=t

def update_time_agent(t,prob_limit):
    for tar in range(NumberOfTargets):
        for m in range(NumberOfAgents):
            if(P_agent[Targets[tar].x_pos][Targets[tar].y_pos][m][t]>prob_limit and time_arr[tar]==0):
                time_arr[tar]=t   



def Algorithm5_RT_EIG_Global_LocalAction(t):
    Rev_Func(t) #False alarm and true events generators + recieve event function 
    Sensor_Probability(t) #calculating the probability of a target for each sensor
    Update_Estimated_FA(t)
    Agent_Probability(t) #sensors' probabilities integration
    Cell_Probability(t) #agents' probabilities integration


    update_time(t,0.95)
    
    move_decision_EIG_global(t)
    Calculate_reward(t)
    #sensor_power_modification(t,0.5,0.8,0.2)
    GeneratePlots(t,True) #Prints the plots


def Algorithm5A_RT_EIG_Global_LocalAction(t):
    Rev_Func(t) #False alarm and true events generators + recieve event function 
    Sensor_Probability(t) #calculating the probability of a target for each sensor
    #Update_Estimated_FA(t)
    Agent_Probability(t) #sensors' probabilities integration
    Cell_Probability(t) #agents' probabilities integration


    update_time(t,0.95)
    
    move_decision10_EIG_global(t)
    Calculate_reward(t)
    #sensor_power_modification(t,0.5,0.8,0.2)
    GeneratePlots(t,True) #Prints the plots

def Algorithm5B_EIG_MAP_Global_LocalAction(t): 

    Rev_Func(t) #False alarm and true events generators + recieve event function 
    Sensor_Probability(t) #calculating the probability of a target for each sensor
    Update_Estimated_FA(t)
    Agent_Probability(t) #sensors' probabilities integration
    Cell_Probability(t) #agents' probabilities integration


    update_time(t,0.95)
    
    move_decision_EIG_MAP_global(t)
    #sensor_power_modification(t,0.5,0.8,0.2)
    GeneratePlots(t,True) #Prints the plots

def Algorithm6_RT_EIG_Agent_LocalAction(t):
    Rev_Func(t) #False alarm and true events generators + recieve event function 
    Sensor_Probability(t) #calculating the probability of a target for each sensor
    Update_Estimated_FA(t)
    Agent_Probability(t) #sensors' probabilities integration
    Cell_Probability(t) #agents' probabilities integration

    update_time(t,0.95)
    #update_time_agent(t,0.95)
    
    move_decision_EIG_agent(t)
    Calculate_reward(t)
    #sensor_power_modification(t,0.5,0.8,0.2)
    GeneratePlots(t,True) #Prints the plots


def Algorithm7_RT_EIG_Global_GlobalAction(t):
    Rev_Func(t) #False alarm and true events generators + recieve event function 
    Sensor_Probability(t) #calculating the probability of a target for each sensor
    Update_Estimated_FA(t)
    Agent_Probability(t) #sensors' probabilities integration
    Cell_Probability(t) #agents' probabilities integration


    update_time(t,0.95)
    
    move_decision_EIG_global_united(t)
    Calculate_reward(t)
    #sensor_power_modification(t,0.5,0.8,0.2)
    GeneratePlots(t,True) #Prints the plots

def Algorithm7A_RT_EIG_Global_GlobalAction(t):
    Rev_Func(t) #False alarm and true events generators + recieve event function 
    Sensor_Probability(t) #calculating the probability of a target for each sensor
    #Update_Estimated_FA(t)
    Agent_Probability(t) #sensors' probabilities integration
    Cell_Probability(t) #agents' probabilities integration


    update_time(t,0.95)
    
    move_decision_EIG_global_united10(t)
    Calculate_reward(t)
    #sensor_power_modification(t,0.5,0.8,0.2)
    GeneratePlots(t,True) #Prints the plots

def Algorithm8_RT_EIG_Static(t):
    Rev_Func(t) #False alarm and true events generators + recieve event function 
    Sensor_Probability(t) #calculating the probability of a target for each sensor
    #Update_Estimated_FA(t)
    Agent_Probability(t) #sensors' probabilities integration
    Cell_Probability(t) #agents' probabilities integration


    update_time(t,0.95)
    Calculate_reward(t) 

    #sensor_power_modification(t,0.5,0.8,0.2)
    GeneratePlots(t,True) #Prints the plots

def P_range_max9(t, whitelist,prob_limit):
    pr_max_indexes=[(-1,-1) for m in range(NumberOfAgents)]
    P_range_max_rec9(pr_max_indexes,t,whitelist,prob_limit)
    return pr_max_indexes


def P_range_max_rec9(pr_max_indexes,t,whitelist,prob_limit):
    if(((-1,-1) in pr_max_indexes)): #pr_max_indexes is still not complete
        index=-1
        for m in range(NumberOfAgents):
            if(pr_max_indexes[m]==(-1,-1)): #we still didn't attach indexes to this agent
                if(index==-1):
                    index=m
                for i in range(N):
                    for j in range(N):
                        if(P_range[t][m][i][j]>P_range[t][index][pr_max_indexes[index][0]][pr_max_indexes[index][1]] and ( (i,j) not in whitelist)):
                              pr_max_indexes[index]=(-1,-1)
                              pr_max_indexes[m]=(i,j)                           
                              index=m

        temp=[P_range[t][m][pr_max_indexes[index][0]][pr_max_indexes[index][1]] for m in range(NumberOfAgents)]

        for m in range(NumberOfAgents):
            P_range[t][m][pr_max_indexes[index][0]][pr_max_indexes[index][1]]=0

        P_range_max_rec9(pr_max_indexes,t,whitelist,prob_limit) #recursive call

        for m in range(NumberOfAgents):
            P_range[t][m][pr_max_indexes[index][0]][pr_max_indexes[index][1]]=temp[m]

def move_decision9(t,pr_max_indexes):
    for k in range(NumberOfAgents): #move each agent one step towards its selected cell
        #change x position:
        if(a[k].x_pos!=pr_max_indexes[k][0]):
            if(a[k].x_pos>pr_max_indexes[k][0]):
                a[k].x_pos-=1
            else:
                a[k].x_pos+=1

        #change y position:
        if(a[k].y_pos!=pr_max_indexes[k][1]):
            if(a[k].y_pos>pr_max_indexes[k][1]):
                a[k].y_pos-=1
            else:
                a[k].y_pos+=1
    #update targets probability
    isInTargetCell=0
    for k in range(NumberOfAgents):
        for tar in range(NumberOfTargets):
            if(Targets[tar].x_pos==a[k].x_pos and Targets[tar].y_pos==a[k].y_pos):
                #P_cell[t][a[k].x_pos][a[k].y_pos]=1
                for sensor in range(len(s)):
                    for agent in range(NumberOfAgents):
                        P[a[k].x_pos][a[k].y_pos][agent][sensor][t]=1
                isInTargetCell=1
                if(time_arr[tar]==0):
                    time_arr[tar]=t
        isInTargetCell=0 #reset for next iteration

whitelist=set()
def update_whitelist(t,prob_limt):
    for i in range(N):
        for j in range(N):
            if(P_cell[t][i][j]>prob_limt and (i,j) not in whitelist):
                whitelist.add((i,j))
                return


def Algorithm9_RT_Global_p_max_range(t):
    Rev_Func(t) #False alarm and true events generators + recieve event function 
    Sensor_Probability(t) #calculating the probability of a target for each sensor
    Update_Estimated_FA(t)
    Agent_Probability(t) #sensors' probabilities integration
    Cell_Probability(t) #agents' probabilities integration

    prob_limt=0.95
    update_whitelist(t,prob_limt)

    update_time(t,prob_limt)

    

    pr_calc(t,True)    #setting p_range, True-network sharing
    pr_max_indexes=P_range_max9(t,whitelist,prob_limt)  #getting the indexes of the max p_range per agent
    move_decision9(t,pr_max_indexes)

    
    #sensor_power_modification(t,0.5,0.8,0.2)
    GeneratePlots(t,True) #Prints the plots

def Expected_targets(t,p_target):
    set_targets=set()
    for i in range(N):
        for j in range(N):
            if(P_cell[t][i][j]>p_target and (i,j) not in whitelist):
                set_targets.add((i,j))
    return set_targets

def centerOfView(t,set_expected_targets):
    
    if(set_expected_targets=={}):
        return (-1,-1)
    return_positions=[(-1,-1) for m in range(NumberOfAgents)]
    for m in range(NumberOfAgents):
        max_view=0
        max_position=(-1,-1)
        for i in range(N):
            for j in range(N):
                view=0
                for sensor in s:
                    for location in set_expected_targets:
                        R1=((i-location[0])**2+(j-location[1])**2)**0.5
                        R2=((a[m].x_pos-location[0])**2+(a[m].y_pos-location[1])**2)**0.5
                        b=P_cell[t][location[0]][location[1]]*exp(-(R1/sensor.pow))
                        c=P_cell[t][location[0]][location[1]]*exp(-(R2/sensor.pow))
                        view+=b*math.log(b/c,2)
                        #view+=b*(R2/sensor.pow-R1/sensor.pow)
                if(view>max_view):                    
                    max_view=view
                    max_position=(i,j)
        return_positions[m]=max_position
        max_view=0
        max_position=(-1,-1)
    
    print(return_positions)
    #print(P_cell[t][return_positions[0][0]][return_positions[0][1]])
    return return_positions

def centerOfView_noShare(t,set_expected_targets_arr):
    return_positions=[(-1,-1) for m in range(NumberOfAgents)]
    for m in range(NumberOfAgents):
        if(set_expected_targets_arr[m]=={}):
            continue
        max_view=0
        max_position=(-1,-1)
        for i in range(N):
            for j in range(N):
                view=0
                for sensor in s:
                    for location in set_expected_targets_arr[m]:
                        R1=((i-location[0])**2+(j-location[1])**2)**0.5
                        R2=((a[m].x_pos-location[0])**2+(a[m].y_pos-location[1])**2)**0.5
                        b=P_agent[location[0]][location[1]][m][t]*exp(-(R1/sensor.pow))
                        c=P_agent[location[0]][location[1]][m][t]*exp(-(R2/sensor.pow))
                        view+=b*math.log(b/c,2)
                        #view+=b*(R2/sensor.pow-R1/sensor.pow)

                if(view>max_view):                    
                    max_view=view
                    max_position=(i,j)
        return_positions[m]=max_position
        max_view=0
        max_position=(-1,-1)
    
    print(return_positions)
    #print(P_cell[t][return_positions[0][0]][return_positions[0][1]])
    return return_positions

def move_decision10(t,destination_loc):
    if(destination_loc==(-1,-1)):
        return
    for m in range(NumberOfAgents):
        if(destination_loc[m]==(-1,-1)):
            continue
        x=a[m].x_pos
        y=a[m].y_pos
        #print(destination_loc[m])
        if(destination_loc[m][0]-x!=0):
            a[m].x_pos+=round((destination_loc[m][0]-x)/abs((destination_loc[m][0]-x)))
        if(destination_loc[m][1]-y!=0):
            a[m].y_pos+=round((destination_loc[m][1]-y)/abs((destination_loc[m][1]-y)))

def Calculate_reward(t):
    global sum_max_avg_reward
    reward=0
    for i in range(N):
         for j in range(N): 
            if (P_cell[t][i][j]>0.0000001 and P_cell[t-1][i][j]>0.0000001 and P_cell[t][i][j]<1.0 and P_cell[t-1][i][j]<1.0):
                reward+=P_cell[t][i][j]*math.log(P_cell[t][i][j]/P_cell[t-1][i][j],2)+(1-P_cell[t][i][j])*math.log((1-P_cell[t][i][j])/(1-P_cell[t-1][i][j]),2)
            else:
                reward+=0
    sum_max_avg_reward+=reward
    print("--------------")
    print("t="+str(t))
    file.write(" %f\n" % (sum_max_avg_reward))
    print("reward")
    print(reward) 
    print("sum_max_avg_reward")
    print(sum_max_avg_reward)

def Algorithm10_RT_Global_CenterOfView_NoOfTargetsUnknown(t):
    Rev_Func(t) #False alarm and true events generators + recieve event function 
    Sensor_Probability(t) #calculating the probability of a target for each sensor
    Update_Estimated_FA(t)
    Agent_Probability(t) #sensors' probabilities integration
    Cell_Probability(t) #agents' probabilities integration

    prob_limt=0.95
    update_whitelist(t,prob_limt)
    update_time(t,prob_limt)

    p_target=0.000000001
    set_expected_targets=Expected_targets(t,p_target)
    destination_locs=centerOfView(t,set_expected_targets)
    move_decision10(t,destination_locs)
    Calculate_reward(t)

    GeneratePlots(t,True) #Prints the plots

def Algorithm11_RT_Global_CenterOfMass_NoOfTargetsUnknown(t):

    Rev_Func(t) #False alarm and true events generators + recieve event function 
    Sensor_Probability(t) #calculating the probability of a target for each sensor
    Update_Estimated_FA(t)
    Agent_Probability(t) #sensors' probabilities integration
    Cell_Probability(t) #agents' probabilities integration

    prob_limt=0.95
    update_whitelist(t,prob_limt)
    update_time(t,prob_limt)

    p_target=0.00001
    set_expected_targets=Expected_targets(t,p_target)
    destination_loc=centerOfMass(t,set_expected_targets)
    move_decision10(t,destination_loc)
    Calculate_reward(t)
    GeneratePlots(t,True) #Prints the plots

def Fixed_Expected_targets_length(t):
    set_targets=set()
    number_expected=NumberOfTargets-len(whitelist)
    if(number_expected==0):
        return{}
    for i in range(N):
        for j in range(N):
            if((i,j) in whitelist):
                continue
            if(len(set_targets)!=number_expected):
                set_targets.add((i,j))
                continue
            if(P_cell[t][i][j]>min([P_cell[t][x][y] for (x,y) in set_targets])):
                    min_loc=(i,j)
                    for loc in set_targets:
                        if(P_cell[t][loc[0]][loc[1]]<P_cell[t][min_loc[0]][min_loc[1]]):
                            min_loc=loc
                    set_targets.remove(min_loc)
                    set_targets.add((i,j))
    
    return set_targets

def Algorithm12_RT_Global_CenterOfView(t):
    Rev_Func(t) #False alarm and true events generators + recieve event function 
    Sensor_Probability(t) #calculating the probability of a target for each sensor
    Update_Estimated_FA(t)
    Agent_Probability(t) #sensors' probabilities integration
    Cell_Probability(t) #agents' probabilities integration

    prob_limt=0.95
    update_whitelist(t,prob_limt)
    update_time(t,prob_limt)


    set_expected_targets=Fixed_Expected_targets_length(t)
    destination_loc=centerOfView(t,set_expected_targets)
    print("destination_loc")
    print(destination_loc)
    print()
    move_decision10(t,destination_loc)

    GeneratePlots(t,True) #Prints the plots


def centerOfMass(t,set_expected_targets):
    dest_x=0
    dest_y=0
    p_tot=0
    for location in set_expected_targets:
        dest_x+=P_cell[t][location[0]][location[1]]*abs(location[0])
        dest_y+=P_cell[t][location[0]][location[1]]*abs(location[1])
        p_tot+=P_cell[t][location[0]][location[1]]
    if(p_tot==0):
        return (-1,-1)
    return [(round(dest_x/p_tot),round(dest_y/p_tot)) for m in range(NumberOfAgents)]

def centerOfMass_noShare(t,set_expected_targets_arr):
    dest_arr=[(-1,-1) for m in range(NumberOfAgents)]
    for m in range(NumberOfAgents):
        dest_x=0
        dest_y=0
        p_tot=0
        for location in set_expected_targets_arr[m]:
            dest_x+=P_agent[location[0]][location[1]][m][t]*abs(location[0])
            dest_y+=P_agent[location[0]][location[1]][m][t]*abs(location[1])
            p_tot+=P_agent[location[0]][location[1]][m][t]
        if(p_tot==0):
            continue
        dest_arr[m]=(round(dest_x/p_tot),round(dest_y/p_tot))
    return dest_arr


def Algorithm13_RT_Global_CenterOfMass(t):
    Rev_Func(t) #False alarm and true events generators + recieve event function 
    Sensor_Probability(t) #calculating the probability of a target for each sensor
    Update_Estimated_FA(t)
    Agent_Probability(t) #sensors' probabilities integration
    Cell_Probability(t) #agents' probabilities integration

    prob_limt=0.95
    update_whitelist(t,prob_limt)
    update_time(t,prob_limt)


    set_expected_targets=Fixed_Expected_targets_length(t)
    destination_loc=centerOfMass(t,set_expected_targets)
    print("destination_loc")
    print(destination_loc)
    print()
    move_decision10(t,destination_loc)

    GeneratePlots(t,True) #Prints the plots

def Expected_targets_no_share(t,p_target):
    set_targets_arr=[0 for m in range(NumberOfAgents)]
    for m in range(NumberOfAgents):
        set_targets=set()
        for i in range(N):
            for j in range(N):
                if(i==2 and j==51):
                    #print(P_cell[t][Targets[1].x_pos][Targets[1].y_pos])
                    #print()
                    print("P_agent--------------------"+str(P_agent[i][j][m][t]))
                    print("P_cell--------------------"+str(P_cell[t][i][j]))
                if(P_agent[i][j][m][t]>p_target and (i,j) not in whitelist):
                    set_targets.add((i,j))
        set_targets_arr[m]=set_targets
    return set_targets_arr

def move_decision14(t,destination_loc_arr):
    print(destination_loc_arr)
    for m in range(NumberOfAgents):
        if(destination_loc_arr[m]==(-1,-1)):
            continue
        x=a[m].x_pos
        y=a[m].y_pos
        if(destination_loc_arr[m][0]-x!=0):
            a[m].x_pos+=round((destination_loc_arr[m][0]-x)/abs((destination_loc_arr[m][0]-x)))
        if(destination_loc_arr[m][1]-y!=0):
            a[m].y_pos+=round((destination_loc_arr[m][1]-y)/abs((destination_loc_arr[m][1]-y)))

def Algorithm14_RT_Agent_CenterOfMass_NoOfTargetsUnknown(t):
    Rev_Func(t) #False alarm and true events generators + recieve event function 
    Sensor_Probability(t) #calculating the probability of a target for each sensor
    Update_Estimated_FA(t)
    Agent_Probability(t) #sensors' probabilities integration
    Cell_Probability(t) #agents' probabilities integration

    prob_limt=0.95
    update_whitelist(t,prob_limt)
    update_time(t,prob_limt)

    p_target=0.00001
    set_expected_targets_arr=Expected_targets_no_share(t,p_target)
    #print(set_expected_targets_arr)
    destination_loc_arr=centerOfMass_noShare(t,set_expected_targets_arr)
    move_decision14(t,destination_loc_arr)
    Calculate_reward(t)  
    GeneratePlots(t,True) #Prints the plots
    


    
def Algorithm15_RT_Agent_CenterOfView_NoOfTargetsUnknown(t):
    Rev_Func(t) #False alarm and true events generators + recieve event function 
    Sensor_Probability(t) #calculating the probability of a target for each sensor
    Update_Estimated_FA(t)
    Agent_Probability(t) #sensors' probabilities integration
    Cell_Probability(t) #agents' probabilities integration

    prob_limt=0.95
    update_whitelist(t,prob_limt)
    update_time(t,prob_limt)

    p_target=0.001
    set_expected_targets_arr=Expected_targets_no_share(t,p_target)
    #print(set_expected_targets_arr)
    destination_loc_arr=centerOfView_noShare(t,set_expected_targets_arr)
    move_decision14(t,destination_loc_arr)
    Calculate_reward(t)
    GeneratePlots(t,True) #Prints the plots

reinforcment_counter=0
sensors_prob_avg=[[[[[[[0 for k in range(T)] for j in range(len(s))] for i in range(NumberOfAgents)] for i in range(N)]for i in range(N)] for i in range(3)] for i in range(3)]
prev_x=0
prev_y=0

def Algorithm16_Reinforcement(t):
    global reinforcment_counter
    global prev_x
    global prev_y
    global sum_max_avg_reward
    max_avg_reward=0
    
    max_action_x=0
    max_action_y=0
    Number_iterations=100


    #Rev_Func(t) #False alarm and true events generators + recieve event function 
    #Sensor_Probability(t) #calculating the probability of a target for each sensor
    if(t>1):
        for i in range(N):
            for j in range(N):
                for m in range(NumberOfAgents):
                    for k in range(len(s)):
                        P[i][j][m][k][t]=sensors_prob_avg[prev_x+1][prev_y+1][i][j][m][k][t]
    else:
        Rev_Func(t) #False alarm and true events generators + recieve event function 
        Sensor_Probability(t) #calculating the probability of a target for each sensor
        Update_Estimated_FA(t)


    
    Agent_Probability(t) #sensors' probabilities integration
    Cell_Probability(t) #agents' probabilities integration    

    prob_limt=0.95
    update_time(t,prob_limt)
    Calculate_reward(t)
    GeneratePlots(t,True) #Prints the plots

    print()
    print("--------------------------------")
    print("t  "+str(t)+"     "+str(P_cell[t][11][16]))
    print("t  "+str(t)+"     "+str(P_cell[t][0][14]))
    print("t  "+str(t)+"     "+str(P_cell[t][7][1]))
    for action_x in [-1,0,1]:
        for action_y in [-1,0,1]:
            
            avg_reward=0
            a[0].x_pos+=action_x
            a[0].y_pos+=action_y

            if(a[0].x_pos<0 or a[0].x_pos>=N or a[0].y_pos<0 or a[0].x_pos>=N):
                #if out of map boundries
                a[0].x_pos-=action_x
                a[0].y_pos-=action_y
                continue
            no_points=0
            for _ in range(Number_iterations):

                

                Rev_Func(t+1) #False alarm and true events generators + recieve event function 
                Sensor_Probability(t+1) #calculating the probability of a target for each sensor
                Update_Estimated_FA(t+1)
                Agent_Probability(t+1) #sensors' probabilities integration
                Cell_Probability(t+1) #agents' probabilities integration

                
                reward=0
                for i in range(N):
                    for j in range(N):
                        for m in range(NumberOfAgents):
                            for k in range(len(s)):
                                #sensors_prob_avg at time T+1 needs to be reseted at this point
                                sensors_prob_avg[action_x+1][action_y+1][i][j][m][k][t+1]+=P[i][j][m][k][t+1]/Number_iterations
                        if(P_cell[t+1][i][j]>0.00001 and P_cell[t][i][j]>0.00001):
                            no_points+=1
                            reward+=P_cell[t+1][i][j]*math.log(P_cell[t+1][i][j]/P_cell[t][i][j],2)
                            #reward+=P_cell[t+1][i][j]-P_cell[t][i][j]
                     
                avg_reward+=reward
                
                for i in range(N):
                    for j in range(N):
                        for m in range(NumberOfAgents):
                            P_cell[t+1][i][j]=0
                            P_agent[i][j][m][t+1]=0
                            P_range[t+1][m][i][j]=0
                            for k in range(len(s)):
                                P[i][j][m][k][t+1]=0
                for tar in range(NumberOfTargets):
                    for k in range(len(s)):
                        Tev[t+1][tar][k]=0
                for i in range(N):
                    for j in range(N):
                        for k in range(len(s)):
                            FA[t+1][i][j][k]=0
                            for m in range(NumberOfAgents):
                                Rev[t+1][i][j][k][m]=0


            if(Number_iterations!=0):
                avg_reward=avg_reward/Number_iterations
                
            print("avg_reward "+str(action_x)+" "+str(action_y)+"      val: "+ str(avg_reward))
            #print(avg_reward)
            print("no_points "+str(no_points/Number_iterations))
            if(avg_reward>max_avg_reward):
                max_avg_reward=avg_reward                
                max_action_x=action_x
                max_action_y=action_y

            a[0].x_pos-=action_x
            a[0].y_pos-=action_y           
   
    
            

    if(max_avg_reward<=0.00001):
        max_action_x=0
        max_action_y=0
    print("max_avg_reward")
    print(max_avg_reward)
    
    sum_max_avg_reward+=max_avg_reward
    print("sum_max_avg_reward")
    print(sum_max_avg_reward)    
  
    p_target=0.001
    set_expected_targets=Expected_targets(t,p_target)
    print(set_expected_targets)
    destination_loc=centerOfMass(t,set_expected_targets)
    
    if(destination_loc!=(-1,-1)):
        print("COM location")
        print(destination_loc)
        if(destination_loc[0][0]-a[0].x_pos!=0):
            action_x_COM=(destination_loc[0][0]-a[0].x_pos)/abs(destination_loc[0][0]-a[0].x_pos)
        else:
            action_x_COM=0
        if(destination_loc[0][1]-a[0].y_pos!=0):
            action_y_COM=(destination_loc[0][1]-a[0].y_pos)/abs(destination_loc[0][1]-a[0].y_pos)
        else:
            action_y_COM=0

        if(action_x_COM==max_action_x and action_y_COM==max_action_y):
            reinforcment_counter+=1
        print("COM")
        print(action_x_COM)
        print(action_y_COM)
        print("ACTION")
        print(max_action_x)
        print(max_action_y)
    else:
        print("COM location")
        print("not enough information for COM")
        print("COM")
        print("0")
        print("0")
        if(max_action_x==0 and max_action_y==0):
            reinforcment_counter+=1
        print("ACTION")
        print(max_action_x)
        print(max_action_y)


    

    a[0].x_pos+=max_action_x
    a[0].y_pos+=max_action_y
    prev_x=max_action_x
    prev_y=max_action_y

    print("reinforcment_counter:")
    print(reinforcment_counter)

def Algorithm17_RT_Integrated_CenterOfView_NoOfTargetsUnknown(t):    
   
    T_exp=20

    if (t<=T_exp):
       Algorithm15_RT_Agent_CenterOfView_NoOfTargetsUnknown(t)
    else:
       Algorithm10_RT_Global_CenterOfView_NoOfTargetsUnknown(t)
    


def isFinished():
    for tar in range(NumberOfTargets):
        isFound=False;
        for m in range(NumberOfAgents):
            if(Targets[tar].x_pos==a[m].x_pos and Targets[tar].y_pos==a[m].y_pos):
                isFound=True;
        if(isFound==False):
            return False;
    return True;



totalEnergy=0
totalTimeToTarget=0;
objValue=0;


GeneratePlots(0,True) #Prints the plots
time.sleep(5)
GeneratePlots(0,True) #Prints the plots

for t in range(1,T):
   # print("--------------")
   # print("t="+str(t))
    
    totalTimeToTarget=sum(time_arr);
    if(isFinished()==False):
        for k in range(len(s)):
            totalEnergy+=s[k].pow;
           # print("sensor- "+str(k)+" value is "+str(s[k].pow));
    else:
     #   print("Total Energy- "+str(totalEnergy));
     #  print("totalTimeToTarget- "+str(totalTimeToTarget))
        objValue=totalEnergy*totalTimeToTarget;
     #   print("OBJ VALUE- "+str(objValue))
     
   
    
 
    #Algorithm1_RT_NetworkSharing(t)

    #Algorithm2_RT_No_NetworkSharing(t)
    
    #Algorithm3_RT_PartialNetworkSharing(t)

    #Algorithm4_RT_LowerBound(t)

    Algorithm5_RT_EIG_Global_LocalAction(t)
   
    #Algorithm5A_RT_EIG_Global_LocalAction(t)
    
    #Algorithm5B_EIG_MAP_Global_LocalAction(t)

    #Algorithm6_RT_EIG_Agent_LocalAction(t)

    #Algorithm7_RT_EIG_Global_GlobalAction(t)

    #Algorithm7A_RT_EIG_Global_GlobalAction(t)
   
    #Algorithm8_RT_EIG_Static(t)

    #Algorithm9_RT_Global_p_max_range(t)

    #Algorithm10_RT_Global_CenterOfView_NoOfTargetsUnknown(t)

    #Algorithm11_RT_Global_CenterOfMass_NoOfTargetsUnknown(t)
    
    #Algorithm12_RT_Global_CenterOfView(t)

    #Algorithm13_RT_Global_CenterOfMass(t)

    #Algorithm14_RT_Agent_CenterOfMass_NoOfTargetsUnknown(t)

    #Algorithm15_RT_Agent_CenterOfView_NoOfTargetsUnknown(t)

    #Algorithm16_Reinforcement(t)

    #Algorithm17_RT_Integrated_CenterOfView_NoOfTargetsUnknown(t)
  
    
    #t0=time.clock()
    #t1=0
    
    #t1=time.clock()
    #print("GeneratePlots- "+str(t1-t0))
    #print(t)
    #print(Estimated_TE)
    #print(Estimated_FA)
    #print(whitelist)    
    #for tar in range(NumberOfTargets): 
       #print(P_cell[t][Targets[tar].x_pos][Targets[tar].y_pos])
    
#    print(str(Tev[t][0][0]))
#    print(str(Rev[t][65][60][0][0]))
#    print(str(P[5][10][1][1][t]))
#    print(str(P[8][8][1][1][t]))
#    print(str(t)+"    "+str(P_cell[t][39][39]))
#     print(P_cell[t][Targets[0].x_pos][Targets[0].y_pos])
#    print(str(P_cell[t][50][35]))
#    print(str(P_cell[t][20][70]))
#    print(str(P_cell[t][40][40]))
#    print(str("Positions:"))
#    print(str(pr_max_indexes))
#    print(str("a0 ")+str(a[0].x_pos)+str(" ")+str(a[0].y_pos))
#    print(str("a1 ")+str(a[1].x_pos)+str(" ")+str(a[1].y_pos))
#    print(str("a2 ")+str(a[2].x_pos)+str(" ")+str(a[2].y_pos))
#   print(str(time_arr[0]))
#   print(str(time_arr[1]))
#    print(str(time_arr[2]))
#   print(str(Targets))

#print("Total Energy- "+str(totalEnergy));
#print("totalTimeToTarget- "+str(totalTimeToTarget))
objValue=totalEnergy*totalTimeToTarget;
#print("OBJ VALUE- "+str(objValue))

for tar in range(NumberOfTargets):
       print(tar)
       print(Targets[tar].x_pos)
       print(Targets[tar].y_pos)
