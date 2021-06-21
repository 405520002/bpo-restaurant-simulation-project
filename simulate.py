# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 00:39:30 2021

@author: User
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('arrival1.csv')
service_rate_data=pd.read_csv('final0.8.csv')

hour_arrival=data['Value.1'].to_list()
#news invender model

daily_amount=np.random.poisson(52,10000)
np.quantile(daily_amount,0.9)










#cost :

ex_rate=27.66
costa=160/60/ex_rate
costb=1
costc=500/60/ex_rate
costd=180/60/ex_rate
coste=180/60/ex_rate
costf=180/60/ex_rate
costg=180/60/ex_rate
costh=1
costi=1
costj=160/60/ex_rate
waiting_cost=5/5/ex_rate#per person
cost_per_meal=37
price=cost_per_meal*1.5
out_of_sale_price=price*0.5


def bad_customer_error(people,mh,prob):
    error=np.random.rand(people)
    li=[1*(1+2/5*mh) if x<=prob else 1 for x in error]
    return np.array(li)

def chief_error(people,prob):
    error=np.random.rand(people)
    li=[1*(1+0.5) if x<=prob else 1 for x in error]
    return np.array(li)

def waiter_error(people,prob):
    error=np.random.rand(people)
    li=[1*(1+0.1) if x<=prob else 1 for x in error]
    return np.array(li)

def customer_amount(people):
    num=np.random.randint(1,4,people)
    return num
    
def arrival_to_people(arrival,ca):
    a=arrival.tolist()
    arr=[]
    for i,j in zip(a,ca):
        arr.append(i)
        for n in range(j-1) :
            arr.append(0)
    return np.array(arr)
            

            


people=42
from tqdm import tqdm

hour_arrival=pd.DataFrame(np.array(hour_arrival).reshape(5,-1))
data=pd.DataFrame([],columns=['simulate','customer','arrival_time','start_seat','end_seat','start_order','end_order','start_assign','end_assign','start_ape','end_ape','start_check_ape','end_check_ape'
                         ,'start_soup','end_soup','start_check_soup','end_check_soup','start_main','end_main','start_check_main','end_check_main','start_dessert','end_dessert','start_check_dessert','end_check_dessert','start_eat','end_eat','start_pay','end_pay','time_add'])
row=0
simulate=0
time_add=0
stock_out_list=[]   
index=0
for col in hour_arrival.columns:
    simulate+=1
    last_end_a=0
    last_end_b=0
    last_end_c=0
    last_end_d=0
    last_end_e=0
    last_end_f=0
    last_end_g=0
    last_end_h_d=0
    last_end_h_e=0
    last_end_h_f=0
    last_end_h_g=0
    last_end_i=0
    last_end_j=0
    time=0
    customer=0
    customer_hour=0
    hour_arrival_t=0
    for people in tqdm(hour_arrival[col].to_list()):
        m_station=service_rate_data.loc[index,:]
        ca=customer_amount(people)
        arrival=np.random.exponential(60/people,people)
        print(sum(arrival))
        if(len(arrival))>62:
            stock_out_list.append(1)
        else:
            stock_out_list.append(0)
        arrival=arrival_to_people(arrival,ca)    
        a=np.random.exponential(1/m_station['a'],people)
        b=np.random.exponential(5/m_station['i']/m_station['b'],people)
        a=arrival_to_people(a,ca)
        b=arrival_to_people(b,ca)
        people=len(arrival)
        c=np.random.exponential(0.5/m_station['c'],people)
        d=np.random.exponential(1/(m_station['d']*2),people)
        e=np.random.exponential(3/m_station['e'],people)
        f=np.random.exponential(15/m_station['f'],people)
        g=np.random.exponential(3/m_station['g'],people)
        h=np.random.exponential(1/m_station['h'],people)
        i=np.random.exponential(30/m_station['i'],people)
        j=np.random.exponential(3/m_station['j'],people)
        be=bad_customer_error(people,m_station['h'],2*(m_station['h'])/250)
        ce=chief_error(people,0.005)
        we=waiter_error(people,0.001)
        d=d*ce*be
        e=e*ce*be
        f=f*ce*be
        g=g*ce*be
        h=h*we*be
        index=index+1
        
        
        for ar,at,bt,ct,dt,et,ft,gt,ht,it,jt in zip(arrival,a,b,c,d,e,f,g,h,i,j):
            
            time+=ar
            arrival_time=time
            customer+=1
            
            if(last_end_a<=arrival_time):
                start_a=arrival_time
            else:
                start_a=last_end_a
            end_a=start_a+at
            
            if(last_end_b<=end_a):
                start_b=end_a
            else:
                start_b=last_end_b
            end_b=start_b+bt
            
            if(last_end_c<=end_b):
                start_c=end_b
            else:
                start_c=last_end_c
            end_c=start_c+ct
            
            if(last_end_d<=end_c):
                start_d=end_c
            else:
                start_d=last_end_d
            end_d=start_d+dt
            
            if(last_end_e<=end_c):
                start_e=end_c
            else:
                start_e=last_end_e
            end_e=start_e+et
            
            if(last_end_f<=end_c):
                start_f=end_c
            else:
                start_f=last_end_f
            end_f=start_f+ft
            
            if(last_end_g<=end_c):
                start_g=end_c
            else:
                start_g=last_end_g
            end_g=start_g+gt
            
            if(last_end_h_d<=end_d):
                start_h_d=end_d
            else:
                start_h_d=last_end_h_d
            end_h_d=start_h_d+ht
            
            if(last_end_h_e<=end_e):
                if(end_e<=end_h_d):
                    start_h_e=end_h_d
                else:
                    start_h_e=end_e     
            else:
                start_h_e=last_end_h_e
            end_h_e=start_h_e+ht
            
            if(last_end_h_f<=end_f):
                if(end_f<=end_h_e):
                    start_h_f=end_h_e
                else:
                    start_h_f=end_f  
            else:
                start_h_f=last_end_h_f
            end_h_f=start_h_f+ht
            
            if(last_end_h_g<=end_g):
                if(end_g<=end_h_f):
                    start_h_g=end_h_f
                else:
                    start_h_g=end_g  
            else:
                start_h_g=last_end_h_g
            end_h_g=start_h_g+ht
            
            if(last_end_i<=max(end_h_d,end_h_e,end_h_f,end_h_g)):
                start_i=max(end_h_d,end_h_e,end_h_f,end_h_g)
            else:
                start_i=last_end_i
            end_i=start_i+it
            
    
            if(last_end_j<=end_i):
                start_j=end_i
            else:
                start_j=last_end_j
            end_j=start_j+jt
            
            last_end_a=end_a
            last_end_b=end_b
            last_end_c=end_c
            last_end_d=end_d
            last_end_e=end_e
            last_end_f=end_f
            last_end_g=end_g
            last_end_h_d=end_h_d
            last_end_h_e=end_h_e
            last_end_h_f=end_h_f
            last_end_h_g=end_h_g
            last_end_i=end_i
            
            add={'simulate':simulate,'customer':customer, 'arrival_time':arrival_time, 'start_seat':start_a,'end_seat':end_a, 'start_order':start_b,
                 'end_order':end_b, 'start_assign':start_c, 'end_assign':end_c, 'start_ape':start_d, 'end_ape':end_d,'start_check_ape':start_h_d,'end_check_ape':end_h_d,'start_soup':start_e, 'end_soup':end_e,'start_check_soup':start_h_e,'end_check_soup':end_h_e ,'start_main':start_f, 'end_main':end_f,
                 'start_check_main':start_h_f,'end_check_main':end_h_f ,'start_dessert':start_g,'end_dessert':end_g, 'start_check_dessert':start_h_g, 'end_check_dessert':end_h_g,'start_eat':start_i, 'end_eat':end_i, 
                 'start_pay':start_j, 'end_pay':end_j,'time_add':end_j-last_end_j}
            last_end_j=end_j
            data=data.append(add,ignore_index=True)

          
          
colli=data.columns 
li=[x.replace('start_','') for x in colli]  
li=[x.replace('end_','') for x in li]
li=pd.DataFrame(li).drop_duplicates()[0].to_list()

def count_service_people(x):
    a=x[x['simulate']==1]['arrival_time']<x['end_check_ape'].value_counts()[True]
    return a
        

for col in li:
    if(col!='customer')&(col!='arrival_time')&(col!='simulate')&(col!='time_add'):
        data[col+'time']=data['end_'+col]-data['start_'+col]            
data['wait_for_take_seat']=-(data['arrival_time']-data['start_seat'])  
data['customer_wait_for_the_first_meal']=-(data['end_order']-data['end_check_ape'])
data['order_wait_for_assign']=-(data['end_order']-data['start_assign'])
data['customer_wait_for_pay']=-(data['end_eat']-data['end_pay'])  
data['customer_wait_for_meal']=data['end_check_ape']-data['arrival_time']
data['people in waiting']= data.apply(lambda x:sum(data.loc[(data['simulate']==x['simulate'])&(data['customer']>x['customer']),'arrival_time']<x['end_check_ape']),axis=1)
     
#data['add_waiting']=data['people in waiting'].diff().apply(lambda x: 0 if x<0 else x).fillna(0)
data['waiting_cost']=data['people in waiting']*waiting_cost

limit=62*5
data['revenue']=data['customer'].apply(lambda x: limit*(price-cost_per_meal)+(limit-x)*out_of_sale_price if x> limit else x*(price-cost_per_meal))
data['cost']=data['waiting_cost']+data['time_add']*(costa*m_station['a']*2+costb*m_station['b']+costc*m_station['c']+costd*m_station['d']+coste*m_station['e']+costf*m_station['f']+costg*m_station['g']+costh*m_station['h']+costj*m_station['j'])        
data['profit']=data['revenue']-data['cost']

data2=data[data['simulate']==1]


        
        
        
        
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


x_data = []
y_data1 = []
y_data2 = []
y_data3 = []
y_data4 = []


fig, ax = plt.subplots(2,2,figsize=(15,15))
plt.subplots_adjust(wspace =0.1, hspace =0.2)

line1, = ax[0][0].plot(0, 0,'r')
line3, = ax[0][1].plot(0, 0,'b')
line4, = ax[1][1].plot(0, 0,'y')

ax[0][0].title.set_text('time that customer wait for meal')
ax[1][0].title.set_text('customer_wait_for_meal')
ax[0][1].title.set_text('time that customer wait for pay')
ax[1][1].title.set_text('total profit')

y1=data2['customer_wait_for_meal'].to_list()
y2=data2['people in waiting'].to_list()
y3=data2['customer_wait_for_pay'].to_list()
y4=data2['profit'].to_list()
ax[0][0].set_xlim(0, 2000)
ax[0][0].set_ylim(0, max(y1)+10)
ax[0][1].set_xlim(0, 2000)
ax[0][1].set_ylim(0, max(y3)+10)
ax[1][1].set_xlim(0, 2000)
ax[1][1].set_ylim(0, max(y4)+10)


def animation_frame(i):
    x_data.append(i)
    y_data1.append(y1[i])
    y_data2.append(y2[i])
    y_data3.append(y3[i])
    y_data4.append(y4[i])
    a_text=ax[0][0].text(0.5,0.5,0,horizontalalignment='right',verticalalignment='top')
    ax[0][0].set_xlim(i-50,i+50)
    ax[0][1].set_xlim(i-50,i+50)
    ax[1][1].set_xlim(i-50,i+50)

    line1.set_xdata(x_data)
    line1.set_ydata(y_data1)
    if i<20:
        ax[1][0].bar(x_data[i],y_data2[i],color='y')
    else:
        ax[1][0].clear()
        ax[1][0].bar(x_data[i-20:i],y_data2[i-20:i])
    ax[0][0].title.set_text('time that customer wait for meal = '+str(round(y_data1[i],1)))
    ax[1][0].title.set_text('customer_wait_for_meal ='+str(int(y_data2[i])))
    ax[0][1].title.set_text('time that customer wait for pay ='+str(round(y_data3[i],1)))
    ax[1][1].title.set_text('total profit = '+str(int(y_data4[i])))    
    line3.set_xdata(x_data)
    line3.set_ydata(y_data3)
    line4.set_xdata(x_data)
    line4.set_ydata(y_data4)

    return line1,line3,line4


animation =FuncAnimation(fig, func=animation_frame, frames=np.arange(0, len(data2)), interval=0.0001,repeat=False)
plt.show()
animation.save('plot_utilization=0.8.final.mp4',fps=15)

        
        
        
        
        

        
    
    
    
            
        
        
        
        
        
    