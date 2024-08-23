import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import ode,odeint,solve_ivp


# 进水负荷浓度
df_SS_inlet = pd.DataFrame([[30,69.5,51.2,202.32,28.17,0,0,0,0,31.56,6.95,10.59,7,18446]])
df_SS_inlet.columns =      ['SI','SS','XI','XS','XBH','XBA','XP','SO','SNO','SNH','SND','XND','SALK','Q']

SIf  = df_SS_inlet['SI'][0]        #进水流量浓度 [g COD/m³] 
SSf  = df_SS_inlet['SS'][0]        #悬浮固体浓度 [g COD/m³]  
XIf  = df_SS_inlet['XI'][0]        #惰性生物质浓度[g COD/m³] 
XSf  = df_SS_inlet['XS'][0]        #底物浓度 [g COD/m³]
XBHf = df_SS_inlet['XBH'][0]       #异养生物质浓度 [g COD/m³]
XBAf = df_SS_inlet['XBA'][0]       #自养生物质浓度 [g COD/m³]
XPf  = df_SS_inlet['XP'][0]        #产物浓度 [g COD/m³]
SOf  = df_SS_inlet['SO'][0]        #溶解氧浓度 [g COD/m³]
SNOf = df_SS_inlet['SNO'][0]       #硝酸盐浓度 [g NO/m³]
SNHf = df_SS_inlet['SNH'][0]       #铵浓度 [g NH/m³]
SNDf = df_SS_inlet['SND'][0]       #亚硝酸盐浓度 [g N/m³]
XNDf = df_SS_inlet['XND'][0]       #不可降解污泥浓度[g N/m³]
SALKf = df_SS_inlet['SALK'][0]     #碱度浓度

# Define the ASM + BSM1 model constants and parameters
# 化学计量参数  
YA = 0.24  # 自养产物得率 [g COD formed/(g N oxidized)]  
YH = 0.67  # 异养产物得率 [g COD formed/(g COD oxidized)]  
fP = 0.08  # 生物质产生颗粒产品的比例 [-]  
iXB = 0.08  # 生物质中COD与氮的质量比 [g N/(g COD)]  
iXP = 0.06  # 颗粒产品中COD与氮的质量比 [g N/(g COD)]  

# 动力学参数  
µH = 4.0    # 异养生物最大比增长速率 [1/d]  
KS = 10     # 异养生物的半饱和系数 [g/m³]  
KOH = 0.2   # 异养生物的氧半饱和系数 [g/m³]  
KNO = 0.5   # 反硝化异养生物的硝酸盐半饱和系数 [g/m³]  
bH = 0.3    # 异养生物的衰减系数 [1/d]  
etag = 0.8  # 缺氧条件下最大生长速率的修正因子 [-]  
etah = 0.8  # 缺氧条件下水解的修正因子 [-]  
kh = 3.0    # 最大水解速率 [1/d]  
KX = 0.1    # 缓慢可降解底物的水解半饱和系数 [-]  
µA = 0.5    # 自养生物最大比增长速率 [1/d]  
KNH = 1     # 自养生物的铵半饱和系数 [g/m³]  
KOA = 0.4   # 自养生物的氧半饱和系数 [g/m³]  
bA = 0.05   # 自养生物的衰减系数 [1/d]  
ka = 0.05   # 氨化速度 [m³/g/day]  
kLa = 240   # 氧传质系数 [1/d]  
kLa_5 = 84  # 第五槽的氧传质系数 [1/day]  
SOsat = 8   # 氧饱和浓度 [g/m³]  

#定义流量
Qf = df_SS_inlet['Q'][0] # 进水流量 [m³/d] 
Qw = 385                 # 排放流量 [m³/d] 
Qa = 3 * Qf              # 内部回流流量 [m³/d] 
Qo = 2 * Qf              # 沉淀池进水流量 [m³/d] 
Qr = Qf                    # 回流流量 [m³/d] 
Qs = Qr + Qw             # 沉淀池底流 [m³/d] 
Qe = Qo - Qs             # 沉淀池溢流 [m³/d]  
Qra = Qa + Qr              # 总回流流量 [m³/d]  
Qi = Q1 = Q2 = Q3 = Q4 = Qf + Qra   # 第1、2、3、4池的流量 [m³/d]  

# 各个池的体积定义    
V1, V2, V3, V4, V5 = 1000, 1000, 1333, 1333,  1333  # 各池体积 [m³] 

# Takacs沉淀池实现  

# 从BSM1模型中获取的模型参数  
# 沉淀参数  
v0_max = 250        # 最大沉降速度               [m/d]  
v0     = 474        # 最大Vesilind沉降速度      [m/d]  
rh     = 0.000576   # 阻碍区沉降参数            [m³/(gSS)]  
rp     = 0.00286    # 絮凝剂区域沉降参数           [m³/(gSS)]  
f_ns   = 0.00228    # 不可沉降部分              [-]  

# 操作参数  
Xt     = 3000       # 阈值浓度                 [g/m³]  
A      = 1500       # 沉淀池的横截面积       [m²]  
zm     = 0.4        # 每层的高度               [m]  
V      = 6000       # 沉淀池的体积             [m³]  

frCOD_SS = 4 / 3    # COD转化为SS的转换因子      [-]  

# Activated Sludge Model No.1 (ASM1)

def procsss_rates(SS, XS, XBH, XBA, XP, SO, SNO, SNH, SND, XND):
    """ Procsss rates
        SS   #悬浮固体浓度 [g COD/m³]  
        KS   # 异养生物的半饱和系数
        XBH  #异养生物质浓度 [g COD/m³]
        XBA  #自养生物质浓度 [g COD/m³]
        SO   #溶解氧浓度 [g COD/m³]
        SNO  #硝酸盐浓度 [g NO/m³]
        SNH  #铵浓度 [g NH/m³]
        SND  #亚硝酸盐浓度 [g N/m³]
        XND  #不可降解污泥浓度[g N/m³]
      """
    
    rho = np.zeros(8)
    # 异养生物的有氧生长  
    rho[0] = µH*(SS/(KS+SS))*(SO/(KOH+SO))*XBH
    # 异养生物的缺氧生长  
    rho[1] = µH*(SS/(KS+SS))*(KOH/(KOH+SO))*(SNO/(KNO+SNO))*etag*XBH  
    # 自养生物的有氧生长  
    rho[2] = µA*(SNH/(KNH+SNH))*(SO/(KOA+SO))*XBA  
    # 异养生物的衰减  
    rho[3] = bH*XBH  
    # 自养生物的衰减  
    rho[4] = bA*XBA  
    # 可溶性有机氮的氨化  
    rho[5] = ka*SND*XBH  
    # 被捕获有机物的水解  
    rho[6] = kh*((XS/XBH)/(KX+(XS/XBH)))*(((SO/(KOH+SO))) \
             + etah*(KOH/(KOH+SO))*(SNO/(KNO+SNO)))*XBH
    # 被捕获有机氮的水解  
    rho[7] = rho[6]*(XND/XS)

    return rho

def conversion_rates(rho):
    """ 转换速率
        rho ：过程速率 
    """
    
    r = np.zeros(12)  
    # 可溶性惰性有机物 SI  
    r[0] = 0  
    # 难降解基质 SS  
    r[1] = (-1 / YH) * (rho[0] + rho[1]) + rho[6]   
    # 固体惰性有机物 XI  
    r[2] = 0  
    # 缓慢生物降解基质 XS  
    r[3] = (1 - fP) * (rho[3] + rho[4]) - rho[6]   
    # 活性异养生物 XBH  
    r[4] = rho[0] + rho[1] - rho[3]   
    # 活性自养生物 XBA  
    r[5] = rho[2] - rho[4]   
    # 由于生物衰减产生的颗粒产品 XP  
    r[6] = fP * (rho[3] + rho[4])   
    # 氧 SO   
    r[7] = (-(1 - YH) / YH) * rho[0] + (-(4.57 - YA) / YA) * rho[2]   
    # 硝酸盐和亚硝酸盐氮 SNO  
    r[8] = (-(1 - YH) / (2.86 * YH)) * rho[1] + (1 / YA) * rho[2]   
    # NH4+/NH3氮 SNH  
    r[9] = -iXB * (rho[0] + rho[1]) - (iXB + 1 / YA) * rho[2] + rho[5]   
    # 可溶性生物降解有机氮 SND  
    r[10] = -rho[5] + rho[7]   
    # 粒状生物降解有机氮 XND  
    r[11] = (iXB - fP * iXP) * (rho[3] + rho[4]) - rho[7]   
    
    return r  

def derivatives_settler(X, Xo, Xmin, Qe, Qs, Qo):
    """ 
        沉淀池模型
        X: 当前层的颗粒物浓度数组，通常表示每层的颗粒物质量浓度（如 [g/m³]）。
        Xo :活性污泥反应器中的浓度
        Xmin: 沉降污泥浓度最低阈值
        Qe : 沉淀池的溢出流量
        Qs : 沉淀池底流
        Qo : 沉淀池进水流量    
    """
    # Definition of upward and downward velocities
    v_up = Qe/A           # 上流速度 v_up 
    v_down = Qs/A         # 下流速度 v_down

     # 定义沉降通量  
    Js = []  # 用于存储每层的沉降通量
    for j in range(len(X)):
        #当前层的沉降通量
        Jl = min(v0_max,v0*(np.exp(-rh*(X[j]-Xmin))-np.exp(-rp*(X[j]-Xmin))))
        #对于前7层，直接使用当前层的沉降通量；从第7层开始，使用当前和前一层沉降通量的较小值
        if j < 7:
            Js.append(max(0,Jl))
        else:
            Js.append(min(Js[j-1],Jl))
    
    # 每层污泥质量守恒
    dXdt = np.zeros(10) 
    dXdt[0] = 1/zm*(v_down*(X[1]-X[0])+min(Js[0],Js[1]))
    dXdt[1] = 1/zm*(v_down*(X[2]-X[1])+min(Js[1], Js[2])-min(Js[1], Js[0]))
    dXdt[2] = 1/zm*(v_down*(X[3]-X[2])+min(Js[2], Js[3])-min(Js[2], Js[1]))
    dXdt[3] = 1/zm*(v_down*(X[4]-X[3])+min(Js[3], Js[4])-min(Js[3], Js[2]))
    dXdt[4] = 1/zm*(v_down*(X[5]-X[4])+min(Js[4], Js[5])-min(Js[4], Js[3]))
    dXdt[5] = 1/zm*((Qo*Xo)/A-(v_up+v_down)*X[5] + Js[6]-min(Js[5], Js[4]))
    dXdt[6] = 1/zm*(v_up*(X[5]-X[6])+Js[7] - Js[6])
    dXdt[7] = 1/zm*(v_up*(X[6]-X[7])+Js[8] - Js[7])
    dXdt[8] = 1/zm*(v_up*(X[7]-X[8])+Js[9] - Js[8])
    dXdt[9] = 1/zm*(v_up*(X[8]-X[9])-Js[9])

    return dXdt

def derivatives_reactor(C, Cin, V, Q, aeration=False, kla_DO=False):  

    """ 生物反应器模型
    C : 当前状态下反应器内各组分的浓度数组
    Cin : 进水中各组分的浓度数组
    V: 反应器的体积
    Q: 进水流量
    aeration(可选): 指示是否进行曝气。默认值为 False
    kla_DO(可选): 指示是否使用特定的气体传输速率。默认值为 False"""

    #变量提取
    SI, SS, XI, XS, XBH, XBA, XP, SO, SNO, SNH, SND, XND = C[:]
    SIin, SSin, XIin, XSin, XBHin, XBAin, XPin, SOin, SNOin, SNHin, SNDin, XNDin = Cin[:]
    
    #定义过程速率
    rho = procsss_rates(SS, XS, XBH, XBA, XP, SO, SNO, SNH, SND, XND)
    #定义转化速率
    r = conversion_rates(rho)
    
    #定义dcdt用于存储各组分的浓度变化率
    dCdt = np.zeros(len(r))
    #溶解惰性有机物 SI
    dCdt[0] = 1/V*(r[0]*V+SIin*Q-SI*Q)
    #易降解底物 SS
    dCdt[1]  = r[1] + Q/V*(SSin - SS)
    #颗粒惰性有机物 XI
    dCdt[2]  = r[2] + Q/V*(XIin - XI)
    #缓慢降解底物 XS
    dCdt[3]  = r[3] + Q/V*(XSin - XS)
    #活性异养生物质 XBH
    dCdt[4]  = r[4] + Q/V*(XBHin - XBH)
    #活性自养生物质 XBA
    dCdt[5]  = r[5] + Q/V*(XBAin - XBA)
    #生物降解产物 XP
    dCdt[6]  = r[6] + Q/V*(XPin - XP)
    #氧气 SO
    if aeration:
        dCdt[7] = r[7] + Q/V*(SOin-SO) + kLa * (SOsat - SO)
    else:
        dCdt[7] = r[7] + Q/V*(SOin-SO)
    #硝酸盐和亚硝酸盐氮 SNO
    dCdt[8]  = r[8] + Q/V*(SNOin - SNO)
    #铵离子/氨氮 SNH
    dCdt[9]  = r[9] + Q/V*(SNHin - SNH)
    #可降解有机氮 SND
    dCdt[10] = r[10] + Q/V*(SNDin - SND)
    #颗粒可降解有机氮 XND
    dCdt[11] = r[11] + Q/V*(XNDin - XND)

    return dCdt

def ODEint (t,C):
    """ 构建一个包含 5 个反应器和一个 10 层沉淀池的常微分方程(ODE)系统
        C: 初始条件的数组，包含了所有反应器和沉淀池的浓度
        t: 时间变量，在 ODE 求解过程中会被传递
    """
    # Reactor 1  
    SI1, SS1, XI1, XS1, XBH1, XBA1, XP1, SO1, SNO1, SNH1, SND1, XND1 = C[0:12]  
    # Reactor 2  
    SI2,SS2,XI2,XS2,XBH2,XBA2,XP2,SO2,SNO2,SNH2,SND2,XND2 = C[12:24]  
    # Reactor 3
    SI3,SS3,XI3,XS3,XBH3,XBA3,XP3,SO3,SNO3,SNH3,SND3,XND3 = C[24:36]  
    # Reactor 4
    SI4,SS4,XI4,XS4,XBH4,XBA4,XP4,SO4,SNO4,SNH4,SND4,XND4 = C[36:48]  
    # Reactor 5
    SI5,SS5,XI5,XS5,XBH5,XBA5,XP5,SO5,SNO5,SNH5,SND5,XND5 = C[48:60]
    # Settler
    X1,X2,X3,X4,X5,X6,X7,X8,X9,X10 = C[60:70]
    #获取每一层之前的生物质浓度值
    X0 = [X1, X2, X3, X4, X5, X6, X7, X8, X9, X10]  
    #从第5隔间的浓度计算污泥浓度
    Xo = 1/frCOD_SS * (XI5 + XS5 + XBH5 + XBA5 + XP5)  
    #Sludge concentration for which settling velocity equals zero  
    Xmin = f_ns * Xo 

    #假设沉淀池的进水与出水保持一致，出水状态变量计算:
    XIs   = XI5  * X1/Xo   # [g COD/m³]
    XSs   = XS5  * X1/Xo   # [g COD/m³]
    XBHs  = XBH5 * X1/Xo   # [g COD/m³]
    XBAs  = XBA5 * X1/Xo   # [g COD/m³]
    XPs   = XP5  * X1/Xo   # [g COD/m³]
    XNDs  = XND5 * X1/Xo   # [g N/m³]

    #进水浓度重新计算
    SIi  = 1/(Qf + Qra)*(Qf*SIf  + Qra*SI5)
    SSi  = 1/(Qf + Qra)*(Qf*SSf  + Qra*SS5)
    XIi  = 1/(Qf + Qra)*(Qf*XIf  + Qa *XI5  +Qr*XIs)
    XSi  = 1/(Qf + Qra)*(Qf*XSf  + Qa *XS5  +Qr*XSs)
    XBHi = 1/(Qf + Qra)*(Qf*XBHf + Qa *XBH5 +Qr*XBHs)
    XBAi = 1/(Qf + Qra)*(Qf*XBAf + Qa *XBA5 +Qr*XBAs)
    XPi  = 1/(Qf + Qra)*(Qf*XPf  + Qa *XP5  +Qr*XPs)
    SOi  = 1/(Qf + Qra)*(Qf*SOf  + Qra*SO5)
    SNOi = 1/(Qf + Qra)*(Qf*SNOf + Qra*SNO5)
    SNHi = 1/(Qf + Qra)*(Qf*SNHf + Qra*SNH5)
    SNDi = 1/(Qf + Qra)*(Qf*SNDf + Qra*SND5)
    XNDi = 1/(Qf + Qra)*(Qf*XNDf + Qa *XND5 +Qr*XNDs)

    #定义变化率数组
    dCdt = np.zeros(len(C))
    
    # Anoxic Reactor
    #reactor1
    dCdt[0:12] = derivatives_reactor( \
                [SI1, SS1, XI1, XS1, XBH1, XBA1, XP1, SO1, SNO1, SNH1, SND1, XND1], \
                [SIi, SSi, XIi, XSi, XBHi, XBAi, XPi, SOi, SNOi, SNHi, SNDi, XNDi], V1, Qi, aeration=False)

    #reactor2
    dCdt[12:24] = derivatives_reactor( \
                [SI2, SS2, XI2, XS2, XBH2, XBA2, XP2, SO2, SNO2, SNH2, SND2, XND2], \
                [SI1, SS1, XI1, XS1, XBH1, XBA1, XP1, SO1, SNO1, SNH1, SND1, XND1], V2, Q1, aeration=False) 

    # Aerated tanks
    #reactor3
    dCdt[24:36] = derivatives_reactor( \
                [SI3, SS3, XI3, XS3, XBH3, XBA3, XP3, SO3, SNO3, SNH3, SND3, XND3], \
                [SI2, SS2, XI2, XS2, XBH2, XBA2, XP2, SO2, SNO2, SNH2, SND2, XND2], V3, Q2, aeration=True,kla_DO=False)                                                                              

    #reactor4
    dCdt[36:48] = derivatives_reactor( \
                [SI4, SS4, XI4, XS4, XBH4, XBA4, XP4, SO4, SNO4, SNH4, SND4, XND4], \
                [SI3, SS3, XI3, XS3, XBH3, XBA3, XP3, SO3, SNO3, SNH3, SND3, XND3], V4, Q3, aeration=True,kla_DO=False) 
    
    # reactor5
    dCdt[48:60] = derivatives_reactor( \
                [SI5, SS5, XI5, XS5, XBH5, XBA5, XP5, SO5, SNO5, SNH5, SND5, XND5], \
                [SI4, SS4, XI4, XS4, XBH4, XBA4, XP4, SO4, SNO4, SNH4, SND4, XND4], V5, Q4, aeration=True,kla_DO=True) 

    dCdt[60:70] = derivatives_settler(X0, Xo, Xmin, Qe, Qs, Qo)
                                                  
    return dCdt

def main():  
    # 基于进水值进行初始化（但提供一些 XBA 和 SNO 以启动生长）  
    C0 = np.array([SIf, SSf, XIf, XSf, XBHf, 50, XPf, 0, 2, SNHf, SNDf, XNDf,   # Tank1  
                   SIf, SSf, XIf, XSf, XBHf, 50, XPf, 0, 2, SNHf, SNDf, XNDf,   # Tank2  
                   SIf, SSf, XIf, XSf, XBHf, 50, XPf, 8, 2, SNHf, SNDf, XNDf,   # Tank3  
                   SIf, SSf, XIf, XSf, XBHf, 50, XPf, 8, 2, SNHf, SNDf, XNDf,   # Tank4  
                   SIf, SSf, XIf, XSf, XBHf, 50, XPf, 2, 2, SNHf, SNDf, XNDf])  # Tank5  

    # 初始化沉降器  
    Xo = 1 / frCOD_SS * (C0[50] + C0[51] + C0[52] + C0[53] + C0[54])  # XI5 + XS5 + XBH5 + XBA5 + XP5  
    X0 = np.zeros(10)  # 创建长度为10的零数组  
    X0[:] = Xo  # 将初始值赋予该数组  

    # 定义生物反应器和沉降器的初始向量  
    C0 = np.append(C0, X0)  # 将初始浓度数组与次级沉降器状态合并  

    # 设置时间信息  
    dt = 0.05     # [d] 时间步长  
    tStart = 0    # [d] 开始时间  
    tEnd = 50     # [d] 结束时间  


    # 设置ODE积分器  
    r = ode(ODEint)  # 创建ODE求解器实例  
    r.set_integrator("vode", method="adams", with_jacobian=False, nsteps=10000)  # 设置求解器为Adams方法  
    r.set_initial_value(C0, tStart)  # 设置初始条件  

    # 定义DataFrame以存储每个时间步的5个反应器及沉降器数据  
    system_data_ode = pd.DataFrame(0, columns=["time",   
                                    "SI1", "SS1", "XI1", "XS1", "XBH1", "XBA1",   
                                    "XP1", "SO1", "SNO1", "SNH1", "SND1", "XND1",  
                                    "SI2", "SS2", "XI2", "XS2", "XBH2", "XBA2",   
                                    "XP2", "SO2", "SNO2", "SNH2", "SND2", "XND2",  
                                    "SI3", "SS3", "XI3", "XS3", "XBH3", "XBA3",   
                                    "XP3", "SO3", "SNO3", "SNH3", "SND3", "XND3",  
                                    "SI4", "SS4", "XI4", "XS4", "XBH4", "XBA4",   
                                    "XP4", "SO4", "SNO4", "SNH4", "SND4", "XND4",  
                                    "SI5", "SS5", "XI5", "XS5", "XBH5", "XBA5",   
                                    "XP5", "SO5", "SNO5", "SNH5", "SND5", "XND5",  
                                    "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10"], index=[0],  
                             dtype='float')  
    
    # 将初始条件赋值给DataFrame的第一行（时间t=0）  
    system_data_ode.loc[0, "time"] = tStart  # 设置时间  
 
    for idx, val in enumerate(C0):          
        system_data_ode.iloc[0, idx + 1] = val  # 从第二列开始赋值  

   
    # 求解ODE  
    # 初始化计数器   
    i = 1
    while r.successful() and r.t < tEnd:
        #执行积分步骤
        r.integrate(r.t + dt)
        #存储结果
        system_data_ode.loc[i] = [r.t] + [r.y[j] for j in range(len(r.y))]
        #更新
        i += 1
           
    
    # 稳定状态结果  
    Rows = ['SI', 'SS', 'XI', 'XS', 'XBH', 'XBA', 'XP', 'SO', 'SNO', 'SNH', 'SND', 'XND']  
    Columns = ['Tank1', 'Tank2', 'Tank3', 'Tank4', 'Tank5']  
   
    Steady_State = pd.DataFrame(index=Rows, columns=Columns)  
    for row in Rows:
        tank = 1
        for column in Columns:
            Steady_State.loc[row,column] = system_data_ode[row+str(tank)].iloc[-1]
            tank += 1
    print("稳定状态的五个池：\n",Steady_State)


    # 创建子图进行绘图  
    fig, axs = plt.subplots(6, 2, figsize=(10, 12))  
    plt.minorticks_on()  # 启用次刻度  

    # 定义 Tank 的数量  
    num_tanks = 5  

    # 绘制溶解惰性有机物 (SI)  
    for tank in range(1, num_tanks + 1):  
        axs[0, 0].plot(system_data_ode["time"], system_data_ode[f"SI{tank}"], label=f'Tank {tank}')  
    axs[0, 0].legend(fontsize=10, ncol=2)  
    axs[0, 0].set_xlabel(r'$t [d]$', fontsize=14)  
    axs[0, 0].set_ylabel(r'$S_I \mathrm{[g/m^3]}$', fontsize=14)  

    # 绘制易降解底物 (SS)  
    for tank in range(1, num_tanks + 1):  
        axs[0, 1].plot(system_data_ode["time"], system_data_ode[f"SS{tank}"], label=f'Tank {tank}')  
    axs[0, 1].legend(fontsize=10, ncol=2)  
    axs[0, 1].set_xlabel(r'$t [d]$', fontsize=14)  
    axs[0, 1].set_ylabel(r'$S_S \mathrm{[g/m^3]}$', fontsize=14)  

    # 绘制颗粒惰性有机物 (XI)  
    for tank in range(1, num_tanks + 1):  
        axs[1, 0].plot(system_data_ode["time"], system_data_ode[f"XI{tank}"], label=f'Tank {tank}')  
    axs[1, 0].legend(fontsize=10, ncol=2)  
    axs[1, 0].set_xlabel(r'$t [d]$', fontsize=14)  
    axs[1, 0].set_ylabel(r'$X_I \mathrm{[g/m^3]}$', fontsize=14)  

    # 绘制缓慢降解底物 (XS)  
    for tank in range(1, num_tanks + 1):  
        axs[1, 1].plot(system_data_ode["time"], system_data_ode[f"XS{tank}"], label=f'Tank {tank}')  
    axs[1, 1].legend(fontsize=10, ncol=2)  
    axs[1, 1].set_xlabel(r'$t [d]$', fontsize=14)  
    axs[1, 1].set_ylabel(r'$X_S \mathrm{[g/m^3]}$', fontsize=14)  

    # 绘制活性异养生物质 (XBH)  
    for tank in range(1, num_tanks + 1):  
        axs[2, 0].plot(system_data_ode["time"], system_data_ode[f"XBH{tank}"], label=f'Tank {tank}')  
    axs[2, 0].legend(fontsize=10, ncol=2)  
    axs[2, 0].set_xlabel(r'$t [d]$', fontsize=14)  
    axs[2, 0].set_ylabel(r'$X_{BH} \mathrm{[g/m^3]}$', fontsize=14)  

    # 绘制活性自养生物质 (XBA)  
    for tank in range(1, num_tanks + 1):  
        axs[2, 1].plot(system_data_ode["time"], system_data_ode[f"XBA{tank}"], label=f'Tank {tank}')  
    axs[2, 1].legend(fontsize=10, ncol=2)  
    axs[2, 1].set_xlabel(r'$t [d]$', fontsize=14)  
    axs[2, 1].set_ylabel(r'$X_{BA} \mathrm{[g/m^3]}$', fontsize=14)  

    # 绘制生物降解产物 (XP)  
    for tank in range(1, num_tanks + 1):  
        axs[3, 0].plot(system_data_ode["time"], system_data_ode[f"XP{tank}"], label=f'Tank {tank}')  
    axs[3, 0].legend(fontsize=10, ncol=2)  
    axs[3, 0].set_xlabel(r'$t [d]$', fontsize=14)  
    axs[3, 0].set_ylabel(r'$X_{P} \mathrm{[g/m^3]}$', fontsize=14)  

    # 绘制氧气 (SO)  
    for tank in range(1, num_tanks + 1):  
        axs[3, 1].plot(system_data_ode["time"], system_data_ode[f"SO{tank}"], label=f'Tank {tank}')  
    axs[3, 1].legend(fontsize=10, ncol=2)  
    axs[3, 1].set_xlabel(r'$t [d]$', fontsize=14)  
    axs[3, 1].set_ylabel(r'$S_O \mathrm{[g/m^3]}$', fontsize=14)  

    # 绘制硝酸盐和亚硝酸盐氮 (SNO)  
    for tank in range(1, num_tanks + 1):  
        axs[4, 0].plot(system_data_ode["time"], system_data_ode[f"SNO{tank}"], label=f'Tank {tank}')  
    axs[4, 0].legend(fontsize=10, ncol=2)  
    axs[4, 0].set_xlabel(r'$t [d]$', fontsize=14)  
    axs[4, 0].set_ylabel(r'$S_{NO} \mathrm{[g/m^3]}$', fontsize=14)  

    # 绘制铵离子/氨氮 (SNH)  
    for tank in range(1, num_tanks + 1):  
        axs[4, 1].plot(system_data_ode["time"], system_data_ode[f"SNH{tank}"], label=f'Tank {tank}')  
    axs[4, 1].legend(fontsize=10, ncol=2)  
    axs[4, 1].set_xlabel(r'$t [d]$', fontsize=14)  
    axs[4, 1].set_ylabel(r'$S_{NH} \mathrm{[g/m^3]}$', fontsize=14)  

    # 绘制可降解有机氮 (SND)  
    for tank in range(1, num_tanks + 1):  
        axs[5, 0].plot(system_data_ode["time"], system_data_ode[f"SND{tank}"], label=f'Tank {tank}')  
    axs[5, 0].legend(fontsize=10, ncol=2)  
    axs[5, 0].set_xlabel(r'$t [d]$', fontsize=14)  
    axs[5, 0].set_ylabel(r'$S_{ND} \mathrm{[g/m^3]}$', fontsize=14)  

    # 绘制颗粒可降解有机氮 (XND)  
    for tank in range(1, num_tanks + 1):  
        axs[5, 1].plot(system_data_ode["time"], system_data_ode[f"XND{tank}"], label=f'Tank {tank}')  
    axs[5, 1].legend(fontsize=10, ncol=2)  
    axs[5, 1].set_xlabel(r'$t [d]$', fontsize=14)  
    axs[5, 1].set_ylabel(r'$X_{ND} \mathrm{[g/m^3]}$', fontsize=14)  
        
    # 设置图形布局  
    fig.tight_layout(pad=0.4, w_pad=1, h_pad=1.0)  
    plt.show()  # 显示图表   

if __name__=="__main__":
    main()



    











    


