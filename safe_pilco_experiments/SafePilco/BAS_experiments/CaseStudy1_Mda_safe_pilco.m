% Case study 1: Deterministic Model with additive 
% disturbances
% author: Nathalie Cauchi
% -------------------------------------------------------
% x_d[k+1] = Ax_d[k] + Bu[k] + F_dad[k] + Q_d
% y_d[k]   = [1 0 0 0; 0 1 0 0]
% x_d = [T_z1 T_z1 T_rw,rad1 T_rw,rad2]^T
% u   = T_sa
% d   = [CO2_1 CO2_2]^T
% -------------------------------------------------------
% -------------------------------------------------------

clc; clear; 
% Load parameters needed to build models
BASParameters;
Ts = 15;                 % Sample time (minutes)
T  = 4*24*3;             % Time horizon (minutes)

% Steady state values
Tswb  = Boiler.Tswbss;              
Tsp   = Zone1.Tsp;             
Twss  = Zone1.Twss;
Trwrss= Radiator.Trwrss;
Pout1 = Radiator.Zone1.Prad;    
Pout2 = Radiator.Zone2.Prad;    
m     = Zone1.m;
w     = Radiator.w_r;

% Defining Deterministic Model corresponding matrices
Ac = zeros(4,4);
Ac(1,1) = -(1/(Zone1.Rn*Zone1.Cz))-((Pout1*Radiator.alpha2 )/(Zone1.Cz)) - ((m*Materials.air.Cpa)/(Zone1.Cz));
Ac(1,3) = (Pout1*Radiator.alpha2 )/(Zone1.Cz);
Ac(2,2) = -(1/(Zone2.Rn*Zone2.Cz))-(Pout2*Radiator.alpha2 )/(Zone2.Cz) - (m*Materials.air.Cpa)/(Zone2.Cz);
Ac(2,4) = (Pout2*Radiator.alpha2 )/(Zone2.Cz);
Ac(3,1) = (Radiator.k1);
Ac(3,3) = -(Radiator.k0*w)-Radiator.k1;
Ac(4,2) = (Radiator.k1);
Ac(4,4) = -(Radiator.k0*w) -Radiator.k1;

Bc = [(m*Materials.air.Cpa)/(Zone1.Cz) (m*Materials.air.Cpa)/(Zone2.Cz) 0 0]';

d = (Twss/(Zone1.Rn*Zone1.Cz))+ (Radiator.alpha1 )/(Zone1.Cz) + (Zone1.zeta/Zone1.Cz);
g = ((Twss-1)/(Zone1.Rn*Zone1.Cz))+ (Radiator.alpha1 )/(Zone1.Cz) + (Zone2.zeta/Zone2.Cz);
k = (Radiator.k0*w*Tswb) ;
p = (Radiator.k0*w*Tswb);

Fc      = zeros(4,3);
Fc(1,1) = Zone1.mu/Zone1.Cz;
Fc(2,2) = Zone2.mu/Zone2.Cz;
Fc(1,3) = d;
Fc(2,3) = g;
Fc(3,3) = k;
Fc(4,3) = p;

Cc = diag([1 1 1 1]);
global Z1m
Z1m = createModel;
% Defining Deterministic Model corresponding matrices
Z1m=InitialiseModel(Z1m,'l','d',Ac,Bc,Cc,Fc,[],[],Ts,0);
Z1m=createSymbModel(Z1m);

% Defining input signal 
% Tsa = 18.*ones(T,1);
% Tsa(32:48)=20.*ones(17,1);
% Tsa(52:72)=20.*ones(21,1);
% Tsa(32+96:48+96)=20.*ones(17,1);
% Tsa(52+96:72+96)=20.*ones(21,1);
% Tsa(32+96*2:48+96*2)=20.*ones(17,1);
% Tsa(52+96*2:72+96*2)=20.*ones(21,1);
% Tsa(32+96*3:48+96*3)=20.*ones(17,1);
% Tsa(52+96*3:72+96*3)=20.*ones(21,1);

% Defining disturbance signal 
% D   = [100*randn(T,1) + 500, 100*randn(T,1) + 500,  ones(T,1)];    

% % Simulate model over given time horizon T 
% Tz1_y      = runModel(Z1m,[ 20  20 Trwrss Trwrss]', Tsa,D,T);
% 
% % Plot Results
% title={{'Zone 1 temperature (^oC)'},{'Zone 2 temperature (^oC)'}};
% plotFigures(1:T,Tz1_y(1:2,1:T),title); 

