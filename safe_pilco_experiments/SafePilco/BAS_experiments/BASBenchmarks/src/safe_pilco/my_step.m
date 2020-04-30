function x2 = my_step(x1, action)
global Z1m
%x1 = [20,20,18,18]';
%action = 2;

t = 1;
if size(x1,1) < 6
    d = [100*randn(t,1) + 500, 100*randn(t,1) + 500,  ones(t,1)];
else
    Tout = 1*randn(t,1) + 9;
    Thall= 1*randn(t,1) + 15;
    CO2_1= 100*randn(t,1) + 500;
    CO2_2= 100*randn(t,1) + 500;
    Trwr1= 5*randn(t,1) + 35;
    Trwr2= 5*randn(t,1) + 35;
    d = [Tout Thall CO2_1 CO2_2 Trwr1 Trwr2 ones(t,1)];
end

x2 = runModel(Z1m, x1, action, d, t);

end
