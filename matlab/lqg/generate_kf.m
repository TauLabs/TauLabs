
%% full model on desired output with parameter learning

% this is a dynamic model that takes control inputs, applies a LPF
% and then an adjustable gain to generate the rotation rate. it
% allows learning the time constant of the LPF as well as the
% gain for each axis. it must be sufficiently excited by inputs
% to converge.

% state variables:
%   - w - rotation
%   - u - force

% fixed parameters (learned by autotune using similar KF)
%   - b - gain
%   - bd - direct gain
%   - tau

% the actuator inputs (four motor speeds)

syms w u_in b bd tau bias Ts real;

x = [w u bias]';

% state transition matrix
A = [1          Ts*exp(b)       -Ts*exp(b); ...
     0   exp(tau)/(exp(tau)+Ts)     0; ...
     0             0                1];

B = [Ts*exp(bd); Ts/(exp(tau) + Ts); 0];

f = A * x + B * u_in;

h = [w]'

F = simplify(jacobian(f, x), 100)

H = jacobian(h, x)

N = length(x)

%% generate the symbolic code

syms P_1_1 P_1_2 P_1_3  real
syms P_2_2 P_2_3  real
syms P_3_3  real

syms s_a real

syms gyro real

syms Q_1 Q_2 Q_3 real

y = [gyro]' - h;

P=[
P_1_1 P_1_2 P_1_3;
0     P_2_2 P_2_3;
0     0     P_3_3];

% we can use this variable to reduce the unused terms out of the equations
% below instead of storing all of the P values.
P_idx = find(P(:));

% make it symmetrical
for(i=2:N)
    for (j=1:i-1)
        P(i,j)=P(j,i);
    end
end
       
Q = diag([Q_1 Q_2 Q_3]);

P2 = simplify((F*P*F') + Q);

% update equations
R = diag([s_a]); 
S = H*P*H' + R;

% remove coupling between axes for efficiency. from the above equation
% we can see that S_1 should be P[0][0] + s_a, S_2 is P[1][1] + s_a
% etc
syms S real
  
K = P*H'/S;

x_new = x + K*y;

I = eye(length(K));
P3 = (I - K*H)*P;  % Output state covariance

%% create strings for update equations


fid = fopen('torque_kf_filter.c','w');


for Pnew = {P2, P3}
    nummults=0;
    numadds =0;

    fprintf(fid, '\n\n\n\n\t// Covariance calculation\n');
    Pnew = simplify(Pnew{1});


    Pstrings=cell(N,N);
    for i=1:N
        for j=i:N
            if P(i,j) == 0
                Pstrings{i,j} = num2str(0);
            else           
                % replace with lots of precalculated constants or invalid C
                Pstrings{i,j} = char(Pnew(i,j));
                Pstrings{i,j} = strrep(Pstrings{i,j},'Ts^2','Tsq');
                Pstrings{i,j} = strrep(Pstrings{i,j},'Ts^3','Tsq3');
                Pstrings{i,j} = strrep(Pstrings{i,j},'Ts^4','Tsq4');
                Pstrings{i,j} = strrep(Pstrings{i,j},'P','D');
                Pstrings{i,j} = strrep(Pstrings{i,j},'exp(b)','e_b');
                Pstrings{i,j} = strrep(Pstrings{i,j},'exp(bd)','e_bd');
                Pstrings{i,j} = strrep(Pstrings{i,j},'exp(2*b)','(e_b*e_b)');
                Pstrings{i,j} = strrep(Pstrings{i,j},'exp(2*bd)','(e_bd*e_bd)');
                Pstrings{i,j} = strrep(Pstrings{i,j},'exp(2*tau)','e_tau2');
                Pstrings{i,j} = strrep(Pstrings{i,j},'exp(3*tau)','e_tau3');
                Pstrings{i,j} = strrep(Pstrings{i,j},'exp(4*tau)','e_tau4');
                Pstrings{i,j} = strrep(Pstrings{i,j},'exp(tau)','e_tau');
                Pstrings{i,j} = strrep(Pstrings{i,j},'s_a^2','s_a2');
                Pstrings{i,j} = strrep(Pstrings{i,j},'s_a^3','s_a3');
                Pstrings{i,j} = strrep(Pstrings{i,j},'u1^2','(u1*u1)');
                Pstrings{i,j} = strrep(Pstrings{i,j},'u2^2','(u2*u2)');
                Pstrings{i,j} = strrep(Pstrings{i,j},'u3^2','(u3*u3)');
                Pstrings{i,j} = strrep(Pstrings{i,j},'bias1^2','(bias1*bias1)');
                Pstrings{i,j} = strrep(Pstrings{i,j},'bias2^2','(bias2*bias2)');
                Pstrings{i,j} = strrep(Pstrings{i,j},'bias3^2','(bias3*bias3)');
                Pstrings{i,j} = strrep(Pstrings{i,j},'u1_in^2','(u1_in*u1_in)');
                Pstrings{i,j} = strrep(Pstrings{i,j},'u2_in^2','(u2_in*u2_in)');
                Pstrings{i,j} = strrep(Pstrings{i,j},'u3_in^2','(u3_in*u3_in)');
                Pstrings{i,j} = strrep(Pstrings{i,j},'(Ts + e_tau)^2','Ts_e_tau2');
                Pstrings{i,j} = strrep(Pstrings{i,j},'(Ts + e_tau)^3','Ts_e_tau3');
                Pstrings{i,j} = strrep(Pstrings{i,j},'(Ts + e_tau)^4','Ts_e_tau4');
                Pstrings{i,j} = strrep(Pstrings{i,j},'exp(b1 + tau)','(e_b1*e_tau)');
                Pstrings{i,j} = strrep(Pstrings{i,j},'exp(b2 + tau)','(e_b2*e_tau)');
                Pstrings{i,j} = strrep(Pstrings{i,j},'exp(b1 + 2*tau)','(e_b1*e_tau2)');
                Pstrings{i,j} = strrep(Pstrings{i,j},'exp(b2 + 2*tau)','(e_b2*e_tau2)');
                Pstrings{i,j} = strrep(Pstrings{i,j},'exp(b3 + 2*tau)','(e_b3*e_tau2)');
                Pstrings{i,j} = strrep(Pstrings{i,j},'(Ts/(Ts + e_tau) - 1)^2','powf(Ts/(Ts + e_tau) - 1,2)');
                
                for n1 = N:-1:1
                    Pstrings{i,j} = strrep(Pstrings{i,j},sprintf('Q_%d',n1),sprintf('Q[%d]', n1-1));
                    Pstrings{i,j} = strrep(Pstrings{i,j},sprintf('S_%d',n1),sprintf('S[%d]', n1-1));
                end

                for n1 = 1:N
                    for n2 = 1:N
                        Pstrings{i,j} = strrep(Pstrings{i,j},sprintf('D_%d_%d^2',n1,n2),sprintf('D_%d_%d*D_%d_%d',n1,n2,n1,n2));
                    end
                end
            end
            s1 = sprintf('P_%d_%d = ',i,j);
            Pstrings{i,j} = [s1, Pstrings{i,j}, ';'];
        end
    end

    Pstrings = Pstrings(P_idx);
    
    for i = 1:length(P_idx)
        s_out = Pstrings{i};
        
        % replace indexes into array with indexes into spare linear
        % array of non-zero elements
        for j = length(P_idx):-1:1
            % index backwards to make sure the big numbers get replaced
            % first
            [k, l] = ind2sub([N N], P_idx(j));
            s1 = sprintf('_%d_%d', k, l);
            s2 = sprintf('[%d]',j-1);
            s_out = strrep(s_out, s1, s2);
        end
        
        nummults=nummults + length(strfind(s_out,'*'));
        numadds=numadds + length(strfind(s_out,'+'));
        numadds=numadds + length(strfind(s_out,'-'));
        
        strings_out{i} = s_out; % for debugging
        fprintf(fid,'\t%s\n',s_out);
    end
    
    fprintf('Multplies: %d Adds: %d\n', nummults, numadds);
    
end

fclose(fid);

%% Create strings for update of x

fid = fopen('torque_kf_filter.c','a');

for x = {f x_new}
    
    fprintf(fid, '\n\n\n\n\t// X update\n');
    
    Pstrings2=cell(N,1);
    for i=1:N
        Pstrings2{i} = char(x{1}(i));
        Pstrings2{i} = strrep(Pstrings2{i},'Ts^2','Tsq');
        Pstrings2{i} = strrep(Pstrings2{i},'exp(b)','e_b');
        Pstrings2{i} = strrep(Pstrings2{i},'exp(bd)','e_bd');
        Pstrings2{i} = strrep(Pstrings2{i},'exp(2*tau)','e_tau*e_tau');
        Pstrings2{i} = strrep(Pstrings2{i},'exp(tau)','e_tau');
        Pstrings2{i} = strrep(Pstrings2{i},'s_a^2','s_a2');
        Pstrings2{i} = strrep(Pstrings2{i},'s_a^3','s_a3');
        Pstrings2{i} = strrep(Pstrings2{i},'(Ts + e_tau)^2','Ts_e_tau2');
        Pstrings2{i} = strrep(Pstrings2{i},'S_1','S[0]');
        Pstrings2{i} = strrep(Pstrings2{i},'S_2','S[1]');
        Pstrings2{i} = strrep(Pstrings2{i},'S_3','S[2]');
        
        
        % replace indexes into array with indexes into spare linear
        % array of non-zero elements
        for j = length(P_idx):-1:1
            % index backwards to make sure the big numbers get replaced
            % first
            [k, l] = ind2sub([N N], P_idx(j));
            s1 = sprintf('_%d_%d', k, l);
            s2 = sprintf('[%d]',j-1);
            Pstrings2{i} = strrep(Pstrings2{i}, s1, s2);
        end
        
        s1 = sprintf('X[%d] = ',i-1);
        Pstrings2{i} = [s1, Pstrings2{i}, ';'];
    end
    
    
    for i=1:N;
        fprintf(fid,'\t%s\n',Pstrings2{i});
    end
        
    nummults=0;
    numadds =0;
    for i=1:N;
        nummults=nummults + length(strfind(Pstrings2{i},'*'));
        numadds=numadds + length(strfind(Pstrings2{i},'+'));
        numadds=numadds + length(strfind(Pstrings2{i},'-'));
    end
    fprintf('Multplies: %d Adds: %d\n', nummults, numadds);
end

fclose(fid);


%%

fid = fopen('torque_kf_filter.c','a');
fprintf(fid, '\n\n\n\n\t// P initialization\n');
for j = 1:length(P_idx)
    % index backwards to make sure the big numbers get replaced
    % first
    [k, l] = ind2sub([N N], P_idx(j));
    
    if k == l
        s_out = sprintf('P[%d] = q_init[%d];',j-1, k-1);
    else
        s_out = sprintf('P[%d] = 0.0f;',j-1);
    end
    fprintf(fid,'\t%s\n', s_out);
end
fclose(fid)
