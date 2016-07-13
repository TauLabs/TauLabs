
%% full model on desired output with parameter learning

% this is a dynamic model that takes control inputs, applies a LPF
% and then an adjustable gain to generate the rotation rate. it
% allows learning the time constant of the LPF as well as the
% gain for each axis. it must be sufficiently excited by inputs
% to converge.

% parameters
syms br bp by1 by2 tau Ts real;
% state variables
syms wr wp wy nur nup nuy biasr biasp biasy real;
% inputs
syms ur up uy real;

x = [wr nur biasr wp nup biasp wy nuy biasy br bp by1 by2 tau]';
u_in = [ur up uy]';

% br, bp, by1, by2 and tau are all stored as log of those variables
% to prevent them going negative. this is a helper variable for tau
tau_s = exp(tau);

% define continuous time dynamics for each of the axes
Ac_r = [0 exp(br) 0; 0 -1/tau_s -1/tau_s; 0 0 0]; Bc_r = [0; 1/tau_s; 0];
Ac_p = [0 exp(bp) 0; 0 -1/tau_s -1/tau_s; 0 0 0]; Bc_p = [0; 1/tau_s; 0];
Ac_y = [0 exp(by1)-exp(by2) -exp(by2); 0 -1/tau_s -1/tau_s; 0 0 0]; Bc_y = [by2; 1/tau_s; 0];

% convert to discrete time and get complete system dynamics. these would
% be seperable except for the shared tau term
R = expm([Ac_r Bc_r; 0 0 0 0]*Ts);
A = R(1:3,1:3);
B = R(1:3,4);
R = expm([Ac_p Bc_p; 0 0 0 0]*Ts);
A(4:6,4:6) = R(1:3,1:3);
B(4:6,2) = R(1:3,4);
R = expm([Ac_y Bc_y; 0 0 0 0]*Ts);
A(7:9,7:9) = R(1:3,1:3);
B(7:9,3) = R(1:3,4);

% parameters dynamics are to remain constant
A(10:numel(x),10:numel(x)) = eye(5);
B(10:numel(x),1) = zeros(5,1);

f = A * x + B * u_in

h = [wr wp wy]'

F = simplify(jacobian(f, x), 100)

H = jacobian(h, x)

N = length(x)

%% useful substitutions

% because we represent so many parameters as exponentials we can
% substitute that before expanding the following equations

syms e_br e_bp e_by1 e_by2 e_tau ets s_ets s_eby2 real

F2 = F;
for i = 1:numel(F)
    F2(i) = F(i);

    F2(i) = subs(F2(i), exp(-Ts*exp(-tau)), ets);
    F2(i) = subs(F2(i), exp(Ts*exp(-tau)),1/ets);
     
    F2(i) = subs(F2(i), exp(tau - (Ts*exp(-tau))/2), e_tau*s_ets);
    F2(i) = subs(F2(i), exp(- tau - Ts*exp(-tau)), ets / e_tau);
 
    F2(i) = subs(F2(i), sinh((Ts*exp(-tau))/2), (1-ets)/(2*s_ets));

    F2(i) = subs(F2(i), sinh(by2/2), (e_by2-1)/(2*s_eby2));
    F2(i) = subs(F2(i), exp(by2/2), s_eby2);

    F2(i) = subs(F2(i), exp(br), e_br);
    F2(i) = subs(F2(i), exp(bp), e_bp);
    F2(i) = subs(F2(i), exp(by1), e_by1);
    F2(i) = subs(F2(i), exp(by2), e_by2);
    
    F2(i) = subs(F2(i), exp(tau), e_tau);
    F2(i) = subs(F2(i), exp(-tau), 1/e_tau);
    
    F2(i) = subs(F2(i), exp(br + tau), e_br*e_tau);
    F2(i) = subs(F2(i), exp(bp + tau), e_bp*e_tau);
    F2(i) = subs(F2(i), exp(by1 + tau), e_by1*e_tau);
    F2(i) = subs(F2(i), exp(by2 + tau), e_by2*e_tau);
end

%% generate the symbolic code

syms P_1_1 P_1_2 P_1_3 P_1_4 P_1_5 P_1_6 P_1_7 P_1_8 P_1_9 P_1_10 P_1_11 P_1_12 P_1_13 P_1_14  real
syms P_2_2 P_2_3 P_2_4 P_2_5 P_2_6 P_2_7 P_2_8 P_2_9 P_2_10 P_2_11 P_2_12 P_2_13 P_2_14  real
syms P_3_3 P_3_4 P_3_5 P_3_6 P_3_7 P_3_8 P_3_9 P_3_10 P_3_11 P_3_12 P_3_13 P_3_14  real
syms P_4_4 P_4_5 P_4_6 P_4_7 P_4_8 P_4_9 P_4_10 P_4_11 P_4_12 P_4_13 P_4_14  real
syms P_5_5 P_5_6 P_5_7 P_5_8 P_5_9 P_5_10 P_5_11 P_5_12 P_5_13 P_5_14  real
syms P_6_6 P_6_7 P_6_8 P_6_9 P_6_10 P_6_11 P_6_12 P_6_13 P_6_14  real
syms P_7_7 P_7_8 P_7_9 P_7_10 P_7_11 P_7_12 P_7_13 P_7_14  real
syms P_8_8 P_8_9 P_8_10 P_8_11 P_8_12 P_8_13 P_8_14  real
syms P_9_9 P_9_10 P_9_11 P_9_12 P_9_13 P_9_14  real
syms P_10_10 P_10_11 P_10_12 P_10_13 P_10_14  real
syms P_11_11 P_11_12 P_11_13  P_11_14  real
syms P_12_12 P_12_13 P_12_14  real
syms P_13_13 P_13_14  real
syms P_14_14  real

syms s_a real

syms gyro_x gyro_y gyro_z real

syms Q_1 Q_2 Q_3 Q_4 Q_5 Q_6 Q_7 Q_8 Q_9 Q_10 Q_11 Q_12 Q_13 Q_14  real

y = [gyro_x gyro_y gyro_z]' - h;

P=[
P_1_1 P_1_2 P_1_3 P_1_4 P_1_5 P_1_6 P_1_7 P_1_8 P_1_9 P_1_10 P_1_11 P_1_12 P_1_13 P_1_14 ;
0     P_2_2 P_2_3 P_2_4 P_2_5 P_2_6 P_2_7 P_2_8 P_2_9 P_2_10 P_2_11 P_2_12 P_2_13 P_2_14 ;
0     0     P_3_3 P_3_4 P_3_5 P_3_6 P_3_7 P_3_8 P_3_9 P_3_10 P_3_11 P_3_12 P_3_13 P_3_14 ;
0     0     0     P_4_4 P_4_5 P_4_6 P_4_7 P_4_8 P_4_9 P_4_10 P_4_11 P_4_12 P_4_13 P_4_14 ;
0     0     0     0     P_5_5 P_5_6 P_5_7 P_5_8 P_5_9 P_5_10 P_5_11 P_5_12 P_5_13 P_5_14 ;
0     0     0     0     0     P_6_6 P_6_7 P_6_8 P_6_9 P_6_10 P_6_11 P_6_12 P_6_13 P_6_14 ;
0     0     0     0     0     0     P_7_7 P_7_8 P_7_9 P_7_10 P_7_11 P_7_12 P_7_13 P_7_14 ;
0     0     0     0     0     0     0     P_8_8 P_8_9 P_8_10 P_8_11 P_8_12 P_8_13 P_8_14 ;
0     0     0     0     0     0     0     0     P_9_9 P_9_10 P_9_11 P_9_12 P_9_13 P_9_14 ;
0     0     0     0     0     0     0     0     0     P_10_10 P_10_11 P_10_12 P_10_13 P_10_14 ;
0     0     0     0     0     0     0     0     0     0       P_11_11 P_11_12 P_11_13 P_11_14 ;
0     0     0     0     0     0     0     0     0     0       0       P_12_12 P_12_13 P_12_14 ;
0     0     0     0     0     0     0     0     0     0       0       0       P_13_13 P_13_14 ;
0     0     0     0     0     0     0     0     0     0       0       0       0       P_14_14 ;
];

x = [wr nur biasr wp nup biasp wy nuy biasy br bp by1 by2 tau]';

% remove cross coupling terms in the covariance
P(1:3,[4:9 11:13]) = 0;
P(4:6,[7:9 10 12:13]) = 0;
P(7:9,[10:11]) = 0;

% make it symmetrical
for(i=2:N)
    for (j=1:i-1)
        P(i,j)=P(j,i);
    end
end
       
Q = diag([Q_1 Q_2 Q_3 Q_4 Q_5 Q_6 Q_7 Q_8 Q_9 Q_10 Q_11 Q_12 Q_13 Q_14]);

%P2 = simplify((F*P*F') + Q, 'Criterion', 'preferReal', 'IgnoreAnalyticConstraints', true, 'Steps', 100);
P2 = F2*P*F2' + Q;

% any variables that are not updated in the covariance propagation will
% tend towards zero, also any assigned to zero need not be tracked.
P_temp = triu(P);
P2_temp = triu(P2);
P_idx = find(P_temp ~= P2_temp & P_temp ~= 0);
P_zero = find(P_temp == P2_temp | P_temp == 0);

% zero out additional terms so they don't show up in equations
P(P_zero) = 0; P = triu(P) + triu(P,1)';
P2 = simplify((F2*P*F2') + Q);
P2(P_zero) = 0; P2 = triu(P2) + triu(P2,1)';

% update equations
R = diag([s_a s_a s_a]); 
S = H*P*H' + R;

% remove coupling between axes for efficiency. from the above equation
% we can see that S_1 should be P[0][0] + s_a, etc
syms S_1 S_2 S_3 real
S = diag([S_1 S_2 S_3])
  
K = P*H'/S;


x_new = x + K*y;

I = eye(length(K));
P3 = (I - K*H)*P;  % Output state covariance

%% create strings for update equations


fid = fopen('autotune_filter.c','w');


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
                Pstrings{i,j} = strrep(Pstrings{i,j},'P','D');
                
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

fid = fopen('autotune_filter.c','a');

for x = {f x_new}
    
    fprintf(fid, '\n\n\n\n\t// X update\n');
    
    Pstrings2=cell(N,1);
    for i=1:N
        x{1}(i) = subs(x{1}(i), exp(br), e_br);
        x{1}(i) = subs(x{1}(i), exp(bp), e_bp);
        x{1}(i) = subs(x{1}(i), exp(by1), e_by1);
        x{1}(i) = subs(x{1}(i), exp(by2), e_by2);
        x{1}(i) = subs(x{1}(i), exp(tau), e_tau);
        x{1}(i) = subs(x{1}(i), exp(-Ts*exp(-tau)), ets);

        Pstrings2{i} = char(x{1}(i));

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

fid = fopen('autotune_filter.c','a');
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
