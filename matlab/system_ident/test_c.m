
[X, P] = system_ident_ml();
%load P
N = length(Gyros.timestamp);
w = nan(N,3);
ud = nan(N,3);
B = nan(N,3);
tau = nan(N,1) - 5;
bias = nan(N,3);
noise = nan(N,3);
ac_idx = nan(N,1);

figure(1)

is = 8000;
for i = is:1:length(Gyros.timestamp)
    [~,idx] = min(abs(Gyros.timestamp(i) - ActuatorDesired.timestamp-0));
    
    if ActuatorDesired.Throttle(idx) < 0
        continue;
    end

    [X, P, noise(i,:)] = system_ident_ml(X,P,[Gyros.x(i) Gyros.y(i) Gyros.z(i)]', 1*[ActuatorDesired.Roll(idx) ActuatorDesired.Pitch(idx) ActuatorDesired.Yaw(idx)]',noise(i-1,:));
    w(i,:) = X(1:3);
    ud(i,:) = X(4:6);
    B(i,:) = X(7:9);
    tau(i) = X(10);
    bias(i,:) = X(11:13);
    
    ac_idx(i) = idx;
     
     if mod(i,3000) == 2999
        t = double(Gyros.timestamp(is:i) - Gyros.timestamp(is)) / 1000;
        
        h1(1) = subplot(321);
        plot(t,Gyros.x(is:i),t,w(is:i,1))
        title('Roll')
        legend('Gyro','Estimate')
        ylabel('deg/s')

        h1(2) = subplot(322);
        plot(t,Gyros.y(is:i),t,w(is:i,2))
        title('Pitch')
        legend('Gyro','Estimate')
        ylabel('deg/s')

        h1(3) = subplot(323);
        plot(t,ud(is:i,:))
        ylim([-1 1])
        title('Estimated Output')
        
        h1(4) = subplot(324);
        plot(t,B(is:i,:))
        %ylim([0 250])
        title('\beta')
        ylabel('Log Gain')
        legend('Roll','Pitch','Yaw')
        
        h1(5) = subplot(325);
        plot(t, exp(tau(is:i)))
        ylim([0 0.2])
        title('\tau')
        ylabel('Time (s)')
        
        h1(6) = subplot(326);
        plot(t, bias(is:i,:))
        title('bias')
%         
%         h(5) = subplot(515);
%         plot(t, bias(is:i))
%         title('bias')
%         ylim([-0.2,0.2])
%         
         linkaxes(h1,'x')
         xlabel('Time (s)')
         drawnow
     end
end
