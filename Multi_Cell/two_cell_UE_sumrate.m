clc;
clear all;
close all;


power_max = 10^(-0.3); % Maximum transmit power level for UEs
UE_location = [10,20,15,30]; % User locations in the Network
Interfere_locations = [70,85,80,90]; % User interference locations
K = 2; % Users per gNB
M = 2;
MRB = 50;
eps = 100;
cycles = 100;
NR = 2; % Number of gNBs

%%%% SET ATMOST 1 flag depending on the scenario %%%%%
SA_finer =1;
SA =0;
HA = 0;
No_SI =0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

MSD_causing_CC = 2;

if SA_finer == 1
    n_m = 2;    % n_m = 2 for 25 resolution
elseif SA == 1 || No_SI == 1 || HA==1
    n_m =1;
end

%insert .mat file generated from its corresponding .py file
data=load("xxxxxxxxxxxxxx.mat");

power=data.actions(:,:,1:K,:);
CCs=data.actions(:,:,K+1:K*((M-1)+n_m),:);
RBs=zeros(eps,cycles,K*MRB*M,NR);
rate=zeros(eps,cycles,K,NR);
n_var=1e-10;
BW_SCC=18e4;
era=zeros(NR,K);
G_dB = 30;
G = 10^(0.1*G_dB);

for e=1:eps
    for c=1:cycles
        for b=1:NR % Loop over gNBs
            j=0;
            for i=1:K
                for k=1:floor(MRB/K)
                    RBs(e,c,(i-1)*MRB*M+(j)*MRB+k,b)=1;
                end
            end
            for CC=2:M
                if MSD_causing_CC == CC
                    for i = 1:K
                        for n_m_idx = 1:n_m
                            if CCs(e,c,(i-1)*((M-1)+(n_m-1))+(n_m_idx-1)+(CC-1),b)==1
                                if mod(MRB,K*n_m) == 0
                                    n_m_len = ceil(MRB/(n_m*K));
                                else
                                    n_m_len = abs((MRB/n_m)*(n_m_idx-1+(CC-2)) - ceil(MRB/(n_m*K)));
                                end
                                kd = 1:n_m_len;
                                RBs(e,c,(i-1)*MRB*M+(CC-1)*MRB+ (n_m_idx-1)*n_m_len +kd,b)=1;
                            end
                        end
                    end
                else
                    if MSD_causing_CC > CC
                        CC_idx = CC;
                    else
                        CC_idx = CC + (n_m-1);
                    end
                    x=0;
                    for i=CC_idx-1:(M-1)+(n_m-1):((M-1)+(n_m-1))*K
                        x=x+CCs(e,c,i,b);
                    end
                    for i=1:K
                        if CCs(e,c,(i-1)*((M-1)+(n_m-1))+(CC_idx-1),b)==1
                            y=floor(MRB/x);
                            v=0;
                            for k=1:MRB
                                if RBs(e,c,(i-1)*MRB*M+(CC-1)*MRB+k,b)==0
                                    v=k;
                                    break
                                end
                            end
                            kd=1:y;
                            RBs(e,c,(i-1)*MRB*M+(CC-1)*MRB+kd+v-1,b)=1;
                        end
                    end
                end
            end
        end
    end
end

d1=sum(Interfere_locations(1:K).^2);
d2=sum(Interfere_locations(K+1:end).^2);
d=[d1 d2];
for b = 1:NR
    for i = 1:K
        global_ue_index = (b - 1) * K + i;
        path_loss = 1 / ((4*pi*UE_location(global_ue_index)) / 0.0857143)^2;
        path_loss_interferer = (1 / ((4*pi*d(b)) / 0.0857143)^2);
        N = (G * power(:,:,i,b) * path_loss);
        D = (G * power(:,:,i,mod(b,NR)+1) * path_loss_interferer)+sum(RBs(:,:,(i-1).*M*MRB+1:i.*M*MRB),3)*n_var;
        SINR = N./D;

        % Calculate rate with interference
        rate(:,:,i,b) = sum(RBs(:, :, (i-1)*M*MRB+1:i*M*MRB,b),3) .* BW_SCC .* log2(1 + SINR);
    end
end

avgrate=sum(rate,2)./cycles;
frate=sum(avgrate,3);

for b=1:NR % Loop over gNBs
    for i=1:K
        figure();
        plot(1:eps,avgrate(:,1,i,b)./1e6,'ro-','LineWidth',2);
        title(['Sum throughput of UE',num2str(i),' at gNB',num2str(b)]);
        xlabel('Episode index','FontSize',20);
        ylabel('Sum throughput (Mbps)','FontSize',20);
        legend("RL based allocation")
        grid on;
    end
end
% Total sum rate per gNB
sum_rate_per_gNB = zeros(eps, NR);
for b = 1:NR
    sum_rate_per_gNB(:, b) = sum(avgrate(:, 1, :, b), 3); % Sum over UEs at gNB b
end

% Plot per-gNB total sum rate
for b = 1:NR
    figure();
    plot(1:eps, sum_rate_per_gNB(:, b)./1e6, 'b.-', 'LineWidth', 2);
    title(['Total Sum Throughput at gNB', num2str(b)]);
    xlabel('Episode index', 'FontSize', 14);
    ylabel('Sum Throughput (Mbps)', 'FontSize', 14);
    legend(['gNB', num2str(b)]);
    grid on;
end


