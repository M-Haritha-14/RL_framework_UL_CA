clc;
clear all;
close all;

K = 2;
M = 2;
MRB = 50;
eps = 100;
cycles = 100;

power_max=10^(-0.3);
G_dB = 30;
G = 10^(0.1*G_dB);
n_var=1e-10;
BW_SCC=18e4;

UE_location= [25,35]; %[25,25];
MSD_CC_idx = [2,2];


MSD_causing_CC = 2;

%%%%%%%%%%%%%%%%%%% SET ATMOST 1 flag depending on the scenario %%%%%%%%%%%

SA_finer = 1; % Set this flag for "SI,SA,Res25","SI,SA,Res10"
SA = 0;       % Set this flag for "SI,SA,Res50"
HA = 0;       % Set this flag for "HA"
No_SI = 0;    % Set this flag for "No SI"

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if SA_finer == 1
    n_m = 2;    % n_m = 2 for "SI,SA,Res25", n_m = 5 for "SI,SA,Res10"
elseif SA == 1 | HA == 1
    n_m = 1;
elseif No_SI == 1
    n_m = 1;
end


%insert .mat file generated from its corresponding .py file
data=load("xxxxxxxxxxxxxx.mat");


power=data.actions(:,:,1:K); 
CCs=data.actions(:,:,K+1:K*((M-1)+n_m));
RBs=zeros(eps,cycles,K*MRB*M);
rate=zeros(eps,cycles,K);


activated_UEs_list = 1:K;

% RB algorithm
for e=1:eps
    %%
    % NOTE: 
    % To avoid online adaptation, update e with a value above 'eps' value
    % if e == 1500 % indicate episode at which UE exits network
    %     activated_UEs_list = activated_UEs_list(find(activated_UEs_list~=2)); %removes UE2
    % elseif e == 2000 % indicate episode at which UE re-enters network
    %     activated_UEs_list = [activated_UEs_list 2]; %adds UE2
    % end

    %%
    K_act = length(activated_UEs_list);

    for c=1:cycles
    j=0;
    for i=activated_UEs_list
         k=1:floor(MRB/K_act);
         RBs(e,c,(i-1)*MRB*M+(j)*MRB+k)=1;
    end
    for CC=2:M
        if MSD_causing_CC == CC && SA_finer == 1

            for n_m_idx = 1:n_m
                UE_CC_nm = zeros(1,K);
                
                for i = activated_UEs_list
                     UE_CC_nm(i) = CCs(e,c,(i-1)*((M-1)+(n_m-1))+(n_m_idx-1)+(CC-1));
                end

               n_m_len = ceil(MRB/(n_m*K_act));

               if n_m == 5
               
                   if UE_CC_nm(1) == 1 && UE_CC_nm(2) == 1

                        kd = 1:n_m_len;
                        for i = activated_UEs_list
                            RBs(e,c,(i-1)*MRB*M+(CC-1)*MRB+ (n_m_idx-1)*(MRB/n_m) +kd)=1;
                        end
                   
                   elseif UE_CC_nm(1) == 1 && UE_CC_nm(2) == 0

                        kd = 1:2*n_m_len;
                        
                        RBs(e,c,(1-1)*MRB*M+(CC-1)*MRB+ (n_m_idx-1)*(MRB/n_m) +kd)=1;
  
                   elseif UE_CC_nm(1) == 0 && UE_CC_nm(2) == 1

                        kd = 1:2*n_m_len;
                        
                        RBs(e,c,(2-1)*MRB*M+(CC-1)*MRB+ (n_m_idx-1)*(MRB/n_m) +kd)=1;

                   else


                   end
                
               end  % n_m = 5 condition ends

               if n_m == 2
               
                   if UE_CC_nm(1) == 1 && UE_CC_nm(2) == 1

                        kd = 1:n_m_len;
                        RBs(e,c,(1-1)*MRB*M+(CC-1)*MRB+ (n_m_idx-1)*(MRB/n_m) +kd)=1;
                        kd = 1:n_m_len-1;
                        RBs(e,c,(2-1)*MRB*M+(CC-1)*MRB+ (n_m_idx-1)*(MRB/n_m) +kd)=1;
                   
                   elseif UE_CC_nm(1) == 1 && UE_CC_nm(2) == 0

                        kd = 1:MRB/n_m;
                        RBs(e,c,(1-1)*MRB*M+(CC-1)*MRB+ (n_m_idx-1)*(MRB/n_m) +kd)=1;
  
                   elseif UE_CC_nm(1) == 0 && UE_CC_nm(2) == 1

                        kd = 1:MRB/n_m;
                        RBs(e,c,(2-1)*MRB*M+(CC-1)*MRB+ (n_m_idx-1)*(MRB/n_m) +kd)=1;

                   else


                   end
                
               end % n_m = 2 condition ends

                   
            end % idx for loop

        else % else of SA_finer flag

            x=0;
            for i=CC-1:(M-1)+(n_m-1):((M-1)+(n_m-1))*K
                x=x+CCs(e,c,i);
            end
            for i=1:K
                if CCs(e,c,(i-1)*((M-1)+(n_m-1))+(CC-1))==1
                y=floor(MRB/x);
                v=0;
                for k=1:MRB
                    if RBs(e,c,(i-1)*MRB*M+(CC-1)*MRB+k)==0
                        v=k;
                        break
                    end
                end
                kd=1:y;
                RBs(e,c,(i-1)*MRB*M+(CC-1)*MRB+kd+v-1)=1;
                end      
            end   
  
        end % if else end of SA_finer flag
    end % SCC loop
    end % cyc for loop
end % eps for loop

for i=1:K
    rate(:,:,i)=(sum(RBs(:,:,(i-1).*M*MRB+1:i.*M*MRB),3).*BW_SCC.*log(1+(G*power(:,:,i)./((((4*pi*UE_location(i))/0.0857143).^2).*sum(RBs(:,:,(i-1).*M*MRB+1:i.*M*MRB),3)*n_var))))./log(2);
end

rate(isnan(rate)) = 0;
avgrate=sum(rate,2)./cycles;

for i=1:1:K
    figure();
    plot(1:eps,avgrate(:,1,i)./1e6,'ro-','LineWidth',2);
    set(gca,'FontSize',14,'TickLabelInterpreter','latex');
    title(['Sum throughput of UE',num2str(i)]);
    xlabel('Episode index','FontSize',14,'interpreter','latex');
    ylabel(['Throughput of UE',num2str(i),' (Mbps)'],'FontSize',14,'interpreter','latex');
    legend("SI, SA, Res10") % modify legend as per the scenario selected
    grid on;
end

if K>1
    figure();
    plot(1:eps,sum(avgrate,3)./1e6,'ro-','LineWidth',2);
    set(gca,'FontSize',14,'TickLabelInterpreter','latex');
    xlabel('Episode index','FontSize',14,'interpreter','latex');
    ylabel('Sum throughput (Mbps)','FontSize',14,'interpreter','latex'); 
    %legend("HA")
    legend("SI, SA, Res10") % modify legend as per the scenario selected
    %legend("No SI")
    grid on;

end

