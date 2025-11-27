clc;
clear all;
close all;

K=2;
M=2;
MRB=50;
eps=100;
cyc=100;

power_max=10^(-0.3);
G_dB = 30;
G = 10^(0.1*G_dB);
n_var=1e-10;
BW_SCC=18e4;
L_c_dB = 35;

% eq = [25,25] neq = [25,35]
UE_location = [25,35]; 
MSD_CC_idx = [2,2];
MSD_causing_CC = 2;

%%%%%%%%%%%%%%%%%%% SET ATMOST 1 flag depending on the scenario %%%%%%%%%%%

SA_finer = 0; % Set this flag for "SI,SA,Res25","SI,SA,Res10"
SA = 1;       % Set this flag for "SI,SA,Res50"
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


power_i = data.actions(1:eps,cyc,1:K);
CCs = data.actions(1:eps,cyc,K+1:K*((M-1)+n_m));
NRB=zeros(eps,K,M);
NCC=zeros(eps,K);
RB=zeros(eps,M*MRB*K);

p_ijk = zeros(eps,M*MRB*K);

activated_UEs_list = 1:K;

% RB algorithm
for e = 1:eps
    %%
    % NOTE: 
    % To avoid online adaptation, update e with a value above 'eps' value
    % if e == 150 % indicate episode at which UE exits network
    %     activated_UEs_list = activated_UEs_list(find(activated_UEs_list~=2)); %removes UE2
    % elseif e == 200 % indicate episode at which UE re-enters network
    %     activated_UEs_list = [activated_UEs_list 2]; %adds UE2
    % end
    
    %%
    K_act = length(activated_UEs_list);

    j = 0;
    for i = activated_UEs_list
         k = 1:floor(MRB/K_act) ;
         RB(e,(i-1)*MRB*M+j*MRB+k)=1;
    end
    %active_subblocks_cntr = 0;
    for CC = 2:M 
        if MSD_causing_CC == CC && SA_finer == 1
           for n_m_idx = 1:n_m
               UE_CC_nm = zeros(1,K);

               for i = activated_UEs_list
                    UE_CC_nm(i) = CCs(e,(i-1)*((M-1)+(n_m-1))+(n_m_idx-1)+(CC-1));
               end
                  
               n_m_len = ceil(MRB/(n_m*K_act));

               if n_m == 5
               
                   if UE_CC_nm(1) == 1 && UE_CC_nm(2) == 1
                        kd = 1:n_m_len;
                        for i = activated_UEs_list
                            RB(e,(i-1)*MRB*M+(CC-1)*MRB+ (n_m_idx-1)*(MRB/n_m) +kd)=1;
                        end
                   
                   elseif UE_CC_nm(1) == 1 && UE_CC_nm(2) == 0
                        kd = 1:2*n_m_len;
                        RB(e,(1-1)*MRB*M+(CC-1)*MRB+ (n_m_idx-1)*(MRB/n_m) +kd)=1;
  
                   elseif UE_CC_nm(1) == 0 && UE_CC_nm(2) == 1
                        kd = 1:2*n_m_len;
                        RB(e,(2-1)*MRB*M+(CC-1)*MRB+ (n_m_idx-1)*(MRB/n_m) +kd)=1;

                   else

                   end
                
               end  % n_m = 5 condition ends

               if n_m == 2          
                   if UE_CC_nm(1) == 1 && UE_CC_nm(2) == 1
                        kd = 1:n_m_len;
                        RB(e,(1-1)*MRB*M+(CC-1)*MRB+ (n_m_idx-1)*(MRB/n_m) +kd)=1;
                        kd = 1:n_m_len-1;
                        RB(e,(2-1)*MRB*M+(CC-1)*MRB+ (n_m_idx-1)*(MRB/n_m) +kd)=1;
                   
                   elseif UE_CC_nm(1) == 1 && UE_CC_nm(2) == 0
                        kd = 1:MRB/n_m;
                        RB(e,(1-1)*MRB*M+(CC-1)*MRB+ (n_m_idx-1)*(MRB/n_m) +kd)=1;
  
                   elseif UE_CC_nm(1) == 0 && UE_CC_nm(2) == 1
                        kd = 1:MRB/n_m;
                        RB(e,(2-1)*MRB*M+(CC-1)*MRB+ (n_m_idx-1)*(MRB/n_m) +kd)=1;

                   else

                   end
                
               end % n_m = 2 condition ends

           end % idx loop ends

        else  % else condition of SA_finer flag

            x=0;
            for i=CC-1:(M-1)+(n_m-1):((M-1)+(n_m-1))*K
                x=x+CCs(e,i);
            end
            for i=1:K
                if CCs(e,(i-1)*((M-1)+(n_m-1))+(CC-1))==1
                y=floor(MRB/x);
                v=0;
                for k=1:MRB
                    if RB(e,(i-1)*MRB*M+(CC-1)*MRB+k)==0
                        v=k;
                        break
                    end
                end
                kd=1:y;
                RB(e,(i-1)*MRB*M+(CC-1)*MRB+kd+v-1)=1;
                end      
            end             


        end % SA_finer flag
    end % SCC loop

    for i = activated_UEs_list
    sum1=sum(RB(e,(i-1)*MRB*M+1:(i)*MRB*M));
        for k = 1:(M*MRB)
            if RB(e,(i-1)*MRB*M+k)==1
                p_ijk(e,(i-1)*MRB*M+k)= power_i(e,i)/sum1;
            end
        end
    end    
    
end % eps for loop

for i=1:K
    for j=1:M
    NRB(:,i,j)=sum(RB(:,MRB*M*(i-1)+MRB*(j-1)+1:MRB*M*(i-1)+MRB*j),2);
    Total_NRB = sum(NRB,3);
    end
end

CC1 = 1; % since PCC is alloted RBs by default
CC2 = zeros(eps,K);

for  i = 1:K
    if SA_finer == 1
        if n_m == 5
            CC2(:,i) = arrayfun(@CC_subblock,CCs(:,1,1 + n_m*(i-1))|CCs(:,1,2 + n_m*(i-1))|CCs(:,1,3 + n_m*(i-1))|CCs(:,1,4 + n_m*(i-1))|CCs(:,1,5 + n_m*(i-1))); 
        else
            CC2(:,i) = arrayfun(@CC_subblock,CCs(:,1,1 + n_m*(i-1))|CCs(:,1,2 + n_m*(i-1)));
        end
        NCC(:,i) = CC1 + CC2(:,i);
    else
        NCC(:,i)=sum(CCs(:,(M-1)*(i-1)+1:(M-1)*i),2)+1;
    end
end

c_2 = 0.1417;
A_sq_py_in = zeros(eps,K*M);
P_SI_py = zeros(eps,K*M);


L_c = 10^(0.1*L_c_dB);


activated_UEs_list = 1:K;
for e = 1:eps
    if e == 150 % indicate episode at which UE exits network
        activated_UEs_list = activated_UEs_list(find(activated_UEs_list~=2)); %removes UE2
    elseif e == 200 % indicate episode at which UE re-enters network
        activated_UEs_list = [activated_UEs_list 2]; %adds UE2
    end
    K_act = length(activated_UEs_list);
    for i = activated_UEs_list
        j = MSD_CC_idx(i); %cc in MSD
        A_sq_py_in(e,(i-1)*M+j) = (2*sum(p_ijk(e,(i-1)*MRB*M+(j-1)*MRB+1:(i-1)*MRB*M+(j)*MRB)));
        P_SI_py(e,(i-1)*M+j) = (c_2^2)*(A_sq_py_in(e,(i-1)*M+j).^2)/8;        
    end
end


%% UE i

for i=1:1:K

%% NCC

    figure();
    plot(1:eps,NCC(:,i),'square-','LineWidth',1,'MarkerEdgeColor','[0.07,0.62,1.00]','MarkerFaceColor','[0.07,0.62,1.00]');
    set(gca,'FontSize',14,'TickLabelInterpreter','latex');
    %title(['NCCs of UE ',num2str(i)]);
    xlabel('Episode index','FontSize',14,'interpreter','latex');
    ylabel('No. of allocated CCs','FontSize',14,'interpreter','latex');
    ylim([0,2])
    yticks(0:1:2)
    xlim([0,eps])
    grid on;
    outfile = ['UE',num2str(i),'_NCC.png'];
    saveas(gcf,outfile);
%% NRB

    for j = 1:1:M
        figure();
        plot(1:eps,NRB(:,i,j),'mdiamond-','LineWidth',1,'MarkerFaceColor','m');
        set(gca,'FontSize',14,'TickLabelInterpreter','latex');
        %title(['RBs in CC',num2str(j)]);
        xlabel('Episode index','FontSize',14,'interpreter','latex');
        ylabel(['No. of allocated RBs in CC',num2str(j)],'FontSize',14,'interpreter','latex');
        ylim([0,100])
        xlim([0,eps])
        grid on;
        outfile = ['UE',num2str(i),'_RB_CC',num2str(j),'.png'];
        saveas(gcf,outfile);
    end


    figure();
    plot(1:eps,Total_NRB(:,i),'bsquare-','LineWidth',1,'MarkerFaceColor','b');
    set(gca,'FontSize',14,'TickLabelInterpreter','latex');
    %title(['Total RBs of UE ',num2str(i)]);
    xlabel('Episode index','FontSize',14,'interpreter','latex');
    ylabel('Total no. of allocated RBs','FontSize',14,'interpreter','latex');
    ylim([0,100])
    xlim([0,eps])
    grid on;
    outfile = ['UE',num2str(i),'_NRB.png'];
    saveas(gcf,outfile);


%% Power Plots of UE i

    for j = 1:1:M

        %(i-1)*MRB*M+(j-1)*MRB+1:(i-1)*MRB*M+(j)*MRB
        power_ijk = G*sum(p_ijk(:,(i-1)*MRB*M+(j-1)*MRB+1:(i-1)*MRB*M+j*MRB),2);
        P_SI_ijk = P_SI_py(:,(i-1)*M+j)/L_c;

        figure();
        plot(1:eps,power_ijk,'ro-','LineWidth',1);
        set(gca,'FontSize',14,'TickLabelInterpreter','latex');
        %title(['Power on CC',num2str(j),' of UE',num2str(i)]);
        xlabel('Episode index','FontSize',14,'interpreter','latex');
        ylabel(['Tx. power on CC',num2str(j),' (W)'],'FontSize',14,'interpreter','latex');
        xlim([0,eps])
        ylim([0,0.5])
        grid on;
        outfile = ['UE',num2str(i),'_Tx_P_CC',num2str(j),'.png'];
        saveas(gcf,outfile);
    end
        figure();
        plot(1:eps,P_SI_ijk,'ro-','LineWidth',1);
        set(gca,'FontSize',14,'TickLabelInterpreter','latex');
        %title(['$P_{SI}$ due to CC',num2str(j),' on $f_{DL}$ in UE',num2str(i)]);
        xlabel('Episode index','FontSize',14,'interpreter','latex');
        ylabel(['$P_{SI}$ due to CC',num2str(j),' (W)'],'FontSize',14,'interpreter','latex');
        xlim([0,eps])
        grid on;
        outfile = ['UE',num2str(i),'_P_SI_CC',num2str(j),'.png'];
        saveas(gcf,outfile);
    %end

    figure();
    plot(1:eps,G*power_i(:,:,i),'ro-','LineWidth',1);
    set(gca,'FontSize',14,'TickLabelInterpreter','latex');
    %title(['Total Power of UE ',num2str(i)]);
    xlabel('Episode index','FontSize',14,'interpreter','latex');
    ylabel('Total Tx. Power (W)','FontSize',14,'interpreter','latex');
    xlim([0,eps])
    grid on;
    outfile = ['UE',num2str(i),'_Total_P.png'];
    saveas(gcf,outfile);
end


function result = CC_subblock(x)

if x == 1
    result = 1;
else
    result = 0;
end

end
