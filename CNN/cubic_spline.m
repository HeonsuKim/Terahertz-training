function [s_eq]=cubic_spline(test,time_scale)

%test(:,1)=시간
%test(:,2~)=데이터
%time_scale, data만 입력되면 가능

[data_size,data_rank]=size(test);
clear s_eq
for i=1:data_size-1
    hh(i,1)=test(i+1,1)-test(i,1);
end
h=mean(hh); clear hh
%time_scale=500;
time_range=zeros(time_scale*(data_size-1),1);
for i=1:data_size-1
    for j=1:time_scale
        time_range((i-1)*time_scale+j,1)=test(i,1)+(test(i+1,1)-test(i,1))/time_scale*j;
    end
end
time_range=[test(1,1); time_range];

a_coeff=zeros(data_size-1,data_rank);
b_coeff=zeros(data_size-1,data_rank);
c_coeff=zeros(data_size-1,data_rank);
d_coeff=zeros(data_size-1,data_rank);

s_eq(:,1)=time_range(:,1);

for ii=2:data_rank
    A_mat=zeros(data_size-2,data_size-2);
    B_mat=zeros(data_size-2,1);
    M_mat=zeros(data_size-2,1);
    for i=1:data_size-2
        
        B_mat(i,1)=(test(i,ii)-2*test(i+1,ii)+test(i+2,ii))*6/(h*h);
        
        for j=1:data_size-2
            if i==1
                A_mat(i,i)=4;
                A_mat(i,i+1)=1;
            elseif i==data_size-2
                A_mat(i,i-1)=1;  
                A_mat(i,i)=4;
            else
                A_mat(i,i-1)=1;
                A_mat(i,i)=4;
                A_mat(i,i+1)=1;
            end
        end
    end
    M_mat=A_mat^(-1)*B_mat;
    M_mat=[0;M_mat;0];
    
    for k=1:data_size-1
        a_coeff(k,ii)=(M_mat(k+1)-M_mat(k))/(6*h);
        b_coeff(k,ii)=M_mat(k)/2;
        c_coeff(k,ii)=(test(k+1,ii)-test(k,ii))/h-(M_mat(k+1)+2*M_mat(k))/6*h;
        d_coeff(k,ii)=test(k,ii);
        for tt=1:time_scale
            s_eq((k-1)*time_scale+tt+1,ii)=a_coeff(k,ii)*(time_range((k-1)*time_scale+tt+1,1)-test(k,1))^3+b_coeff(k,ii)*(time_range((k-1)*time_scale+tt+1,1)-test(k,1))^2+c_coeff(k,ii)*(time_range((k-1)*time_scale+tt+1,1)-test(k,1))+d_coeff(k,ii);
        end
    end
    s_eq(1,ii)=test(1,ii);
    
%     figure(ii)
%     plot(test(:,1),test(:,ii))
%     hold on
%     plot(time_range,s_eq(:,ii))
% 
%     ylim([-200 200])
end



end