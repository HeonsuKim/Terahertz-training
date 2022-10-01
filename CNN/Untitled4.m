for del=11:110
count=0;
c=299792458;
n_1=1;
%n_2=2.1;
n_2=2.256;
del_distance=c/n_2*(time(2,1)-time(1,1))*10^(-12)/2;
while del_distance <= 0.001-0.000001*del
count=count+1;
del_distance=c/n_2*(time(count+1,1)-time(1,1))*10^(-12)/2;
end

aa(del-10)=count;
end

a=zeros(1995,1000);
a=result_pulse(:,:,10);