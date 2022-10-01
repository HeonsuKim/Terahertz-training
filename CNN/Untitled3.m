signal_sur=zeros(210,200*8);
signal_del=zeros(210,200*8);
signal_bot=zeros(210,200*8);
signal_noise=zeros(210,200*8);
ccount=0;

for step=29:36
    

mat_step=sprintf("C:\\Users\\BARO\\Downloads\\CNN\\THz_first_%d.mat",step);
clear result_bot result_del1 result_del2 result_pulse result_suf

load(mat_step, 'result_pulse')
fprintf('%d\n',step);

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

a=zeros(1995,count);
a=result_pulse(:,1:count,del);
[a_peak,a_index]=findpeaks(a(:,count),'SortStr','descend','NPeaks',1,'MinPeakDistance', 20);
[a_peak2,a_index2]=findpeaks(-a(1400:1600,1),'SortStr','descend','NPeaks',1,'MinPeakDistance', 20);

for j=1:count
ccount=ccount+1;
signal_sur(:,ccount)=a(a_index-105:a_index+104,j);
signal_del(:,ccount)=a(a_index-105+j:a_index+104+j,j);
signal_bot(:,ccount)=a(a_index2+1399-105:a_index2+1399+104,j);
signal_noise(:,ccount)=a(500-110:599,j);
end


end
end