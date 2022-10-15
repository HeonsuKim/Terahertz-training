c=299792458;
n_1=1;
n_2=group_refractive_index;

parpool(32)

for repeat=51:80

result_pulse=zeros(1995,1000,200);
result_del1=zeros(1995,1000,200);
result_del2=zeros(1995,1000,200);
result_suf=zeros(1995,1000,200);
result_bot=zeros(1995,1000,200);


parfor del_size=1:110

result_pulse_temp=zeros(1995,1000);
result_del1_temp=zeros(1995,1000);
result_del2_temp=zeros(1995,1000);
result_suf_temp=zeros(1995,1000);
result_bot_temp=zeros(1995,1000);    
    
count=0;
del_distance=c/n_2*(time(2,1)-time(1,1))*10^(-12)/2;

while del_distance <= 0.001-0.000001*del_size

count=count+1;
fprintf('del_size=%d, n=%d\n',del_size,count);


resolution=10;
Amp_limit=0;
del_distance=c/n_2*(time(count+1,1)-time(1,1))*10^(-12)/2;
a=zeros(1995,3);

velocity=c/n_2;

theta=0;    %ют╥б
rad_1=asin(sin(deg2rad(theta))/n_2*n_1);
rad_2=asin(sin(rad_1)/n_1*n_2);
rad_3=asin(sin(rad_2)/n_2*n_1);
%rad_4=asin(sin(rad_1)/n_1*n_2);
%rad_5=asin(sin(rad_2)/n_2*n_1);

degree_1=rad2deg(rad_1);
degree_2=rad2deg(rad_2);
degree_3=rad2deg(rad_3);
%degree_4=rad2deg(rad_4);
%degree_5=rad2deg(rad_5);

depth_1=del_distance;
depth_2=0.000001*del_size;
depth_3=0.001-depth_1-depth_2;

% depth_1=0.000475;
% depth_2=0.000045;
% depth_3=0.000490;
%depth_4=0.000025;
%depth_5=0.000475;


length_1=depth_1/cos(rad_1);
length_2=depth_2/cos(rad_2);
length_3=depth_3/cos(rad_3);
%length_4=depth_4/cos(rad_4);
%length_5=depth_5/cos(rad_5);



delta_t1=depth_1*2/velocity/cos(rad_1);
delta_t2=depth_2*2/c/cos(rad_2)+delta_t1;
delta_t3=depth_3*2/velocity/cos(rad_3)+delta_t2;
%delta_t4=depth_4*2/c/cos(rad_4)+delta_t3;
%delta_t5=depth_5*2/velocity/cos(rad_5)+delta_t4;

delta=[delta_t1;delta_t2;delta_t3];
%delta=[delta_t1;delta_t2;delta_t3;delta_t4;delta_t5];

a(:,1)=pulse_restore_td(length_1*2,refractive_index,extinction_coeff,THz_waveform_denoised(:,repeat),time(:,1));
a(:,2)=a(:,1);
a(:,3)=pulse_restore_td(length_3*2,refractive_index,extinction_coeff,a(:,2),time(:,1));

spline_temp=zeros(1995,2);
spline_temp(:,1)=time(:,1);
spline_temp(:,2)=THz_waveform_denoised(:,repeat);
spline_ref=cubic_spline(spline_temp,resolution);

generate_pulse=zeros(19941,4);
for i=1:3
    if mod(i,2)==0
    aa=reflection_generator(THz_waveform_denoised(:,repeat),a(:,i),time(:,1),resolution,delta(i,1));
    generate_pulse(:,i+1)=aa(:,2);
    generate_pulse(:,1)=aa(:,1);
    else
    aa=reflection_generator(THz_waveform_denoised(:,repeat),a(:,i),time(:,1),resolution,delta(i,1));
    generate_pulse(:,i+1)=-aa(:,2);
    generate_pulse(:,1)=aa(:,1);
    end
    
end

simulation_result=zeros(19941,2);
simulation_result(:,1)=generate_pulse(:,1);
simulation_result(:,2)=spline_ref(:,2)+generate_pulse(:,2)+generate_pulse(:,3)+generate_pulse(:,4);

reduced_wave=zeros(1995,1);
reduced_wave2=zeros(1995,1);
reduced_wave3=zeros(1995,1);
reduced_wave4=zeros(1995,1);
reduced_wave5=zeros(1995,1);
for i=1:int16(19941/10)+1
    reduced_wave(i,1)=simulation_result(10*(i-1)+1,2);
    reduced_wave2(i,1)=generate_pulse(10*(i-1)+1,2);
    reduced_wave3(i,1)=generate_pulse(10*(i-1)+1,3);
    reduced_wave4(i,1)=spline_ref(10*(i-1)+1,2);
    reduced_wave5(i,1)=generate_pulse(10*(i-1)+1,4);
end


result_pulse_temp(:,count)=reduced_wave(:,1);
result_del1_temp(:,count)=reduced_wave2(:,1);
result_del2_temp(:,count)=reduced_wave3(:,1);
result_suf_temp(:,count)=reduced_wave4(:,1);
result_bot_temp(:,count)=reduced_wave5(:,1);

end

result_pulse(:,:,del_size)=result_pulse_temp;
result_del1(:,:,del_size)=result_del1_temp;
result_del2(:,:,del_size)=result_del2_temp;
result_suf(:,:,del_size)=result_suf_temp;
result_bot(:,:,del_size)=result_bot_temp;

end

file_name_save=sprintf('D:\\heonsu\\data\\THz_first_%d.mat',repeat);
save(file_name_save,'result_bot','result_del1','result_del2','result_pulse','result_suf','-v7.3')

end