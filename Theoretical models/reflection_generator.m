function [spline_a1]=reflection_generator(THz_Amp,a ,time, resolution, delta_t1)

%THz_amp : reference signal
%a       : sample signal
%time    : time scale

spline_temp(:,1)=time(:,1);
spline_temp(:,2)=THz_Amp(:,1);
spline_ref=cubic_spline(spline_temp,resolution);
clear spline_temp

spline_temp(:,1)=time(:,1);
spline_temp(:,2)=a(:,1);
spline_a1=cubic_spline(spline_temp,resolution);
clear spline_temp

spline_ref_amp=max(spline_ref(:,2));
spline_ref_index=find(spline_ref(:,2)==spline_ref_amp);

spline_a1_amp=max(spline_a1(:,2));
spline_a1_index=find(spline_a1(:,2)==spline_a1_amp);


delay_present=(spline_a1(spline_a1_index,1)-spline_ref(spline_ref_index,1))*10^(-12);


    while delta_t1>delay_present
        delay_present=delay_present+(spline_a1(2,1)-spline_a1(1,1))*10^(-12);
        spline_temp(:,1)=spline_a1(:,2);
        spline_temp=[spline_temp(1,1); spline_temp];
        spline_temp(end,:)=[];
        spline_a1(:,2)=spline_temp(:,1);
        clear spline_temp
    end


% figure
% hold on
% plot(time(:,1), THz_Amp(:,1))
% plot(time(:,1), a(:,1))
% plot(spline_ref(:,1),spline_ref(:,2))
% plot(spline_a1(:,1),spline_a1(:,2))

end