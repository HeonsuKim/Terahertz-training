function [restore_td]=pulse_restore_td(thickness,refractive_index,extinction_coeff,THz_Amp, time)
%%
tt=1;
t_window = 3;

[pka,pkai]=findpeaks(THz_Amp(:,tt),'MinpeakDistance',30);
peaks=sort(pka,'descend'); % 내림차순으로 정렬
peak_id(1,1)=find(pka==peaks(1)); % peak index
t_peak=time(pkai(peak_id(1,1)));
t_peak_l(1) = t_peak-t_window;
t_peak_h(1) = t_peak+t_window;

x2(1996:4096,tt)=0;
for i=1:1995
    if time(i,tt)>=t_peak_l && time(i,tt)<=t_peak_h
        x2(i,tt)=THz_Amp(i,tt);
    else
        x2(i,tt)=0;
    end
end

N=length(x2(:,tt))/2; 
n=0:N;
T=0.055210526*1e-12; % 0.055 ps
t(:,tt)=0:T:T*N;
fs = 1/T; % 18THz
f(:,tt)=fs*n/(2*N);

X2(:,tt)=fft(x2(:,tt));
cutoff = ceil(N);
X(:,tt)=X2(1:cutoff+1,tt);
x(:,tt)=x2(1:cutoff+1,tt);
mag(:,tt) =1/sqrt(N)*abs(X(:,tt));
phase_rad(:,tt)=angle(X(:,tt));

% figure(1)
% subplot(2,1,1)
% plot(t(:,tt)*1e12,x(:,tt))
% hold on
% title('Time Domain')
% xlabel('Time(ps)')
% ylabel('THz Amplitude')
% xlim([0, 110])
% ylim([-2000, 2000])
% 
% subplot(2,1,2)
% hold on
% plot(f(:,tt),20*log10(mag(:,tt)));
% title('Normalized(dB)')
% xlabel('Frequency(THz)')
% ylabel('THz Amplitude')
% xlim([0,10*1e12]) 
% ylim([0,60])

clear buffer datatest fid fs i ii iii n peak_id peaks pka pkai read_data read_path T t_peak t_peak_h t_peak_l t_window tt ans
%%

%thickness=0.000;
N=2048;
cutoff = ceil(N);
phase_new(:,1)=unwrap(phase_rad(:,1))-(refractive_index(:,1)-1)*2*pi.*f(:,1)*thickness/299792458;
phase_new(1,1)=pi;

mag_new(:,1)=4*refractive_index(:,1)./(exp(extinction_coeff(:,1).*2*pi.*f(:,1)*thickness/299792458).*(refractive_index(:,1)+1).^2).*mag(:,1);

a=size(mag_new);
restore=zeros((a(1)-1)*2,1);
for i=1:a(1)
    restore(i,1)=sqrt(N)*mag_new(i,1).*exp(1i*phase_new(i,1));
end

restore(1,1)=0;
restore(2049,1)=real(restore(2049,1));
for i=1:a(1)-1
    restore((a(1)-1)*2+1-i,1)=conj(restore(i+1,1));
end

restore_td(:,1)=ifft(restore);
restore_td=restore_td(1:1995,1);
restore_fd(:,1)=1/sqrt(N)*restore(1:cutoff+1,1);
% 
% figure
% plot(restore_td)
% hold on
% plot(x(:,1))
% plot(x(:,2))
% plot(x(:,3))
% plot(x(:,4))
% plot(x(:,5))
% 
% figure
% plot(abs(restore_fd))
% hold on
% plot(mag(:,1))
% plot(mag(:,2))
% plot(mag(:,3))
% plot(mag(:,4))
% plot(mag(:,5))

end