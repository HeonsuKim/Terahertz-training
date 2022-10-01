function [restore_fd]=pulse_restore_fd(thickness,refractive_index,extinction_coeff,f,mag,phase_rad)

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