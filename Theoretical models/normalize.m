%normalize the THz signal
for i=1:25
    plot(THz_waveform_denoised(:,i))
end

for i=1:size(THz_waveform_denoised,2)
    max_plus=max(THz_waveform_denoised(:,i));
    max_minus=max(-THz_waveform_denoised(:,i));
    for j=1:1995
        if THz_waveform_denoised(j,i)<0
            THz_waveform_denoised(j,i)=THz_waveform_denoised(j,i)/max_minus;
        else
            THz_waveform_denoised(j,i)=THz_waveform_denoised(j,i)/max_plus;
        end
    end
end