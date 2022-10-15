load("D:\Dropbox\python_data\data\THz_signal.mat", "del25");
load('D:\Dropbox\python_data\data\THz_signal.mat', 'del110');
load("D:\Dropbox\python_data\data\THz_signal.mat", "del275");
load("D:\Dropbox\python_data\data\THz_signal.mat", "del450");
load("D:\Dropbox\python_data\data\THz_signal.mat", "delmulti");

load("D:\Dropbox\python_data\data\predictions_mat.mat", "predictions_25");
load("D:\Dropbox\python_data\data\predictions_mat.mat", "predictions_110");
load("D:\Dropbox\python_data\data\predictions_mat.mat", "predictions_275");
load("D:\Dropbox\python_data\data\predictions_mat.mat", "predictions_450");
load("D:\Dropbox\python_data\data\predictions_mat.mat", "predictions_multi");

figure()
hold on
subplot(6,1,1)
plot(predictions_25(:,1),predictions_25(:,6))
xlim([60 90])
title("25 um")
subplot(6,1,2)
plot(predictions_25(:,1),predictions_25(:,5))
xlim([60 90])
title("24 um")
subplot(6,1,3)
plot(predictions_25(:,1),predictions_25(:,4))
xlim([60 90])
title("23 um")
subplot(6,1,4)
plot(predictions_25(:,1),predictions_25(:,3))
xlim([60 90])
title("22 um")
subplot(6,1,5)
plot(predictions_25(:,1),predictions_25(:,2))
xlim([60 90])
title("single signal")
subplot(6,1,6)
plot(del25(:,1),del25(:,2))
xlim([60 90])
title("THz signal")

figure()
hold on
subplot(6,1,1)
plot(predictions_110(:,1),predictions_110(:,6))
xlim([60 90])
title("110 um")
subplot(6,1,2)
plot(predictions_110(:,1),predictions_110(:,5))
xlim([60 90])
title("109 um")
subplot(6,1,3)
plot(predictions_110(:,1),predictions_110(:,4))
xlim([60 90])
title("108 um")
subplot(6,1,4)
plot(predictions_110(:,1),predictions_110(:,3))
xlim([60 90])
title("107 um")
subplot(6,1,5)
plot(predictions_110(:,1),predictions_110(:,2))
xlim([60 90])
title("single signal")
subplot(6,1,6)
plot(del110(:,1),del110(:,2))
xlim([60 90])
title("THz signal")

figure()
hold on
subplot(2,1,1)
plot(predictions_275(:,1),predictions_275(:,2))
xlim([60 90])
title("single signal")
subplot(2,1,2)
plot(del275(:,1),del275(:,2))
xlim([60 90])
title("THz signal")

figure()
hold on
subplot(2,1,1)
plot(predictions_450(:,1),predictions_450(:,2))
xlim([60 90])
title("single signal")
subplot(2,1,2)
plot(del450(:,1),del450(:,2))
xlim([60 90])
title("THz signal")