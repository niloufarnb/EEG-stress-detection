% Stress detection from EEG‑S Kaggle CSV
csvFile = 'eeg_emotions.csv';
holdOut = 0.2;

T = readtable(csvFile);
aIdx = contains(T.Properties.VariableNames,'Alpha','IgnoreCase',true);
bIdx = contains(T.Properties.VariableNames,'Beta','IgnoreCase',true);
tIdx = contains(T.Properties.VariableNames,'Theta','IgnoreCase',true);

alphaP = mean(T{:,aIdx},2);
betaP  = mean(T{:,bIdx},2);
thetaP = mean(T{:,tIdx},2);

X      = [alphaP betaP thetaP];
labels = double(betaP > alphaP);

cv     = cvpartition(labels,'HoldOut',holdOut);
Xtr    = X(training(cv),:);
Xte    = X(test(cv),:);
ytr    = labels(training(cv));
yte    = labels(test(cv));

mu = mean(Xtr);
sg = std(Xtr);
Xtr = (Xtr - mu)./sg;
Xte = (Xte - mu)./sg;

Mdl = fitcensemble(Xtr,ytr,'Method','Bag','NumLearningCycles',100);
yp  = predict(Mdl,Xte);

cm = confusionmat(yte,yp);
fprintf('Accuracy: %.1f%%\n',100*sum(diag(cm))/sum(cm(:)));
confusionchart(cm,{'Relaxed','Stress'});

save('StressModel_Kaggle.mat','Mdl','mu','sg');
