function [F] =feature_extraction(I,K,BF,NP,NB,W,Rand)
F=zeros(NP,NB);
for i=1:NP
    %% Moments
    M = BF{i,2}*double(I)*BF{i,1}';
    %% Randomization
    M = M.*Rand;
    %% Ring Integral
    [IM] = ring_integral(K,NB,abs(M));
    %% Weighting
    WIM=W.*IM;
    F(i,:)=WIM;
end
F=reshape(F',1,[]);
% plot(F);
end