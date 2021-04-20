%% Vida Movahedi, elderlab.yorku.ca

% find contour mapping distance
% input: coordinates of points on two contours
%   ia, ja : each are a vector of length n, defining n points on contour A
%   ib, jb : each are a vector of length m, defining m points on contour B
%   CMM = optMapMae(double(ia), double(ja), double(ib), double(jb));
% output: the CM distance
% note: you need to run "mex optMapMae.cpp" before running this code

function [CMM, t] = getCMM( ia, ja, ib, jb)
%    ia = ia(:);
%    ja = ja(:);
%    ib = ib(:);
%    jb = jb(:);
    tic;
    CMM = optMapMae(ia,ja,ib,jb);
    t = toc;
end