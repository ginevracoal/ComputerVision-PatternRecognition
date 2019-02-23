%rng(0,'twister')

cd('/home/ginevracoal/MEGA/Università/DSSC/semester_3/ComputerVision-PatternRecognition/MatlabCode/00_estimating_homographies/')
close all

%Ia = imread('images/cam2_00002.jpg') ;
%Ib = imread('images/cam2_00003.jpg') ;
Ia = imread('images/notredame1.jpg') ;
Ib = imread('images/notredame2.jpg') ;
%Ia = imread('images/rushmore1.jpg') ;
%Ib = imread('images/rushmore2.jpg') ;
%Ia = imread('images/govpalace1.jpg') ;
%Ib = imread('images/govpalace2.jpg') ;
%Ia=imresize(Ia,.5);
%Ib=imresize(Ib,.5);


% extract features using:
% [frames,descriptors] = vl_sift(image);
% The matrix frames has a column for each frame. 
% A frame is a disk of center frames(1:2), scale frames(3) and orientation frames(4).

% we need to convert from rgb to greyscale
[frames_a,descriptors_a] = vl_sift(im2single(rgb2gray(Ia))) ;
[frames_b,descriptors_b] = vl_sift(im2single(rgb2gray(Ib))) ;

% matching keypoints
% there are 200 matches
[matches, scores] = vl_ubcmatch(descriptors_a,descriptors_b) ;


figure
show_correspondence(Ia, Ib, frames_a(1,matches(1,:))', frames_a(2,matches(1,:))', frames_b(1,matches(2,:))', frames_b(2,matches(2,:))')

% RANSAC 

ransac_th=0.0005; %algebraic threshold

nmatches = size(matches,2);

ntrials=2000;
% allocating a 3x3 matrix and vector for all the trials
FF=zeros(3,3,ntrials); % to store the F matrices
consensus=zeros(1,ntrials); % to store the consensus

for ii=1:ntrials

    % select random matches
    selected=randperm(nmatches,8);

    % coordinates of points in the two images
    Ma=frames_a(1:2,matches(1,selected));
    Mb=frames_b(1:2,matches(2,selected));

    % compute the fit based on the selected keypoints
    F=estimateF(Ma,Mb);
    FF(:,:,ii)=F;
    
    

    % cast votes from all (including the selected ones for the estimation)
    Ma_all=frames_a(1:2,matches(1,:));
    Mb_all=frames_b(1:2,matches(2,:));
    
    % homogeneous coordinates
    ma = [Ma; ones(1,size(Ma,2))];
    mb = [Mb; ones(1,size(Mb,2))];
    
    % line coefficients in the first image
    lambda=F*ma;

    consensus(ii)=0;
    for jj=1:nmatches
       % ideally mb^T * F *ma should be zero
       % thus we put a threshold on the absolute value
       %if abs([Mb_all(:,jj);1]'*F*[Ma_all(:,jj);1])<ransac_th
       if abs(lambda*transpose(mb))/norm(lambda(1:2))<ransac_th 
          consensus(ii)=consensus(ii)+1;    
       end

    end

end

figure
plot (consensus)

[~,imax]=max(consensus);
F=FF(:,:,imax); % the fundamental matrix with largest consensus


Ma=frames_a(1:2,matches(1,:));
Mb=frames_b(1:2,matches(2,:));

for ii=size(matches,2):-1:1
     
    if abs([Mb(:,ii);1]'*F*[Ma(:,ii);1])>=ransac_th 
        matches(:,ii)=[]; % discard as outlier
    end

end

% plot all the inliers
hnd=figure;
figure(hnd)
show_correspondence(Ia, Ib, frames_a(1,matches(1,:))', frames_a(2,matches(1,:))', frames_b(1,matches(2,:))', frames_b(2,matches(2,:))')

return

function epipolar(F,points)

    lambda=F*points;
    
end

function F=estimateF(Ma,Mb)
%Ma and Mb are 2x8

    A=zeros(8,9);
    for ii=1:8
       % using the kronecker product
       A(ii,:)=kron([Ma(:,ii); 1]',[Mb(:,ii); 1]');
    end

    [U,S,V]=svd(A);

    F=reshape(V(:,9),[3 3 ]);
    
    %enforce singularity by zeroing the smallest singular value
    [U,S,V]=svd(F);
    F=U*diag([S(1,1),S(2,2),0])*V';
end


