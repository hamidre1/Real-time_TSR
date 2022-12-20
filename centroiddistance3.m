clc;clear
close all
CDtime=0;
SVMtime=0;
CDfalse=0;
SVMfalse=0;
CDcorrect=0;
SVMcorrect=0;
signsizeCD=50;
signsizeSVM=50;
showpic=false;
testdir=fullfile(pwd , '\mix');
testSet = imageDatastore(testdir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

syntheticDir   = fullfile(pwd , '\Training_TrafficSign_2'); %'k:\dr\proposal\matlab_simulations\Training_TrafficSign';
trainingSet = imageDatastore(syntheticDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');  % ,'IncludeSubfolders', true, 'LabelSource', 'foldernames');
%testSet     = imageDatastore('H:\dr\proposal\matlab_simulations\1','IncludeSubfolders', true, 'LabelSource', 'foldernames');
%traininglabels(:,1)=70;
TrainingSet=countEachLabel(trainingSet)
%countEachLabel(testSet)
cellSize = [4 4];
%img=red_region2_candid_D;
%% Extract HOG features and HOG visualization
%[hog_2x2, vis2x2] = extractHOGFeatures(img,'CellSize',[2 2]);
%[hog_4x4, vis4x4] = extractHOGFeatures(img,'CellSize',[4 4]);
%[hog_8x8, vis8x8] = extractHOGFeatures(img,'CellSize',[8 8]);
%% Show the original image
%figure; 
%subplot(2,3,1:3); imshow(img);

%% Visualize the HOG features
%subplot(2,3,4);  
%plot(vis2x2); 
%title({'CellSize = [2 2]'; ['Length = ' num2str(length(hog_2x2))]});

%subplot(2,3,5);
%plot(vis4x4); 
%title({'CellSize = [4 4]'; ['Length = ' num2str(length(hog_4x4))]});

%subplot(2,3,6);
%plot(vis8x8); 
%title({'CellSize = [8 8]'; ['Length = ' num2str(length(hog_8x8))]});
%%A good compromise is a 4-by-4 cell size. This size setting encodes enough spatial information to visually identify a digit shape while limiting the number of dimensions in the HOG feature vector, which helps speed up training. In practice, the HOG parameters should be varied with repeated classifier training and testing to identify the optimal parameter settings.


%hogFeatureSize = length(hog_4x4);

% Train a Digit Classifier
% Start by extracting HOG features from the training set. These features will be used to train the classifier.

% Loop over the trainingSet and extract HOG features from each image. A
% similar procedure will be used to extract features from the testSet.
numImages = numel(trainingSet.Files);
%trainingFeatures = zeros(numImages, hogFeatureSize, 'single');

for i = 1:numImages
    %tmpimg=string(trainingSet.Files(i,1));
    %img = readimage([tmpimg]);
    img = imread(string(trainingSet.Files(i,1)));
    img=imresize(img,[signsizeSVM,signsizeSVM],'bilinear');
    %img=red_region2_candid_D;
    img = rgb2gray(img);
    % Apply pre-processing steps
    img = imbinarize(img);
    hog_4x4= extractHOGFeatures(img, 'CellSize', cellSize);
    trainingFeatures(i, 1:size(hog_4x4,2)) =  hog_4x4;
    %trainingFeatures(i,:) =  hog_4x4(2);
end

% Get labels for each image.
trainingLabels = trainingSet.Labels;
% train a classifier using the extracted features.
% fitcecoc uses SVM learners and a 'One-vs-One' encoding scheme.
classifier = fitcecoc(trainingFeatures, trainingLabels);

% Evaluate the Digit Classifier
% Evaluate the digit classifier using images from the test set, and generate a confusion matrix to quantify the classifier accuracy.
% As in the training step, first extract HOG features from the test images. These features will be used to make predictions using the trained classifier.
% Extract HOG features from the test set. The procedure is similar to what
% was shown earlier and is encapsulated as a helper function for brevity.


testno=size(testSet.Files,1);
for im=1:1:testno
strr=testSet.Files{im};

rgb_image=imread(strr);
%rgb_image=imresize(rgb_imaget,[480,640],'bilinear');
%rgb_image_o=rgb_image;
%figure;subplot(1,3,1);imshow(rgb_image(:,:,:))

%image_width=numel(rgb_image(:,1,1));
%image_height=numel(rgb_image(1,:,1));
[h,w,~] = size(rgb_image);

for y=1:1:w
    for x=1:1:h
        r=double(rgb_image(x,y,1));g=double(rgb_image(x,y,2));b=double(rgb_image(x,y,3));
        if (0.45*(b+(r-1))-g)>=0 && (2*b-(r-1)-g)<=0  %0.42
            red_region_f(x,y)=1;
        else
            red_region_f(x,y)=0;
        end    
    end
end
%red_region_f=medfilt2(red_region2,[2 2]);
%red_region_f=red_region2;
[L,n] = bwlabel(red_region_f);
%figure
% Size Filter & Aspect Ratio Filter:
for l=1:1:n
    [r, c] = find(L==l);
    %rc = [r c];
    %size(rc);
    %Sr=size(r,1)*size(c,1);
    %Src2=size(rc,2)
    La=max(r)-min(r);
    Ha=max(c)-min(c);
    Sr=La*Ha;
    AsF=La/Ha;
    if ((Sr )<2500)%|((Sr )>50500)
        red_region_f(r,c)=0;
        %'((Sr )<10)|((Sr )>2500)'
    else
        %red_region_f(r,c);
        %if n>1
            if (AsF>2)|(AsF<0.5)
                red_region_f(r,c)=0;
                %l
                %AsF
            end
        %end
    end
    %subplot(n,1,l);
    %imshow(red_region_f(r,c));
end   
%imshow(red_region_f);


[Lf,nf] = bwlabel(red_region_f);

%figure;subplot(1,3,2);imshow(red_region_f)
%subplot(1,3,3);imshow(red_region_f)
%figure;imshow(Lf)

%figure
 roi_candidate={};

for l=1:1:nf
    [rf, cf] = find(Lf==l);
    rcf = [rf cf];
    %roi_candidate(l)=0;
    %roi_candidate{l}=red_region2(min(rf):1:max(rf),min(cf):1:max(cf));
    %bboxes(l,:)=[min(cf) min(rf) max(cf)-min(cf) max(rf)-min(rf)];
    %subplot(nf,8*nf,(nf*(l-1)*8)+1);
    rgb_image_candidate=rgb_image(min(rf):1:max(rf),min(cf):1:max(cf),:,:,:);
    %imshow(rgb_image_candidate);
    
    %subplot(nf,8*nf,(nf*(l-1)*8)+2);
    %imshow(red_region_f(min(rf):1:max(rf),min(cf):1:max(cf)));
    %title('A');
    red_region2_candid=red_region_f(min(rf):1:max(rf),min(cf):1:max(cf));
    %rgb_image(min(rf):1:max(rf),min(cf):1:max(cf),2)=150;
    rgb_image = insertShape(rgb_image, 'Rectangle',[min(cf),min(rf),max(cf)-min(cf),max(rf)-min(rf)],'LineWidth',3);
    
    red_region2_candid_B=imfill(red_region2_candid);
    %subplot(nf,8*nf,(nf*(l-1)*8)+4);
    %imshow(red_region2_candid_B)
    %title('B');
    
    red_region2_candid_C= xor(red_region2_candid_B, red_region2_candid);
    %subplot(nf,8*nf,(nf*(l-1)*8)+5);
    %imshow(red_region2_candid_C) 
    %title('C');
    
    red_region2_candid_rz=imresize(red_region2_candid_B,[signsizeCD,signsizeCD]);
    edge_red_region2_candid=edge(red_region2_candid_rz,'sobel');
    %subplot(nf,8*nf,(nf*(l-1)*8)+3);imshow(edge_red_region2_candid);

    %roi_candidateCD{l}=edge_red_region2_candid;
    roi_p_candidateCD{im,l}= edge(imresize(red_region2_candid_C,[signsizeCD,signsizeCD]));
    %title(s);

    red_region2_candid_D=uint8(double(rgb_image_candidate).*double(red_region2_candid_B));
    %subplot(nf,8*nf,(nf*(l-1)*8)+6);imshow(red_region2_candid_D)  
    %title('D');
    
    roi_candidate{l}=imresize(red_region2_candid_D,[signsizeSVM,signsizeSVM]);
      
end 
   
display(sprintf('path=%s nfs=%d',strr,nf));
for i=1:nf
     %disp(strcat('Sign(s) no. : ', num2str(i)));     
    tic
 
    s(im,i)=circulardistance(roi_p_candidateCD{im,i});
    CD_execution_time=toc;
    CDtime=CDtime+CD_execution_time;
    signCD='Other';
    shape='None';
    if (s(im,i)>1000)&&(s(im,i)<1570)     % 50:1500-1900
        shape='Circle';
        signCD='No Entry';
    end    
    if (s(im,i)>1570)&&(s(im,i)<1700) %24:420-450 50:3900-4100 100:10000-30000
        shape='Octagon';        
        signCD='Stop';
    end    
    if (s(im,i)>1700)&&(s(im,i)<2100)   %24:380-420 50:3800-4300
        shape='Triangle';
        signCD='Yield';
    end
    if (s(im,i)>2100)&&(s(im,i)<2500)
        signCD='Speed Limit';
    end    
    if (s(im,i)>2500)&&(s(im,i)<3200)
        signCD='No Parking';
    end

    if (s(im,i)>3200)&&(s(im,i)<3700)
        signCD='No Standing';
    end
    resultCD=sprintf(' CD = %d Sign : %s in %fs',s(i),signCD,CD_execution_time);
    %sign=strr(58:end-13);
    sign=string(testSet.Labels(im,1));
    if strcmp(signCD,sign)
        stateCD='CD Correct Prediction:';
        CDcorrect=CDcorrect+1;
    else
        stateCD=strcat('CD False Prediction:',signCD,' Must Be: ',sign);
        CDfalse=CDfalse+1;
    end
    display(strcat(stateCD,resultCD));

    %if nf>0 

    %try 
     tic
    test_image_gray_binary=roi_candidate{i};  %test_image_gray_binary=roi_candidate{1,i};
    %test_image_gray_binary=imresize(test_image_gray_binary,[signsizeSVM,signsizeSVM],'bilinear');
    %size(trainingFeatures,2)

    hog_4x4=extractHOGFeatures(test_image_gray_binary, 'CellSize', cellSize);
    %testFeatures=zeros(1,size(trainingFeatures,2));
    %testFeatures=zeros(1,100000);
    testFeatures(1,1:size(hog_4x4,2))=hog_4x4;
    % Make class predictions using the test features.
    [predictedLabels,distance2HP] = predict(classifier, testFeatures);
    %accuracy=sum(predict(classifier, testFeatures)==y_test)/length(y_test)*100

    SVM_classification_execution_time=toc;
    SVMtime=SVMtime+SVM_classification_execution_time;
    resultSVM=sprintf(' Detected Sign : %s in %fs',predictedLabels,SVM_classification_execution_time);
    if (sign==predictedLabels)%or( (sign=='Speed Limit')|(predictedLabels(2)=='L') )
        stateSVM='SVM Correct Prediction';
        SVMcorrect=SVMcorrect+1;
    else
        stateSVM=sprintf('SVM False Prediction:%s Must Be:%s',string(predictedLabels),sign);
        SVMfalse=SVMfalse+1;
    end
    display(strcat(stateSVM,resultSVM));
    %catch
    %disp('An error occurred while Classification. Execution will continue.');
    if showpic
        if i==1
            figure
            subplot(3,2,1);imshow(rgb_image);title(sprintf('P=%s N=%d Ns=%d',strr,n,nf));
            subplot(3,2,2);imshow(red_region_f);%title();        
            subplot(3,2,4);imshow(roi_p_candidateCD{im,1});title(sprintf('%s By %d',stateCD,s(im,1)));
            subplot(3,2,3);imshow(roi_candidate{1});title(stateSVM);
        end 
        if i==2    
            subplot(3,2,6);imshow(roi_p_candidateCD{im,2});title(sprintf('%s By %d',stateCD,s(im,2)));
            subplot(3,2,5);imshow(roi_candidate{2});title(stateSVM);
        end
    end
end



%end

end

SVMcorrect
SVMfalse
SVMTruePercent=SVMcorrect/testno
CDcorrect
CDfalse
CDTruePercent=CDcorrect/testno
SVMtime
CDtime
%end