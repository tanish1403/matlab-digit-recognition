% Clear workspace and set up paths
clc;
close all;
clear;

% Load dataset
digitDatasetPath = fullfile(toolboxdir('nnet'), 'nndemos', 'nndatasets', 'DigitDataset');
imds = imageDatastore(digitDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Display some sample images
figure;
numImages = 10000;
perm = randperm(numImages, 20);
for i = 1:20
    subplot(4, 5, i);
    imshow(imds.Files{perm(i)});
    drawnow;
end

% Split dataset into training, validation, and test sets
numTrainingFiles = 600;  % Number of training images per class
numValidationFiles = 200; % Number of validation images per class

[imdsTrain, imdsTemp] = splitEachLabel(imds, numTrainingFiles, 'randomize');
[imdsValidation, imdsTest] = splitEachLabel(imdsTemp, numValidationFiles, 'randomize');

% Define a denser CNN architecture
layers = [
    imageInputLayer([28 28 1])
    
    % First Convolution Block
    convolution2dLayer(5, 32, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    % Second Convolution Block
    convolution2dLayer(3, 64, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
     
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

% Set up training options with validation data
options = trainingOptions('sgdm', ...
    'MaxEpochs', 20, ...
    'InitialLearnRate', 1e-4, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency', 50, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the network
net = trainNetwork(imdsTrain, layers, options);

% Resize test set using augmentedImageDatastore to match input size [28 28 1]
augmentedTest = augmentedImageDatastore([28 28], imdsTest);

% Classify and evaluate on test set
YPred = classify(net, augmentedTest);
YTest = imdsTest.Labels;

% Calculate accuracy on test set
accuracy = sum(YPred == YTest) / numel(YTest);

% Confusion matrix and additional metrics on test set
confMat = confusionmat(YTest, YPred);
precision = diag(confMat) ./ sum(confMat, 2);
recall = diag(confMat) ./ sum(confMat, 1)';
f1Scores = 2 * (precision .* recall) ./ (precision + recall);

% Overall metrics
overallPrecision = mean(precision, 'omitnan');
overallRecall = mean(recall, 'omitnan');
overallF1Score = mean(f1Scores, 'omitnan');

% Display overall metrics
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Overall Precision: %.2f\n', overallPrecision);
fprintf('Overall Recall: %.2f\n', overallRecall);
fprintf('Overall F1 Score: %.2f\n', overallF1Score);

% Plot the confusion matrix
figure;
plotconfusion(YTest, YPred);
