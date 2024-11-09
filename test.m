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

% Load pre-trained MobileNetV2 and modify for digit classification
net = mobilenetv2;
inputSize = net.Layers(1).InputSize;  % [224 224 3]

% Convert the network to layerGraph for easier modification
lgraph = layerGraph(net);

% Replace the final fully connected and classification layers for digit classification
numClasses = numel(categories(imdsTrain.Labels));
newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'new_fc')
    softmaxLayer('Name', 'new_softmax')
    classificationLayer('Name', 'new_output')];

% Remove the original final layers and add new layers
lgraph = replaceLayer(lgraph, 'Logits', newLayers(1));
lgraph = replaceLayer(lgraph, 'Logits_softmax', newLayers(2));
lgraph = replaceLayer(lgraph, 'ClassificationLayer_Logits', newLayers(3));

% Set up the training and validation datastores with resizing
augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'ColorPreprocessing', 'gray2rgb');
augValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation, 'ColorPreprocessing', 'gray2rgb');

% Define training options
options = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-4, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the modified MobileNetV2
mobilenetTransfer = trainNetwork(augTrain, lgraph, options);

% Resize test set to fit the input layer of the network
augTest = augmentedImageDatastore(inputSize(1:2), imdsTest, 'ColorPreprocessing', 'gray2rgb');

% Classify and evaluate on test set
YPred = classify(mobilenetTransfer, augTest);
YTest = imdsTest.Labels;

% Calculate accuracy on test set
accuracy = sum(YPred == YTest) / numel(YTest);

% Compute confusion matrix and related metrics on test set
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

