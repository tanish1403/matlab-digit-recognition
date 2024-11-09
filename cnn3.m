% Initialize workspace
clc;
close all;
clear;

% Load and display the dataset
digitDatasetPath = fullfile(toolboxdir('nnet'), 'nndemos', 'nndatasets', 'DigitDataset');
datastore = imageDatastore(digitDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Define split sizes for training, validation, and testing
trainingCount = 600;  
validationCount = 200; 

% Divide the dataset into training, validation, and test sets
[datastoreTrain, datastoreTemp] = splitEachLabel(datastore, trainingCount, 'randomize');
[datastoreValidation, datastoreTest] = splitEachLabel(datastoreTemp, validationCount, 'randomize');

% Define a smaller CNN architecture
cnnLayers = [
    imageInputLayer([28 28 1], 'Normalization', 'zerocenter') % Input layer
    
    % First Convolution Block
    convolution2dLayer(3, 32, 'Padding', 'same')  % 32 filters of size 3x3
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)            % Downsampling
    
    % Second Convolution Block
    convolution2dLayer(3, 64, 'Padding', 'same')  % 64 filters of size 3x3
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)            % Downsampling
    
    % Fully Connected Layers
    fullyConnectedLayer(128)                     % Fully connected layer with 128 neurons
    reluLayer
    dropoutLayer(0.5)                            % Dropout layer to reduce overfitting
    fullyConnectedLayer(10)                      % Output layer with 10 neurons
    softmaxLayer
    classificationLayer];

% Set training options with Adam optimizer
options = trainingOptions('adam', ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', datastoreValidation, ...
    'ValidationFrequency', 30, ...
    'InitialLearnRate', 3e-4, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the CNN
cnnNet = trainNetwork(datastoreTrain, cnnLayers, options);

% Resize test set to fit the input layer of the network
testDatastoreResized = augmentedImageDatastore([28 28], datastoreTest);

% Perform classification on the test set and evaluate performance
predictions = classify(cnnNet, testDatastoreResized);
actualLabels = datastoreTest.Labels;

% Calculate accuracy on test set
testAccuracy = sum(predictions == actualLabels) / numel(actualLabels);

% Compute confusion matrix and related metrics for test set
confMatrix = confusionmat(actualLabels, predictions);
precisionVals = diag(confMatrix) ./ sum(confMatrix, 2);
recallVals = diag(confMatrix) ./ sum(confMatrix, 1)';
f1Scores = 2 * (precisionVals .* recallVals) ./ (precisionVals + recallVals);

% Overall precision, recall, and F1 score
avgPrecision = mean(precisionVals, 'omitnan');
avgRecall = mean(recallVals, 'omitnan');
avgF1Score = mean(f1Scores, 'omitnan');

% Display metrics in command window
fprintf('Test Accuracy: %.2f%%\n', testAccuracy * 100);
fprintf('Average Precision: %.2f\n', avgPrecision);
fprintf('Average Recall: %.2f\n', avgRecall);
fprintf('Average F1 Score: %.2f\n', avgF1Score);

% Plot the confusion matrix
figure;
plotconfusion(actualLabels, predictions);
