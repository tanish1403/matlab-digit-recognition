% Initialize workspace
clc;
close all;
clear;

% Load and display the dataset
digitDatasetPath = fullfile(toolboxdir('nnet'), 'nndemos', 'nndatasets', 'DigitDataset');
datastore = imageDatastore(digitDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Show sample images
figure;
totalImages = 10000;
selected = randperm(totalImages, 20);
for idx = 1:20
    subplot(4, 5, idx);
    imshow(datastore.Files{selected(idx)});
    drawnow;
end

% Define split sizes for training, validation, and testing
trainingCount = 600;  
validationCount = 200; 

% Divide the dataset into training, validation, and test sets
[datastoreTrain, datastoreTemp] = splitEachLabel(datastore, trainingCount, 'randomize');
[datastoreValidation, datastoreTest] = splitEachLabel(datastoreTemp, validationCount, 'randomize');

% Define a denser ANN architecture
annLayers = [
    imageInputLayer([28 28 1], 'Normalization', 'none')  % Input layer
    
    % Hidden layers with increased neurons
    fullyConnectedLayer(512)                             % Hidden layer 1
    reluLayer                                            % Activation function
    fullyConnectedLayer(256)                             % Hidden layer 2
    reluLayer                                            % Activation function
    fullyConnectedLayer(128)                             % Hidden layer 3
    reluLayer                                            % Activation function
    fullyConnectedLayer(64)                              % Hidden layer 4
    reluLayer                                            % Activation function
    
    % Output layers
    fullyConnectedLayer(10)                              % Output layer for 10 classes
    softmaxLayer                                         % Softmax activation
    classificationLayer];                                % Classification layer

% Set training options with validation data
trainingOptions = trainingOptions('sgdm', ...
    'MaxEpochs', 20, ...
    'InitialLearnRate', 1e-4, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', datastoreValidation, ...
    'ValidationFrequency', 50, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the denser ANN network
annNet = trainNetwork(datastoreTrain, annLayers, trainingOptions);

% Resize test set to fit the input layer of the network
testDatastoreResized = augmentedImageDatastore([28 28], datastoreTest);

% Perform classification on the test set and evaluate performance
predictions = classify(annNet, testDatastoreResized);
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
