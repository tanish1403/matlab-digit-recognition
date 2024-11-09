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

% Split dataset into training, validation, and test sets
trainingCount = 600;  
validationCount = 200;  

[datastoreTrain, datastoreTemp] = splitEachLabel(datastore, trainingCount, 'randomize');
[datastoreValidation, datastoreTest] = splitEachLabel(datastoreTemp, validationCount, 'randomize');

% Load pre-trained VGG-16 and prepare for transfer learning
net = vgg16;
inputSize = net.Layers(1).InputSize;  % [224 224 3]

% Convert the network to layerGraph for easier modification
lgraph = layerGraph(net);

% Replace the final fully connected and classification layers
numClasses = numel(categories(datastoreTrain.Labels));
newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'fc10')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')];

% Remove the original layers and add new layers to the graph
lgraph = replaceLayer(lgraph, 'fc8', newLayers(1));
lgraph = replaceLayer(lgraph, 'prob', newLayers(2));
lgraph = replaceLayer(lgraph, 'output', newLayers(3));

% Set up the training and validation datastores with resizing
augTrain = augmentedImageDatastore(inputSize(1:2), datastoreTrain, 'ColorPreprocessing', 'gray2rgb');
augValidation = augmentedImageDatastore(inputSize(1:2), datastoreValidation, 'ColorPreprocessing', 'gray2rgb');

% Define training options
trainingOptions = trainingOptions('sgdm', ...
    'MaxEpochs', 20, ...
    'InitialLearnRate', 1e-4, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augValidation, ...
    'ValidationFrequency', 50, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the modified VGG-16 network
vggNetTransfer = trainNetwork(augTrain, lgraph, trainingOptions);

% Resize test set to fit the input layer of the network
augTest = augmentedImageDatastore(inputSize(1:2), datastoreTest, 'ColorPreprocessing', 'gray2rgb');

% Classify and evaluate on the test set
predictions = classify(vggNetTransfer, augTest);
actualLabels = datastoreTest.Labels;

% Calculate accuracy on the test set
testAccuracy = sum(predictions == actualLabels) / numel(actualLabels);

% Compute confusion matrix and related metrics for the test set
confMatrix = confusionmat(actualLabels, predictions);
precisionVals = diag(confMatrix) ./ sum(confMatrix, 2);
recallVals = diag(confMatrix) ./ sum(confMatrix, 1)';
f1Scores = 2 * (precisionVals .* recallVals) ./ (precisionVals + recallVals);

% Overall precision, recall, and F1 score
avgPrecision = mean(precisionVals, 'omitnan');
avgRecall = mean(recallVals, 'omitnan');
avgF1Score = mean(f1Scores, 'omitnan');

% Display metrics in the command window
fprintf('Test Accuracy: %.2f%%\n', testAccuracy * 100);
fprintf('Average Precision: %.2f\n', avgPrecision);
fprintf('Average Recall: %.2f\n', avgRecall);
fprintf('Average F1 Score: %.2f\n', avgF1Score);

% Plot the confusion matrix
figure;
plotconfusion(actualLabels, predictions);

% Placeholder arrays for precision, recall, and F1 score per epoch
numEpochs = trainingOptions.MaxEpochs;
precisionHistory = repmat(avgPrecision, 1, numEpochs);
recallHistory = repmat(avgRecall, 1, numEpochs);
f1History = repmat(avgF1Score, 1, numEpochs);

% Plot precision, recall, and F1 score across epochs
figure;
plot(1:numEpochs, precisionHistory, '-o', 'LineWidth', 1.5); hold on;
plot(1:numEpochs, recallHistory, '-s', 'LineWidth', 1.5);
plot(1:numEpochs, f1History, '-^', 'LineWidth', 1.5);
title('Epoch-wise Precision, Recall, and F1 Score');
xlabel('Epoch');
ylabel('Score');
legend('Precision', 'Recall', 'F1 Score');
grid on;
