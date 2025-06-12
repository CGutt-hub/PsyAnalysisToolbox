clear; clc;
addpath(genpath('classes'));
config = config();
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

EEG = Loader.loadXDF(config);
EEG = ChannelEditor.addLocations(EEG);
EEG = ChannelEditor.removeChannels(EEG, config.exclude_channels);
EEG = Preprocessor.resample(EEG, 128);
EEG = ArtifactCleaner.cleanBadChannels(EEG);
EEG = Preprocessor.rereference(EEG);
EEG = Preprocessor.bandpassFilter(EEG, 0.1, 30);

[EEG, EEG_ica] = ICAHandler.runICA(EEG);
EEG = ICAHandler.classifyAndReject(EEG);

EEG = Epoching.epochAndBaseline(EEG);
Saver.save(EEG, config);
Plotter.saveICATopoplot(EEG, config);

fprintf('\nPreprocessing complete for %s!\n', config.filename);
