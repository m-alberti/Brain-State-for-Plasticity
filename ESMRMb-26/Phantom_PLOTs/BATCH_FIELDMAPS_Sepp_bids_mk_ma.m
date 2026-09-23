%% BATCH to get B1 / dB0 maps for 2026 ESMRMB abstract Manfredi
% author: mkreis@tuebingen.mpg.de
%
% INFO: This is a BIDS compatible version of BATCH_FieldMaps_Sepp.m with
% less functionaility when i comes to plotting, but saves 'relevant' files
% as NIFTI
% 
%   
% smueller: "Disclaimer: will obviously recycle quite some soffisticated code that
% others had written. Make sure not to violate their rights, i.e. under no
% circumstances do share this code outside MPI!
% This is not very elegant code but just a script that may help to call the
% correct functions :-)"
% 


addpath(genpath('/home/malberti/Unix_Folders/ESMRMB2026_Manfredi/b1map_recos')); % Siemens tfl
addpath(genpath('/home/malberti/Unix_Folders/ESMRMB2026_Manfredi/tinytools'));
   
rawDir = '/home/malberti/wks14/temp/FF_DWI_Drift/rawdata';
derivDir = '/home/malberti/Unix_Folders/ESMRMB2026_Manfredi/';

% Ensure derivatives directory exists
if ~exist(derivDir, 'dir')
    mkdir(derivDir);
end

% Define subjects to process
subjects = 16:45; % Cell array of subject IDs to process

%% Loop over subjects
for s = 1:length(subjects)
    subj = sprintf('%02d', subjects(s));
    fprintf('Processing subject: %s\n', subj);

    % Define subject directories
    subjRawDir = fullfile(rawDir, ['sub-' subj]);
    subjDerivDir = fullfile(derivDir, ['sub-' subj]);

    % Get list of session directories for this subject
   % sessions = dir(fullfile(subjRawDir, 'ses-*'));
   % sessions = sessions([sessions.isdir]);
    sessions = ["ses-01", "ses-13"];
    % Loop over sessions

    for ses = 1:length(sessions)
        
        session = sessions(ses);
        % Define input and output directories for twix files
        inputDir = fullfile(subjRawDir, session, 'twix');
        outputDir = fullfile(subjDerivDir, session, 'twix');

        % Ensure output directory exists
        if ~exist(outputDir, 'dir')
            mkdir(outputDir);
        end

        %%Reco

        fn = dir(fullfile(inputDir, '*.dat')); 
        fn = {fn.name};
        fn = fn(~ismember(fn, {'.', '..'})); % Remove . and ..
  
        
        for nn=1:size(fn,2)
           saveReco(fullfile(inputDir, fn{nn}), fullfile(outputDir, fn{nn}(1:end-4)));
           clear outfile
        end

        %% b1+ plotting
        fn = dir(fullfile(outputDir, '*tfl_map-sag.mat'));
        fn = {fn.name};
        fn = fn(~ismember(fn, {'.', '..'})); 

        for nn=1:size(fn,2)
            P       = load(fullfile(outputDir,fn{nn}) );

            % Extract the base filename (without extension)
            [~, baseName, ~] = fileparts(fn{nn});

            % Save P.alpha, P.phase, P.b1 as NIfTI files
           %if 240 < P.meta.Uref && P.meta.Uref < 242
            %urefStr = sprintf('Uref%.1f', P.meta.Uref);
            urefStr = strrep(num2str(P.meta.Uref), '.', 'p');
            fprintf('Saved NIfTI files for %s (Uref = %f).\n', baseName, P.meta.Uref);
            niftiwrite(P.alpha, fullfile(outputDir, [baseName '_' urefStr '_alpha.nii']))
            niftiwrite(P.nT_per_V, fullfile(outputDir, [baseName '_' urefStr '_nT_per_V.nii']));
            niftiwrite(P.b1, fullfile(outputDir, [baseName '_' urefStr '_b1.nii']));
            %end
        end

        %% dB0 plotting
        fn = dir(fullfile(outputDir, '*field_mapping.mat'));
        fn = {fn.name};
        fn = fn(~ismember(fn, {'.', '..'})); 
        % gyromag ratio in [MHz/T]
        gamma_      = 2.6752218708 * 10^8 / 2  / pi /10^6; %i do think this might now be MHz/T

        for nn=1:size(fn,2)
            [~, baseName, ~] = fileparts(fn{nn});
            P               = load(fullfile(outputDir,fn{nn}) );
            dB0(:,:,:,nn)   = P.deltaB0_Hz / (P.reco.Meas.lFrequency / gamma_ / 10^6); %convert to ppm since this is independent of the field strength so people can interpret it more easy hopefully :-)
            
            uRefStr = strrep(num2str(P.meta.Uref), '.', 'p');

            niftiwrite(dB0, fullfile(outputDir, [baseName '_' uRefStr '_dB0.nii']));
           
        end
    end
end


%% Potatoes here and there
