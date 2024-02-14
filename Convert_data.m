% Compute across files. Check files to see how soz needs to be fixed for
% each case
clear;clc;
files = ["/home/ROBARTS/mcespedes/Documents/Data/CC01.mat", "/home/ROBARTS/mcespedes/Documents/Data/CC02.mat", "/home/ROBARTS/mcespedes/Documents/Data/CC03.mat"];
for n_file=1:3
    load(files(n_file));
    %% Data
    data_orig =  bruzzone.bci2000.bipolar;

    %% Initialize variable if not exist
    if ~exist('segment_id', 'var')
        segments_info = cell(1,12);
        segments_info{1,1} = 'index';
        segments_info{1,2} = 'anatomy';
        segments_info{1,3} = 'category_id';
        segments_info{1,4} = 'channel';
        segments_info{1,5} = 'electrode_type';
        segments_info{1,6} = 'institution';
        segments_info{1,7} = 'patient_id';
        segments_info{1,8} = 'reviewer_id';
        segments_info{1,9} = 'segment_id';
        segments_info{1,10} = 'soz';
        segments_info{1,11} = 'category_name';
        segments_info{1,12} = 'time_series_index';
    %     channels = strings(0);
    %     anat = strings(0);
    %     electrode_type = strings(0);
    %     institution = strings(0);
    %     reviewer_id = strings(0);
    %     soz = [];
    %     category = strings(0);
    %     segment_ids = strings(0);
    %     patient_ids = [];
        segment_id = 1;
        patient_id = 1;
    end

    %% Channels
    % Get channels
    channels_names = cell2array(horzcat(preproInfo.bipolarInfo.channelNames{:}));
    % Convert to array
    % channels = cat(2, channels, cell2array(channels_cell));

    % Anatomy
    anat_chn = horzcat(preproInfo.bipolarInfo.anatBipolar{:});
    % anat = cat(2, anat, cell2array(anat_cell));

    %% Electrode type
    size_cell = size(anat_chn);
    % electrode_type_subj = strings(1,size_cell(2));
    electrode_type_subj = "depth";
    % electrode_type = cat(2, electrode_type, electrode_type_subj);

    %% Institution
    % institution_subj = strings(1,size_cell(2));
    institution_subj = "U_Florida";
    % institution = cat(2, institution, institution_subj);


    %% segment_id 
    % segment_id_subj = strings(1,size_cell(2));
    % for i=1:size_cell(2)
    %     segment_id_subj(i) = sprintf('%06d', segment_id);
    %     segment_id = segment_id + 1;
    % end

    %% Patient_id
    % patient_id_subj = strings(1,size_cell(2));
    % patient_id_subj(:) = patient_id;
    % % patient_ids = cat(2, patient_ids, patient_id_subj);
    % patient_id = patient_id + 1;

    %% SOZ
    % delete extra rows in bipolarSOZ element 4 and convert to array
    bipolarSOZ_updated = bipolarSOZ;
    if n_file==1
        bipolarSOZ_updated{4} = bipolarSOZ_updated{4}(1:8);
    elseif n_file==2
        bipolarSOZ_updated{5} = bipolarSOZ_updated{5}(1:12);
    elseif n_file==3
        bipolarSOZ_updated{6} = bipolarSOZ_updated{6}(1:8);
    end
    soz_iter = cat(1,bipolarSOZ_updated{:})';
    % soz = cat(2, soz, soz_iter);

    %% Category ID and name
    % Combine manual labels of 2 reviews in the way: pathological > noise >
    % physiological. In both studies: (1 = artifact | 2 = pathology | 3 = physiology)
    % Combine the automatic label with the manual (manual > automatic)
    events_mitro = cell(1,size(mitro.Events.autoClass,2));
    fs = bruzzone.bci2000.fs;
    for i=1:size(mitro.Events.autoClass,2)
        if ~isempty(mitro.Events.autoIdx{i}) && ~isempty(mitro.Events.manualIdx{i})
            %disp(i);
            all_id = cat(1, int32(mitro.Events.autoIdx{i}*fs-1.5*fs), int32(mitro.Events.manualIdx{i}*fs-1.5*fs));
            all_labels = cat(1, mitro.Events.autoClass{i}, mitro.Events.manualClass{i});
            events_mitro{i} = vertcat(all_id', all_labels');
        elseif ~isempty(mitro.Events.autoIdx{i})
            all_id = int32(mitro.Events.autoIdx{i}*fs-1.5*fs);
            all_labels = mitro.Events.autoClass{i};
            events_mitro{i} = vertcat(all_id', all_labels');
        elseif ~isempty(mitro.Events.manualIdx{i})
            all_id = int32(mitro.Events.manualIdx{i}*fs-1.5*fs);
            all_labels = mitro.Events.manualClass{i};
            events_mitro{i} = vertcat(all_id', all_labels');
        else
            events_mitro{i} = [];
        end
    end

    events_b = cell(1,size(bruzzone.Events.autoClass,2));
    for i=1:size(bruzzone.Events.autoClass,2)
        if ~isempty(bruzzone.Events.autoIdx{i}) && ~isempty(bruzzone.Events.manualIdx{i})
            %disp(i);
            all_id = cat(1, int32(bruzzone.Events.autoIdx{i}*fs-1.5*fs), int32(bruzzone.Events.manualIdx{i}*fs-1.5*fs));
            all_labels = cat(1, bruzzone.Events.autoClass{i}, bruzzone.Events.manualClass{i});
            events_b{i} = vertcat(all_id', all_labels');
        elseif ~isempty(bruzzone.Events.autoIdx{i})
            all_id = int32(bruzzone.Events.autoIdx{i}*fs-1.5*fs);
            all_labels = bruzzone.Events.autoClass{i};
            events_b{i} = vertcat(all_id', all_labels');
        elseif ~isempty(bruzzone.Events.manualIdx{i})
            all_id = int32(bruzzone.Events.manualIdx{i}*fs-1.5*fs);
            all_labels = bruzzone.Events.manualClass{i};
            events_b{i} = vertcat(all_id', all_labels');
        else
            events_b{i} = [];
        end
    end
    % Now iterate over channels and for each channel review if an id is the
    % both sides
    % powerline = 0
    importance = [1,3,4,2]; % this way: pathology has importance 4, powerline of 3, artifacts has importance 2
    events = events_b;
    for chn=1:size(events_b,2)
       chn_info_b = events_b{chn};
       chn_info_m = events_mitro{chn};
       % Both have useful info
       if ~isempty(chn_info_b) && ~isempty(chn_info_m)
           % Check segments that are in mitro but not in bruzzone
           mask_shared = ismember(chn_info_m(1,:), chn_info_b(1,:));
           % Append to events those that are in m but not in b
           if ~all(mask_shared)
               events{chn} = horzcat(events{chn}, chn_info_m(:,~mask_shared));
           end
           % For the rest of the events, compare
           events_repeated_m = chn_info_m(:,mask_shared);
           for seg=1:size(events_repeated_m,2)
               event_m = events_repeated_m(:,seg);
               event_id = event_m(1);
               % Find same event in b
               event_b = chn_info_b(:,chn_info_b(1,:)==event_id);
               % Classify based on importance
               % summing 1 as powerline has a val of 0.
               [~, argmax] = max([importance(event_m(2)+1), importance(event_b(2)+1)]);
               event_list = [event_m(2), event_b(2)];
               % Change event in general events
               %disp(event_list(argmax));
               events{chn}(2,events{chn}(1,:)==event_id) = event_list(argmax);
           end
       % Only mitro has useful info
       elseif ~isempty(chn_info_m)
           events{chn} = chn_info_m;
       end
       % The other case doesn't matter as we are copying bruzzone to events at
       % the beginning
    end

    %% Sort the channels
    for chn=1:size(events,2)
        events_chn = events{chn};
        if ~isempty(events_chn)
            [~,I] = sort(events_chn(1,:));
            events{chn} = events{chn}(:,I);
        end
    end

    %% Create info file

    % Now get all segments, for this create the segments tsv and mat files at
    % the same time. Check if there's overlap of segments and keep based on
    % importance.
    max_idx = size(data_orig,1)-3*fs+1; % +1 as the final segment could go from size(data,1)-3*fs+1 to size(data,1)-3*fs
    for chn=1:size(channels_names,2)
        events_chn = events{1,chn};
        event_id_considered = [];
        for event_id=1:size(events_chn, 2)
            event_seg = events_chn(:,event_id);
            index_event = event_seg(1,1);
            % Check if event has been considered and there are at least 3
            % seconds at it at least begins at zero
            if ~ismember(index_event, event_id_considered) && (event_seg(1) <= max_idx) && (event_seg(1) >= 0)
               [seg_info, segment_id, events_id_eval] = eval_segment(event_id, segment_id, patient_id, chn, events_chn, anat_chn, channels_names, soz_iter, fs, max_idx);
               % Add events evaluated
               event_id_considered = [event_id_considered, events_id_eval];
               % Add to segments info 
               for i=1:size(seg_info,2)
                  segments_info{segment_id,i} = seg_info{1,i}; 
               end
               % Save segments
               data = data_orig(seg_info{1,12}:seg_info{1,12}+3*fs-1, chn)';
               save("DATASET_UFLORIDA/" + seg_info{9}+".mat", "data");
            end
        end
    end
    %% Update patient 
    patient_id = patient_id + 1;
end
writecell(segments_info, 'DATASET_UFLORIDA/segments_new.csv');
% writecell(segments_info, 'segments_tmp.csv');
%% Functions
function [seg_info, id_seg, events_id_eval] = eval_segment(event_id, segment_id, patient_id, chn, events_chn, anat_chn, channels_names, soz_iter, fs, max_idx)
   event_seg = events_chn(:,event_id);   
   index_event = event_seg(1,1);
   events_id_eval = [];
   importance = [1,3,4,2]; % this way: pathology has importance 4, powerline of 3, artifacts has importance 2
   % Check if there's any other event less than 3 seconds away before the
   % end of the file - 3 seconds
   end_id = min([index_event + 1*fs, max_idx+1]); % +1 due to the < below
   mask_events = events_chn(1,:) >= index_event & events_chn(1,:) < end_id;
   overlapping_events = events_chn(:,mask_events);
   %disp(event_id);
   %disp(overlapping_events);
   %disp(chn);
   % if no overlapping events, just add it
   if isempty(overlapping_events)
       % Add info
       seg_info = get_seg_info(segment_id, anat_chn{1,chn}, event_seg, channels_names{1,chn}, patient_id, soz_iter(1,chn));
       id_seg = segment_id + 1;
       events_id_eval = [events_id_eval, index_event];
   else
       % get most important event
       %disp(overlapping_events(2,:));
       [~, argmax] = max(importance(overlapping_events(2,:)+1));
       % Case 1: the current event is the most important
       if argmax == 1
           % Add info
           seg_info = get_seg_info(segment_id, anat_chn{1,chn}, event_seg, channels_names{1,chn}, patient_id, soz_iter(1,chn));
           id_seg = segment_id + 1;
           events_id_eval = [events_id_eval, overlapping_events(1,:)];
       else
           % Repeat with the max sample
           % but first attach to events evaluated the ones that are below
           % the max sample
           events_id_eval = [events_id_eval, overlapping_events(1,1:argmax-1)];
           [seg_info, id_seg, events_id_eval_next] = eval_segment(event_id+argmax-1, segment_id, patient_id, chn, events_chn, anat_chn, channels_names, soz_iter, fs, max_idx);
           events_id_eval = [events_id_eval, events_id_eval_next];
       end
   end
end

function [seg_info] = get_seg_info(segment_id, anat_seg, event_seg, chn_name, patient_id, chn_soz)
   seg_info = cell(1,11);
   % index
   seg_info{1,1} = segment_id;
   % anatomy
   seg_info{1,2} = anat_seg;
   % category_id
   seg_info{1,3} = event_seg(2,1);
   % channel
   seg_info{1,4} = chn_name;
   % electrode_type
   seg_info{1,5} = "depth";
   % institution	
   seg_info{1,6} = "UFlorida";
   % patient_id
   seg_info{1,7} = patient_id;  % reviewer_id: not distinguising between the 2
   seg_info{1,8} = 1;
   % segment_id
   seg_info{1,9} = sprintf('x%06d', segment_id);
   % soz
   seg_info{1,10} = chn_soz;
   % category_name
   category_name = ["powerline", "noise", "pathology", "physiology"];
   seg_info{1,11} = category_name(event_seg(2,1)+1);
   % time series id
   seg_info{1,12} = event_seg(1,1);
end

function [array] = cell2array(A)
    size_cell = size(A);
    array = strings(1,size_cell(2));
    for i=1:size_cell(2)
        array(i) = A(i);
    end
end