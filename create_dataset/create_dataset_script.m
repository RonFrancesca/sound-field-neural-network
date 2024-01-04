clear all
close all
clc

root_path = '/nas/home/fronchini/sound-field-neural-network/'
path = strcat(root_path, '/dataset/test');
mkdir(path);
num_rooms = 1;
seed = 1;
plot = 0;


T60_list = 0.6:0.2:1.2;

% create one folder or each T60
for T60 = T60_list
    mkdir(strcat(path, '/T60_', num2str(T60)));
end


rng(1)

for j = 1:num_rooms
    display(j)
    global Setup
    init

    % keyboard
    %% Generate observed data
    % Calculate baseline frequency response
    FreqLim = 400;

    
    for T60 = T60_list
        [Psi_r,Mu,Psi_s] = Green3D_freq_ModalResponse_Z0(FreqLim, T60);

        % [Psi_rA,Mu,Psi_s] = Green3D_freq_ModalResponseJacobsen(FreqLim);
        FrequencyResponse = zeros(size(Psi_r,1),size(Mu,2),size(Psi_s,2));
        for i = 1:size(Mu,2)
            FrequencyResponse(:,i,:) = Psi_r * diag(Mu(:,i)) * Psi_s;
        end

        %% Reshape Psi_r and FrequencyResponse to three dimensional arrays of x,y,omega
        % Define the x, and y-coordinates
        xCoor = 0:Setup.Observation.xSamplingDistance:Setup.Room.Dim(1);
        yCoor = 0:Setup.Observation.ySamplingDistance:Setup.Room.Dim(2);
        % Calculated frequencies
        Frequency = 0:1/(Setup.Duration):Setup.Fs/2-1/(Setup.Duration);
        %Psi_r = reshape(Psi_r,length(xCoor),length(yCoor),size(Psi_r,2));
        FrequencyResponse = reshape(FrequencyResponse,length(xCoor),length(yCoor),length(Frequency));
        AbsFrequencyResponse = abs(FrequencyResponse);
        AbsFrequencyResponse = AbsFrequencyResponse(:, :, :);
        filename = strcat(num2str(j), '_t_', num2str(T60), '_d_', num2str(Setup.Room.Dim(1)), '_', num2str(Setup.Room.Dim(2)), '_', num2str(Setup.Room.Dim2), '_s_', num2str(Setup.Source.Position{1}(1)), '_', num2str(Setup.Source.Position{1}(2)),'_.mat');
        save_path = strcat(path, '/T60_', num2str(T60));
        save(fullfile(save_path, filename));
        if plot
            % Draw the simulated setup
            DrawSetup
            title('Simulation setup')

            %% Plot the Transfer function at a given frequency
            %Pick a frequency index
            freqIdx = 100;


            figure
            contourf(xCoor, yCoor, AbsFrequencyResponse(:,:,freqIdx).','edgeColor','none');
            xlabel('X-dimension [m]'); ylabel('Y-dimension [m]');
            title(['Contour plot of TF magnitude thoughout the room at f = ' num2str(Frequency(freqIdx),'%.1f') ' Hz'])
        end
    end
end
