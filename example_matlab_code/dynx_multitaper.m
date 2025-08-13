clear;

path_dynx = pwd;

%%
% Load DyNeuMo-X data
dynx_eeg.fs = 4000; % Sampling rate (Hz)
dynx_eeg.gain = 24; % Channel gain (x)

x = adc_to_uV(read_binary(path_dynx, 'SdioLogger.bin', 9), dynx_eeg.gain);
dynx_eeg.t = 0:1/dynx_eeg.fs:(length(x)-1)/dynx_eeg.fs;
dynx_eeg.raw = x(7, :) - x(1, :);

clear x;

% Filter data
dynx_eeg.offset = filter_offset(dynx_eeg.raw);
dynx_eeg.filt = filter_band(dynx_eeg.offset, [0.1, 40], dynx_eeg.fs);

% Multitaper setup
df = 0.5; % Frequency resolution (Hz)
N  = 15;  % Stationary window (s)

TW = N*df/2; % Time-bandwidth product (Hz*s)
L = floor(2*TW)-1; % Number of tapers

frequency_range=[0; 25]; % Limit frequencies from 0 to 25 Hz
taper_params=[TW; L]; % Time bandwidth and number of tapers
window_params=[30; 15]; % Window size is 30s with step size of 15s
min_nfft=0; % No minimum nfft
detrend_opt='constant'; % detrend each window by subtracting the average
weighting='unity'; % weight each taper at 1
plot_on=true; % plot spectrogram
verbose=false; % print extra info

%%
% Display the signal and its multitaper spectrogram
figure(1); clf;
tiledlayout(1, 1, "TileSpacing", "compact", "Padding", "compact");
%{
nexttile;
plot(dynx_eeg.t, dynx_eeg.filt, 'Color', 'k');
ylabel("Signal (μV)");
xticks(0:900:length(dynx_eeg.filt));
xticklabels(0:0.25:length(dynx_eeg.filt)/900);
xlim([0, 6*3600]);
%}
nexttile;
multitaper_spectrogram(dynx_eeg.filt, dynx_eeg.fs, frequency_range, taper_params, window_params, min_nfft, detrend_opt, weighting, plot_on, verbose);
colormap(flipud(lbmap(100, 'RedBlue')));

xticks(0:1800:length(dynx_eeg.filt));
xticklabels(0:0.5:length(dynx_eeg.filt)/1800);
xlim([0, 6*3600]);
xlabel("Time (h)");

set_size(18, 8, 11);
%export(pwd, 'fig');


%%

function [data] = read_intan_rhd(path, analog_chs, digital_chs)
if isempty(path)
  error('Path does not point to a valid directory');
end

files = dir(fullfile(path, '*.rhd'));
if isempty(files)
  error('No .rhd files in directory');
end

data.raw = [];
data.dig = [];
for i = 1:length(files)
  read_Intan_RHD2000_file(path, files(i).name, 'caller');
  data.raw = [data.raw, amplifier_data(analog_chs, :)]; %#ok<USENS> 
  data.dig = [data.dig, board_dig_in_data(digital_chs, :)]; %#ok<USENS> 
end

data.fs = frequency_parameters.amplifier_sample_rate;
data.t = 0:1/data.fs:(length(data.raw)-1)/data.fs;
end



function [x] = read_binary(path, file_name, num_channels)

fileinfo = dir(fullfile(path, file_name));
num_samples = fileinfo.bytes / 4 / num_channels; % int32 = 4 bytes

fid = fopen(fullfile(path, file_name), 'r');
x = fread(fid, [num_channels, num_samples], 'int32'); % read as ADC samples
fclose(fid);

end


function [y] = filter_offset(x)
if width(x) < height(x)
  x = x';
  unflip = 1;
else
  unflip = 0;
end

x = [x(:, 1), x];
y = zeros(height(x), width(x));
for i = 2:length(x)
  y(:, i) = 0.996 * (y(:, i-1) + x(:, i) - x(:, i-1));
end

if unflip
 y = y(:, 2:end)';
else
  y = y(:, 2:end);
end
end


function [x] = filter_harmonics(x, f, fs, bw)

harmonics = setdiff(f:f:(fs/2), fs/2);

for i = 1:length(harmonics)
  [b, a] = butter(2, harmonics(i) * [1-bw, 1+bw] / (fs/2), "stop");
  if width(x) > height(x)
    % x = filter(b, a, x')';
    x = filtfilt(b, a, x')';
  else
    % x = filter(b, a, x);
    x = filtfilt(b, a, x);
  end
end

end


function [y] = filter_band(x, f, fs)
[b, a] = butter(2, f / (fs/2), "bandpass");
y = filtfilt(b, a, x')';
end


function [y] = adc_to_uV(x, gain)
y = x * ((2.*4.5)./gain)./(2.^24).*1e6;
end


function plot_patch(t, x_in, color, alpha)

x = double(x_in);

% 1 marks first and last sample of HIGH logic level
tmp1 = find(xor([diff(x) < 0, 0], [0, diff(x) > 0]));
% +/- 1 offset for falling / rising edges to insert LOW before / after
% event marker as appropriate based on edge type
tmp2 = - x([0, diff(x)] ~= 0) * 2 + 1;

[s, ix] = sort([tmp1, tmp1 + tmp2]);
x_out = ones(1, 2 * length(tmp1));
x_out(ix > length(ix) / 2) = 0;

ts = t(s);

x_out_scaled = x_out;
x_out_scaled(x_out==0) = min(ylim);
x_out_scaled(x_out==1) = max(ylim);

patch(ts, x_out_scaled, ...
      color, 'EdgeColor', 'none', 'FaceAlpha', alpha);

end


function set_size(im_width, im_height, im_fsz)
if strcmp(gcf().Children.Type, 'tiledlayout')
  for i = 1:prod(gcf().Children.GridSize)
    nexttile(i);
    set(gca, 'FontSize', im_fsz);
  end
end
set(gcf,'Units', 'centimeters');
pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1), pos(2), im_width, im_height]);
papersize = get(gcf,'PaperSize');
figuresize=[(papersize(1) - im_width) / 2, ...
            (papersize(2) - im_height) / 2, ...
            im_width, im_height];
set(gcf,'InvertHardcopy', 'on',...
        'PaperUnits', 'centimeters',...
        'PaperPosition', figuresize);
end


function export(path, file_name)
print(fullfile(path, [file_name, '.png']), '-dpng', '-r300');
%set(gcf, 'renderer', 'Painters');
%print('-depsc','-tiff','-r300', ... %'-painters', ...
%      fullfile(path, [file_name, '.eps']));
end
