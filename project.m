clc; clear; close all;

%% ── 1. LOAD YOUR IMAGE ───────────────────────────────────────────────────────
img_path = 'images.jpg';
img_rgb  = imread(img_path);
img_rgb  = imresize(img_rgb, [256 256]);
img_gray = im2double(rgb2gray(img_rgb));

% Get original file size
orig_info = dir(img_path);
orig_kb   = orig_info.bytes / 1024;
fprintf('Original image size: %.2f kB\n', orig_kb);

%% ── 2. PARAMETERS ────────────────────────────────────────────────────────────
Q_levels = [2, 8, 20];
Q_labels = {'Low Compression Q=2', 'Medium Compression Q=8', 'High Compression Q=20'};

%% ── 3. FEATURE EXTRACTION + MANUAL NEURAL NET ───────────────────────────────
fprintf('Extracting features and training neural net...\n');
patchSize = 32;
[R, C]    = size(img_gray);

patches    = zeros(patchSize, patchSize, 500);
qualLabels = zeros(500, 1);
X_feat     = zeros(500, 3);

for p = 1:500
    r = randi(R - patchSize);
    c = randi(C - patchSize);
    patch = img_gray(r:r+patchSize-1, c:c+patchSize-1);
    patches(:,:,p) = patch;

    v  = var(patch(:));
    D  = dct2(patch);
    te = sum(D(:).^2);
    hfe = sum(sum(D(17:end,17:end).^2));
    [gx,gy] = gradient(patch);
    es = mean(sqrt(gx(:).^2 + gy(:).^2));

    complexity     = 0.5*v + 0.3*(hfe/(te+eps)) + 0.2*es;
    qualLabels(p)  = 1 + 19 / (1 + exp(-60*(complexity - 0.03)));
    X_feat(p,:)    = [v, hfe/(te+eps), es];
end

% Normalise
X_min  = min(X_feat);
X_max  = max(X_feat);
X_norm = (X_feat - X_min) ./ (X_max - X_min + eps);

% Train/val split
splitIdx = round(0.8*500);
Xtr = X_norm(1:splitIdx,:);  Ytr = qualLabels(1:splitIdx);
Xvl = X_norm(splitIdx+1:end,:); Yvl = qualLabels(splitIdx+1:end);

% Manual 2-layer neural net
rng(42);
W1 = randn(3,16)*sqrt(2/3);  b1 = zeros(1,16);
W2 = randn(16,1)*sqrt(2/16); b2 = 0;
relu  = @(x) max(0,x);
drelu = @(x) double(x>0);
lr = 0.01;

for ep = 1:300
    Z1 = Xtr*W1+b1; A1 = relu(Z1); Z2 = A1*W2+b2;
    err = Z2 - Ytr;
    dZ2 = 2*err/size(Xtr,1);
    dW2 = A1'*dZ2; db2 = sum(dZ2);
    dA1 = dZ2*W2'; dZ1 = dA1.*drelu(Z1);
    dW1 = Xtr'*dZ1; db1 = sum(dZ1);
    W1=W1-lr*dW1; b1=b1-lr*db1;
    W2=W2-lr*dW2; b2=b2-lr*db2;
end
fprintf('Neural net training complete!\n');

%% ── 4. PREDICT Q FOR YOUR IMAGE ──────────────────────────────────────────────
cx = round(R/2)-16; cy = round(C/2)-16;
cp = img_gray(cx:cx+31, cy:cy+31);
v=var(cp(:)); D=dct2(cp); te=sum(D(:).^2);
hfe=sum(sum(D(17:end,17:end).^2));
[gx,gy]=gradient(cp); es=mean(sqrt(gx(:).^2+gy(:).^2));
feat_norm = ([v,hfe/(te+eps),es] - X_min)./(X_max-X_min+eps);
predicted_Q = relu(feat_norm*W1+b1)*W2+b2;
predicted_Q = max(1, min(20, predicted_Q));
fprintf('DL predicted Q = %.2f\n', predicted_Q);

%% ── 5. COMPRESS & SAVE ALL VERSIONS ─────────────────────────────────────────
all_Q      = [Q_levels, round(predicted_Q)];
all_labels = [Q_labels, {sprintf('DL Predicted Q=%.0f', predicted_Q)}];
numCols    = length(all_Q);

recon_imgs  = cell(1, numCols);
psnr_all    = zeros(1, numCols);
ssim_all    = zeros(1, numCols);
filesize_kb = zeros(1, numCols);    % ← file sizes in kB
filenames   = cell(1, numCols);

% Output folder
outFolder = 'compressed_outputs';
if ~exist(outFolder, 'dir'), mkdir(outFolder); end

for k = 1:numCols
    %-- Compress
    recon_imgs{k} = blockDCT(img_gray, all_Q(k));

    %-- Convert back to uint8 RGB for saving as JPEG
    recon_rgb = repmat(uint8(recon_imgs{k} * 255), [1 1 3]);

    %-- Save with JPEG quality tied to Q:
    %   low Q  (fine quant) → high JPEG quality
    %   high Q (coarse)     → low  JPEG quality
    jpeg_quality = max(10, min(95, round(100 - all_Q(k)*4)));

    fname = fullfile(outFolder, sprintf('compressed_Q%d.jpg', all_Q(k)));
    imwrite(recon_rgb, fname, 'jpg', 'Quality', jpeg_quality);
    filenames{k} = fname;

    %-- Read back actual saved file size
    finfo = dir(fname);
    filesize_kb(k) = finfo.bytes / 1024;

    %-- Quality metrics
    mse = mean((img_gray - recon_imgs{k}).^2, 'all');
    if mse < eps, psnr_all(k) = 100;
    else,         psnr_all(k) = 10*log10(1/mse);
    end
    ssim_all(k) = ssim(recon_imgs{k}, img_gray);

    fprintf('Q=%-3d | JPEG quality=%2d%% | Size=%6.2f kB | PSNR=%5.1f dB | SSIM=%.3f\n', ...
        all_Q(k), jpeg_quality, filesize_kb(k), psnr_all(k), ssim_all(k));
end

%-- Compression ratio vs original
compress_ratio = orig_kb ./ filesize_kb;

%% ── 6. COMPARISON FIGURE: Original vs All Compressed ────────────────────────
figure('Name','Original vs Compressed','Position',[20 20 1400 420]);

% Original
subplot(1, numCols+1, 1);
imshow(img_rgb);
title({'\bf ORIGINAL', sprintf('%.2f kB', orig_kb)}, ...
      'FontSize',11,'Interpreter','tex','Color','k');
xlabel('(reference)','FontSize',9);
axis image;

% Compressed versions
for k = 1:numCols
    subplot(1, numCols+1, k+1);
    imshow(recon_imgs{k}, []);

    % Green title for DL result, black for others
    col = 'k';
    if k == numCols, col = [0 0.5 0]; end

    title({sprintf('\\bf %s', all_labels{k}), ...
           sprintf('%.2f kB  (%.1fx smaller)', filesize_kb(k), compress_ratio(k))}, ...
          'FontSize',10,'Interpreter','tex','Color',col);
    xlabel(sprintf('PSNR=%.1f dB   SSIM=%.3f', psnr_all(k), ssim_all(k)), ...
           'FontSize',9);
    axis image;
end
sgtitle('Original vs Compressed Images  |  Adaptive DCT + Neural Net', ...
        'FontSize',13,'FontWeight','bold');

%% ── 7. SIDE-BY-SIDE ZOOM COMPARISON (centre crop 64×64) ─────────────────────
cx64 = round(R/2)-32;  cy64 = round(C/2)-32;
orig_crop = img_gray(cx64:cx64+63, cy64:cy64+63);

figure('Name','Zoomed Comparison (Centre Crop)','Position',[20 20 1200 340]);

subplot(1, numCols+1, 1);
imshow(imresize(orig_crop, [128 128], 'nearest'));
title({'\bf ORIGINAL (zoomed)', sprintf('%.2f kB', orig_kb)}, ...
      'FontSize',11,'Interpreter','tex');
axis image;

for k = 1:numCols
    recon_crop = recon_imgs{k}(cx64:cx64+63, cy64:cy64+63);
    subplot(1, numCols+1, k+1);
    imshow(imresize(recon_crop, [128 128], 'nearest'));
    col = 'k'; if k==numCols, col=[0 0.5 0]; end
    title({sprintf('\\bf Q=%d  →  %.2f kB', all_Q(k), filesize_kb(k))}, ...
          'FontSize',11,'Interpreter','tex','Color',col);
    xlabel(sprintf('PSNR=%.1f dB', psnr_all(k)),'FontSize',9);
    axis image;
end
sgtitle('Zoomed Centre Crop  — Compression Artifact Visibility', ...
        'FontSize',12,'FontWeight','bold');

%% ── 8. DIFFERENCE MAPS ───────────────────────────────────────────────────────
figure('Name','Pixel Difference Maps','Position',[20 20 1200 310]);
for k = 1:numCols
    subplot(1, numCols, k);
    diff_amp = abs(img_gray - recon_imgs{k}) * 5;
    imshow(diff_amp, []);
    colormap(gca, hot);
    title(sprintf('Error Map  Q=%d\n%.2f kB  (×5 amplified)', ...
          all_Q(k), filesize_kb(k)), 'FontSize',10);
    axis image;
end
sgtitle('Pixel-wise Error  (brighter = more data lost)', ...
        'FontSize',12,'FontWeight','bold');

%% ── 9. FILE SIZE + QUALITY METRICS CHART ────────────────────────────────────
figure('Name','Compression Analytics','Position',[80 80 1100 420]);

labels_short = arrayfun(@(q) sprintf('Q=%d',q), all_Q, 'UniformOutput',false);
colors = [0.2 0.5 0.8; 0.3 0.7 0.3; 0.8 0.4 0.2; 0.5 0.2 0.7];

% File size bar
subplot(1,3,1);
b = bar(filesize_kb, 'FaceColor','flat');
b.CData = colors;
hold on;
yline(orig_kb,'--r','LineWidth',1.5);
text(numCols+0.3, orig_kb, sprintf('Original\n%.1f kB',orig_kb), ...
     'FontSize',8,'Color','r','VerticalAlignment','bottom');
ylabel('File size (kB)');
title('File Size Comparison');
xticks(1:numCols); xticklabels(labels_short);
grid on;

% PSNR bar
subplot(1,3,2);
b2 = bar(psnr_all,'FaceColor','flat');
b2.CData = colors;
ylabel('PSNR (dB)'); ylim([0 50]);
title('PSNR (higher = better)');
xticks(1:numCols); xticklabels(labels_short);
yline(30,'--r','Good (30dB)','LabelHorizontalAlignment','left');
grid on;

% SSIM bar
subplot(1,3,3);
b3 = bar(ssim_all,'FaceColor','flat');
b3.CData = colors;
ylabel('SSIM'); ylim([0 1]);
title('SSIM (higher = better)');
xticks(1:numCols); xticklabels(labels_short);
yline(0.8,'--r','Good (0.8)','LabelHorizontalAlignment','left');
grid on;

sgtitle('Compression Analytics: File Size vs Quality', ...
        'FontSize',13,'FontWeight','bold');

%% ── 10. PRINT FULL SUMMARY TABLE ────────────────────────────────────────────
fprintf('\n%s\n', repmat('=',1,70));
fprintf('%-22s %10s %10s %10s %10s %8s\n', ... 
        'Version','Size(kB)','Ratio','PSNR(dB)','SSIM','Saved%');
fprintf('%s\n', repmat('-',1,70));
for k = 1:numCols
    saved_pct = (1 - filesize_kb(k)/orig_kb)*100;
    fprintf('%-22s %10.2f %10.1fx %10.1f %10.3f %7.1f%%\n', ...
            all_labels{k}, filesize_kb(k), compress_ratio(k), ...
            psnr_all(k), ssim_all(k), saved_pct);
end
fprintf('%s\n', repmat('=',1,70));
fprintf('Compressed files saved to: %s/\n', outFolder);

%% ── LOCAL FUNCTION ───────────────────────────────────────────────────────────
function recon = blockDCT(img_d, Q)
    [R, C] = size(img_d);
    recon  = zeros(R, C);
    for r = 1:8:R-7
        for c = 1:8:C-7
            blk   = img_d(r:r+7, c:c+7);
            coeff = dct2(blk);
            quant = round(coeff ./ Q);
            recon(r:r+7, c:c+7) = idct2(quant .* Q);
        end
    end
    recon = min(max(recon, 0), 1);
end

%% COLOR DCT — compress each channel separately
function recon_rgb = colorBlockDCT(img_rgb, Q)
    % Convert RGB → YCbCr (how real JPEG works)
    img_ycbcr = rgb2ycbcr(img_rgb);
    recon_ycbcr = zeros(size(img_ycbcr), 'like', img_ycbcr);

    for ch = 1:3
        channel = im2double(img_ycbcr(:,:,ch));
        % Compress color channels more aggressively (human eye less sensitive)
        Q_ch = Q * (1 + (ch-1)*0.5);   % Y=Q, Cb=1.5Q, Cr=2Q
        recon_ycbcr(:,:,ch) = uint8(blockDCT(channel, Q_ch) * 255);
    end

    recon_rgb = ycbcr2rgb(recon_ycbcr);
end
