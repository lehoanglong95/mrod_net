function mr_collateral_dce_reformat_dicoms(rThickness, rDistance, rSlices, rSkipDistance, rImageRotation,...
    save_dir_path, HDR, ColArt, ColCap, ColEVen, ColLVen, ColDel,...
    pColArt, pColCap, pColEVen, pColLVen, pColDel, number_of_slice, width, gfs, IMG, minTempRes, peak_phase1, peak_phase2, ...
    sap, eap, svp, evp, mvp)

%% Making Save Directory
pixel_width = 1; %HDR.PixelSpacing(1);
pixel_height = 1; %HDR.PixelSpacing(2);
ColDir = strcat('DCE_SFS', num2str(gfs), '_TCK', num2str(rThickness*pixel_height),...
    'mm_DIS', num2str(rDistance*pixel_height), 'mm_NOS', num2str(rSlices),...
    '_SD', num2str(rSkipDistance), 'mm_ROT', num2str(rImageRotation), 'deg');
ColDir_path = strcat(save_dir_path, '/', ColDir);
if isdir(ColDir_path) == 0
    mkdir(ColDir_path);
else
    queststr = ['Directory ', ColDir, ' ', 'is already existing. Do you want replace it?']; 
    createDir = questdlg(queststr,'Choose Option','Yes','No', 'No');
    if isequal(createDir, 'Yes')
        % delete directory and sub files
        rmdir(ColDir_path, 's');
        % recreate directory
        mkdir(ColDir_path);
    else
        return;
    end
end
clear pixel_width;
clear pixel_height;
clear ColDir;
clear queststr;
clear createDir;


%% Generating TTPmap
%sizeIMG = size(IMG);
%IMGHeight = sizeIMG(1);
%IMGWidth = sizeIMG(2);
%IMGPhase = sizeIMG(3);
%IMGSlice = sizeIMG(4);
%clear sizeIMG;

[~, TTPmap] = max(IMG, [], 3);
TTPmap = single((TTPmap - 1) .* minTempRes);
%TTPmap = double(zeros(IMGHeight, IMGWidth, 1, IMGSlice));
%h = waitbar(0,'TTPmap Generation');
%for z = 1:IMGSlice
%    for y = 1:IMGHeight
%        for x = 1:IMGWidth
%            max_tmp = max(squeeze(IMG(y, x, :, z)));
%            for p = 1:IMGPhase
%                if IMG(y, x, p, z) == max_tmp
%                    TTPmap(y, x, 1, z) = (p - 1) * minTempRes;
%                    break;
%                end
%            end
%            clear max_tmp;
%        end
%    end
%    status_str = strcat('TTPmap Generation (', num2str(z), '/', num2str(IMGSlice), ')');
%    waitbar( double(z) / double(IMGSlice), h, status_str );
%end
%close(h);
%clear h;
%clear x y z p;
%clear IMGHeight IMGWidth IMGSlice;

%tic;
%% Image Rotation
if rImageRotation ~= 0
    IMGsize = size(ColArt);
    IMGheight = IMGsize(1);
    IMGwidth = IMGsize(2);
    IMGslice = IMGsize(4);
    rColArt = double(zeros(IMGheight, IMGwidth, 1, IMGslice));
    rColCap = double(zeros(IMGheight, IMGwidth, 1, IMGslice));
    %rColVen = double(zeros(IMGheight, IMGwidth, 1, IMGslice));
    rColEVen = double(zeros(IMGheight, IMGwidth, 1, IMGslice));
    rColLVen = double(zeros(IMGheight, IMGwidth, 1, IMGslice));
    rColDel = double(zeros(IMGheight, IMGwidth, 1, IMGslice));
    rpColArt = double(zeros(IMGheight, IMGwidth, 1, IMGslice));
    rpColCap = double(zeros(IMGheight, IMGwidth, 1, IMGslice));
    %rpColVen = double(zeros(IMGheight, IMGwidth, 1, IMGslice));
    rpColEVen = double(zeros(IMGheight, IMGwidth, 1, IMGslice));
    rpColLVen = double(zeros(IMGheight, IMGwidth, 1, IMGslice));
    rpColDel = double(zeros(IMGheight, IMGwidth, 1, IMGslice));
    rTTPmap = double(zeros(IMGheight, IMGwidth, 1, IMGslice));
    clear IMGsize IMGheight IMGwidth IMGslice;
    h = waitbar(0,'Image Rotation');
    tic;
    for x = 1:width
        rColArt(:, x, 1, :) = imrotate(squeeze(ColArt(:, x, 1, :)), rImageRotation, 'bilinear', 'crop');
        rColCap(:, x, 1, :) = imrotate(squeeze(ColCap(:, x, 1, :)), rImageRotation, 'bilinear', 'crop');
        %rColVen(:, x, 1, :) = imrotate(squeeze(ColVen(:, x, 1, :)), rImageRotation, 'bilinear', 'crop');
        rColEVen(:, x, 1, :) = imrotate(squeeze(ColEVen(:, x, 1, :)), rImageRotation, 'bilinear', 'crop');
        rColLVen(:, x, 1, :) = imrotate(squeeze(ColLVen(:, x, 1, :)), rImageRotation, 'bilinear', 'crop');
        rColDel(:, x, 1, :) = imrotate(squeeze(ColDel(:, x, 1, :)), rImageRotation, 'bilinear', 'crop');
        rpColArt(:, x, 1, :) = imrotate(squeeze(pColArt(:, x, 1, :)), rImageRotation, 'bilinear', 'crop');
        rpColCap(:, x, 1, :) = imrotate(squeeze(pColCap(:, x, 1, :)), rImageRotation, 'bilinear', 'crop');
        %rpColVen(:, x, 1, :) = imrotate(squeeze(pColVen(:, x, 1, :)), rImageRotation, 'bilinear', 'crop');
        rpColEVen(:, x, 1, :) = imrotate(squeeze(pColEVen(:, x, 1, :)), rImageRotation, 'bilinear', 'crop');
        rpColLVen(:, x, 1, :) = imrotate(squeeze(pColLVen(:, x, 1, :)), rImageRotation, 'bilinear', 'crop');
        rpColDel(:, x, 1, :) = imrotate(squeeze(pColDel(:, x, 1, :)), rImageRotation, 'bilinear', 'crop');
        rTTPmap(:, x, 1, :) = imrotate(squeeze(TTPmap(:, x, 1, :)), rImageRotation, 'bilinear', 'crop');
        
        CurrentLoop = double(x);
        TotalLoops = double(width);
        ET = double(floor(toc));
        %timestr = mr_collateral_time_cal(ET, TotalLoops, CurrentLoop);
        status_str{2, 1} = mr_collateral_time_cal(ET, TotalLoops, CurrentLoop);
        status_str{1, 1} = strcat('Image Rotation (', num2str(x), '/', num2str(width), ')');
        waitbar( double(x) / double(width), h, status_str );
    end
    close(h);
    clear h;
    ColArt = rColArt;
    ColCap = rColCap;
    %ColVen = rColVen;
    ColEVen = rColEVen;
    ColLVen = rColLVen;
    ColDel = rColDel;
    pColArt = rpColArt;
    pColCap = rpColCap;
    %pColVen = rpColVen;
    pColEVen = rpColEVen;
    pColLVen = rpColLVen;
    pColDel = rpColDel;
    TTPmap = rTTPmap;
    clear rColArt;
    clear rColCap;
    %clear rColVen;
    clear rColEVen;
    clear rColLVen;
    clear rColDel;
    clear rpColArt;
    clear rpColCap;
    %clear rpColVen;
    clear rpColEVen;
    clear rpColLVen;
    clear rpColDel;
    clear rTTPmap;
    clear rPreIMG;
    clear status_str;
    clear CurrentLoop TotalLoops ET;
end

%disp(strcat('Image Rotation : ', num2str(toc)));


%% Generate Reformat Images
% collateral
mColArt = double(zeros(number_of_slice, width, 1, rSlices));
mColCap = double(zeros(number_of_slice, width, 1, rSlices));
%mColVen = double(zeros(number_of_slice, width, 1, rSlices));
mColEVen = double(zeros(number_of_slice, width, 1, rSlices));
mColLVen = double(zeros(number_of_slice, width, 1, rSlices));
mColDel = double(zeros(number_of_slice, width, 1, rSlices));
mpColArt = double(zeros(number_of_slice, width, 1, rSlices));
mpColCap = double(zeros(number_of_slice, width, 1, rSlices));
%mpColVen = double(zeros(number_of_slice, width, 1, rSlices));
mpColEVen = double(zeros(number_of_slice, width, 1, rSlices));
mpColLVen = double(zeros(number_of_slice, width, 1, rSlices));
mpColDel = double(zeros(number_of_slice, width, 1, rSlices));
% collateral Angio
mColArtMIP = double(zeros(number_of_slice, width, 1, rSlices));
mColCapMIP = double(zeros(number_of_slice, width, 1, rSlices));
%mColVenMIP = double(zeros(number_of_slice, width, 1, rSlices));
mColVenEMIP = double(zeros(number_of_slice, width, 1, rSlices));
mColVenLMIP = double(zeros(number_of_slice, width, 1, rSlices));
mColDelMIP = double(zeros(number_of_slice, width, 1, rSlices));
% TTPmap
mTTPmap = double(zeros(number_of_slice, width, 1, rSlices));

h = waitbar(0,'Generating Reformat Images');
tic;
for slice_loop = 1:rSlices
    slice_center = (rSkipDistance + 1) + (slice_loop - 1)*rDistance;
    slice_merge_start = slice_center - floor(rThickness / 2);
    slice_merge_end = slice_center + (rThickness - floor(rThickness / 2)) - 1;
    % collateral calculation
    for slice_merge_loop = slice_merge_start:slice_merge_end
        mColArt(:, :, 1, slice_loop) = mColArt(:, :, 1, slice_loop) + transpose(squeeze(ColArt(slice_merge_loop, :, 1, :)));
        mColCap(:, :, 1, slice_loop) = mColCap(:, :, 1, slice_loop) + transpose(squeeze(ColCap(slice_merge_loop, :, 1, :)));
        %mColVen(:, :, 1, slice_loop) = mColVen(:, :, 1, slice_loop) + transpose(squeeze(ColVen(slice_merge_loop, :, 1, :)));
        mColEVen(:, :, 1, slice_loop) = mColEVen(:, :, 1, slice_loop) + transpose(squeeze(ColEVen(slice_merge_loop, :, 1, :)));
        mColLVen(:, :, 1, slice_loop) = mColLVen(:, :, 1, slice_loop) + transpose(squeeze(ColLVen(slice_merge_loop, :, 1, :)));
        mColDel(:, :, 1, slice_loop) = mColDel(:, :, 1, slice_loop) + transpose(squeeze(ColDel(slice_merge_loop, :, 1, :)));
        mpColArt(:, :, 1, slice_loop) = mpColArt(:, :, 1, slice_loop) + transpose(squeeze(pColArt(slice_merge_loop, :, 1, :)));
        mpColCap(:, :, 1, slice_loop) = mpColCap(:, :, 1, slice_loop) + transpose(squeeze(pColCap(slice_merge_loop, :, 1, :)));
        %mpColVen(:, :, 1, slice_loop) = mpColVen(:, :, 1, slice_loop) + transpose(squeeze(pColVen(slice_merge_loop, :, 1, :)));
        mpColEVen(:, :, 1, slice_loop) = mpColEVen(:, :, 1, slice_loop) + transpose(squeeze(pColEVen(slice_merge_loop, :, 1, :)));
        mpColLVen(:, :, 1, slice_loop) = mpColLVen(:, :, 1, slice_loop) + transpose(squeeze(pColLVen(slice_merge_loop, :, 1, :)));
        mpColDel(:, :, 1, slice_loop) = mpColDel(:, :, 1, slice_loop) + transpose(squeeze(pColDel(slice_merge_loop, :, 1, :)));
        mTTPmap(:, :, 1, slice_loop) = mTTPmap(:, :, 1, slice_loop) + transpose(squeeze(TTPmap(slice_merge_loop, :, 1, :)));
    end
    avgmColArt(:, :, 1, slice_loop) = mColArt(:, :, 1, slice_loop) / rThickness;
    avgmColCap(:, :, 1, slice_loop) = mColCap(:, :, 1, slice_loop) / rThickness;
    %avgmColVen(:, :, 1, slice_loop) = mColVen(:, :, 1, slice_loop) / rThickness;
    avgmColEVen(:, :, 1, slice_loop) = mColEVen(:, :, 1, slice_loop) / rThickness;
    avgmColLVen(:, :, 1, slice_loop) = mColLVen(:, :, 1, slice_loop) / rThickness;
    avgmColDel(:, :, 1, slice_loop) = mColDel(:, :, 1, slice_loop) / rThickness;
    avgmpColArt(:, :, 1, slice_loop) = mpColArt(:, :, 1, slice_loop) / rThickness;
    avgmpColCap(:, :, 1, slice_loop) = mpColCap(:, :, 1, slice_loop) / rThickness;
    %avgmpColVen(:, :, 1, slice_loop) = mpColVen(:, :, 1, slice_loop) / rThickness;
    avgmpColEVen(:, :, 1, slice_loop) = mpColEVen(:, :, 1, slice_loop) / rThickness;
    avgmpColLVen(:, :, 1, slice_loop) = mpColLVen(:, :, 1, slice_loop) / rThickness;
    avgmpColDel(:, :, 1, slice_loop) = mpColDel(:, :, 1, slice_loop) / rThickness;
    avgmTTPmap(:, :, 1, slice_loop) = mTTPmap(:, :, 1, slice_loop) / rThickness;
    % collateral Angio calculation
    mColArtMIP(:, :, 1, slice_loop) = squeeze(max(ColArt(slice_merge_start:slice_merge_end, :, 1, :)))';
    mColCapMIP(:, :, 1, slice_loop) = squeeze(max(ColCap(slice_merge_start:slice_merge_end, :, 1, :)))';
    %mColVenMIP(:, :, 1, slice_loop) = squeeze(max(ColVen(slice_merge_start:slice_merge_end, :, 1, :)))';
    mColEVenMIP(:, :, 1, slice_loop) = squeeze(max(ColEVen(slice_merge_start:slice_merge_end, :, 1, :)))';
    mColLVenMIP(:, :, 1, slice_loop) = squeeze(max(ColLVen(slice_merge_start:slice_merge_end, :, 1, :)))';
    mColDelMIP(:, :, 1, slice_loop) = squeeze(max(ColDel(slice_merge_start:slice_merge_end, :, 1, :)))';
    
    CurrentLoop = double(slice_loop);
    TotalLoops = double(rSlices);
    ET = double(floor(toc));
    %timestr = mr_collateral_time_cal(ET, TotalLoops, CurrentLoop);
    status_str{2, 1} = mr_collateral_time_cal(ET, TotalLoops, CurrentLoop);
    status_str{1, 1} = strcat('Generating Reformat Images (', num2str(slice_loop), '/', num2str(rSlices), ')');
    waitbar( double(slice_loop) / double(rSlices), h, status_str );
end
close(h);
clear h;
clear slice_loop;
clear slice_merge_loop;
clear status_str;
clear ColArt;
clear ColCap;
%clear ColVen;
clear ColEVen;
clear ColLVen;
clear ColDel;
clear pColArt;
clear pColCap;
%clear pColVen;
clear pColEVen;
clear pColLVen;
clear pColDel;
clear mColArt;
clear mColCap;
%clear mColVen;
clear mColEVen;
clear mColLVen;
clear mColDel;
clear mpColArt;
clear mpColCap;
%clear mpColVen;
clear mpColEVen;
clear mpColLVen;
clear mpColDel;
clear mTTPmap;
clear CurrentLoop TotalLoops ET;

%figure, montage(avgmColArt, [0 100]);
%figure, montage(avgmColCap, [0 100]);
%figure, montage(avgmColVen, [0 100]);
%figure, montage(avgmColDel, [0 100]);
%figure, montage(avgmpColArt, [0 100]);
%figure, montage(avgmpColCap, [0 150]);
%figure, montage(avgmpColVen, [0 100]);
%figure, montage(avgmpColDel, [0 100]);
%imcontrast;

%disp(strcat('Reformat Generation : ', num2str(toc)));


%% Thresholding
% Collateral
avgmColArt_mask = avgmColArt > 0; avgmColArt_tmp = avgmColArt .* avgmColArt_mask;
clear avgmColArt; avgmColArt = avgmColArt_tmp; clear avgmColArt_tmp;

avgmColCap_mask = avgmColCap > 0; avgmColCap_tmp = avgmColCap .* avgmColCap_mask;
clear avgmColCap; avgmColCap = avgmColCap_tmp; clear avgmColCap_tmp;

%avgmColVen_mask = avgmColVen > 0; avgmColVen_tmp = avgmColVen .* avgmColVen_mask;
%clear avgmColVen; avgmColVen = avgmColVen_tmp; clear avgmColVen_tmp;

avgmColEVen_mask = avgmColEVen > 0; avgmColEVen_tmp = avgmColEVen .* avgmColEVen_mask;
clear avgmColEVen; avgmColEVen = avgmColEVen_tmp; clear avgmColEVen_tmp;

avgmColLVen_mask = avgmColLVen > 0; avgmColLVen_tmp = avgmColLVen .* avgmColLVen_mask;
clear avgmColLVen; avgmColLVen = avgmColLVen_tmp; clear avgmColLVen_tmp;

avgmColDel_mask = avgmColDel > 0; avgmColDel_tmp = avgmColDel .* avgmColDel_mask;
clear avgmColDel; avgmColDel = avgmColDel_tmp; clear avgmColDel_tmp;

avgmpColArt_mask = avgmpColArt > 0 & avgmpColArt < 500; avgmpColArt_tmp = avgmpColArt .* avgmpColArt_mask;
clear avgmpColArt; avgmpColArt = avgmpColArt_tmp; clear avgmpColArt_tmp;

avgmpColCap_mask = avgmpColCap > 0 & avgmpColCap < 500; avgmpColCap_tmp = avgmpColCap .* avgmpColCap_mask;
clear avgmpColCap; avgmpColCap = avgmpColCap_tmp; clear avgmpColCap_tmp;

%avgmpColVen_mask = avgmpColVen > 0 & avgmpColVen < 500; avgmpColVen_tmp = avgmpColVen .* avgmpColVen_mask;
%clear avgmpColVen; avgmpColVen = avgmpColVen_tmp; clear avgmpColVen_tmp;

avgmpColEVen_mask = avgmpColEVen > 0 & avgmpColEVen < 500; avgmpColEVen_tmp = avgmpColEVen .* avgmpColEVen_mask;
clear avgmpColEVen; avgmpColEVen = avgmpColEVen_tmp; clear avgmpColEVen_tmp;

avgmpColLVen_mask = avgmpColLVen > 0 & avgmpColLVen < 500; avgmpColLVen_tmp = avgmpColLVen .* avgmpColLVen_mask;
clear avgmpColLVen; avgmpColLVen = avgmpColLVen_tmp; clear avgmpColLVen_tmp;

avgmpColDel_mask = avgmpColDel > 0 & avgmpColDel < 500; avgmpColDel_tmp = avgmpColDel .* avgmpColDel_mask;
clear avgmpColDel; avgmpColDel = avgmpColDel_tmp; clear avgmpColDel_tmp;

% Collateral MIP
mColArtMIP_mask = mColArtMIP > 0; mColArtMIP_tmp = mColArtMIP .* mColArtMIP_mask;
clear mColArtMIP; mColArtMIP = mColArtMIP_tmp; clear mColArtMIP_tmp;

mColCapMIP_mask = mColCapMIP > 0; mColCapMIP_tmp = mColCapMIP .* mColCapMIP_mask;
clear mColCapMIP; mColCapMIP = mColCapMIP_tmp; clear mColCapMIP_tmp;

%mColVenMIP_mask = mColVenMIP > 0; mColVenMIP_tmp = mColVenMIP .* mColVenMIP_mask;
%clear mColVenMIP; mColVenMIP = mColVenMIP_tmp; clear mColVenMIP_tmp;

mColEVenMIP_mask = mColEVenMIP > 0; mColEVenMIP_tmp = mColEVenMIP .* mColEVenMIP_mask;
clear mColEVenMIP; mColEVenMIP = mColEVenMIP_tmp; clear mColEVenMIP_tmp;

mColLVenMIP_mask = mColLVenMIP > 0; mColLVenMIP_tmp = mColLVenMIP .* mColLVenMIP_mask;
clear mColLVenMIP; mColLVenMIP = mColLVenMIP_tmp; clear mColLVenMIP_tmp;

mColDelMIP_mask = mColDelMIP > 0; mColDelMIP_tmp = mColDelMIP .* mColDelMIP_mask;
clear mColDelMIP; mColDelMIP = mColDelMIP_tmp; clear mColDelMIP_tmp;

%disp(strcat('Thresholding : ', num2str(toc)));


% MIP invert
imColArtMIP = max(max(max(max(mColArtMIP)))) - mColArtMIP;
imColCapMIP = max(max(max(max(mColCapMIP)))) - mColCapMIP;
%imColVenMIP = max(max(max(max(mColVenMIP)))) - mColVenMIP;
imColEVenMIP = max(max(max(max(mColEVenMIP)))) - mColEVenMIP;
imColLVenMIP = max(max(max(max(mColLVenMIP)))) - mColLVenMIP;
imColDelMIP = max(max(max(max(mColDelMIP)))) - mColDelMIP;

%disp(strcat('Inverting : ', num2str(toc)));


%% Delaymap Generation
mixedACV = double(zeros(number_of_slice, width, 3, rSlices));
mixedCVD = double(zeros(number_of_slice, width, 3, rSlices));

ColArtmax = (max(max(max(max(avgmColArt)))));
ColCapmax = (max(max(max(max(avgmColCap)))));
%ColVenmax = (max(max(max(max(avgmColVen)))));
ColEVenmax = (max(max(max(max(avgmColEVen)))));
ColLVenmax = (max(max(max(max(avgmColLVen)))));
ColDelmax = (max(max(max(max(avgmColDel)))));

mixedACV(:, :, 1, :) = avgmColArt/ColArtmax .* avgmTTPmap;
mixedACV(:, :, 2, :) = avgmColCap/ColCapmax .* avgmTTPmap;
%mixedACV(:, :, 3, :) = avgmColVen/ColVenmax .* avgmTTPmap;
mixedACV(:, :, 3, :) = avgmColEVen/ColEVenmax .* avgmTTPmap;
maxACV1 = max(max(max(max(mixedACV(:, :, 1, :)))));
maxACV2 = max(max(max(max(mixedACV(:, :, 2, :)))));
maxACV3 = max(max(max(max(mixedACV(:, :, 3, :)))));
tmpACV(:, :, 1, :) = mixedACV(:, :, 1, :) / maxACV1;
tmpACV(:, :, 2, :) = mixedACV(:, :, 2, :) / maxACV2;
tmpACV(:, :, 3, :) = mixedACV(:, :, 3, :) / maxACV3;
clear mixedACV; mixedACV = tmpACV; clear tmpACV;

mixedCVD(:, :, 1, :) = avgmColCap/ColCapmax .* avgmTTPmap;
%mixedCVD(:, :, 2, :) = avgmColVen/ColVenmax .* avgmTTPmap;
mixedCVD(:, :, 2, :) = avgmColEVen/ColEVenmax .* avgmTTPmap;
mixedCVD(:, :, 3, :) = avgmColDel/ColDelmax .* avgmTTPmap;
maxCVD1 = max(max(max(max(mixedCVD(:, :, 1, :)))));
maxCVD2 = max(max(max(max(mixedCVD(:, :, 2, :)))));
maxCVD3 = max(max(max(max(mixedCVD(:, :, 3, :)))));
tmpCVD(:, :, 1, :) = mixedCVD(:, :, 1, :) / maxCVD1;
tmpCVD(:, :, 2, :) = mixedCVD(:, :, 2, :) / maxCVD2;
tmpCVD(:, :, 3, :) = mixedCVD(:, :, 3, :) / maxCVD3;
clear mixedCVD; mixedCVD = tmpCVD; clear tmpCVD;


MIPACV = double(zeros(number_of_slice, width, 3, rSlices));
MIPArtmax = (max(max(max(max(mColArtMIP)))));
MIPCapmax = (max(max(max(max(mColCapMIP)))));
%MIPVenmax = (max(max(max(max(mColVenMIP)))));
MIPEVenmax = (max(max(max(max(mColEVenMIP)))));
MIPLVenmax = (max(max(max(max(mColLVenMIP)))));
MIPDelmax = (max(max(max(max(mColDelMIP)))));
MIPACV(:, :, 1, :) = mColArtMIP/MIPArtmax .* avgmTTPmap;
MIPACV(:, :, 2, :) = mColCapMIP/MIPCapmax .* avgmTTPmap;
%MIPACV(:, :, 3, :) = mColVenMIP/MIPVenmax .* avgmTTPmap;
MIPACV(:, :, 3, :) = mColEVenMIP/MIPEVenmax .* avgmTTPmap;
maxMIPACV1 = max(max(max(max(MIPACV(:, :, 1, :)))));
maxMIPACV2 = max(max(max(max(MIPACV(:, :, 2, :)))));
maxMIPACV3 = max(max(max(max(MIPACV(:, :, 3, :)))));
tmpMIPACV(:, :, 1, :) = MIPACV(:, :, 1, :) / maxMIPACV1;
tmpMIPACV(:, :, 2, :) = MIPACV(:, :, 2, :) / maxMIPACV2;
tmpMIPACV(:, :, 3, :) = MIPACV(:, :, 3, :) / maxMIPACV3;
clear MIPACV; MIPACV = tmpMIPACV; clear tmpMIPACV;

MIPCVD = double(zeros(number_of_slice, width, 3, rSlices));
MIPCVD(:, :, 1, :) = mColCapMIP/MIPCapmax .* avgmTTPmap;
%MIPCVD(:, :, 2, :) = mColVenMIP/MIPVenmax .* avgmTTPmap;
MIPCVD(:, :, 2, :) = mColEVenMIP/MIPEVenmax .* avgmTTPmap;
MIPCVD(:, :, 3, :) = mColDelMIP/MIPDelmax .* avgmTTPmap;
maxMIPCVD1 = max(max(max(max(MIPCVD(:, :, 1, :)))));
maxMIPCVD2 = max(max(max(max(MIPCVD(:, :, 2, :)))));
maxMIPCVD3 = max(max(max(max(MIPCVD(:, :, 3, :)))));
tmpMIPCVD(:, :, 1, :) = MIPCVD(:, :, 1, :) / maxMIPCVD1;
tmpMIPCVD(:, :, 2, :) = MIPCVD(:, :, 2, :) / maxMIPCVD2;
tmpMIPCVD(:, :, 3, :) = MIPCVD(:, :, 3, :) / maxMIPCVD3;
clear MIPCVD; MIPCVD = tmpMIPCVD; clear tmpMIPCVD;

%disp(strcat('Delaymap Generation : ', num2str(toc)));


%% Color TTPmap
% Color Code
cR = [1.0, 0.0, 0.0]; % RED
cO = [1.0, 0.5, 0.0]; % ORANGE
cY = [1.0, 1.0, 0.0]; % YELLOW
cG = [0.0, 1.0, 0.0]; % GREEN
cB = [0.0, 0.0, 1.0]; % BLUE
cN = [0.0, 0.0, 0.5]; % NAVY
cP = [0.5, 0.0, 0.5]; % PURPLE

cTTPmapAP = double(zeros(number_of_slice, width, 3, rSlices));
cTTPmapVP = double(zeros(number_of_slice, width, 3, rSlices));
for c = 1:3
    cTTPmapAP(:, :, c, :) = avgmTTPmap;
    cTTPmapVP(:, :, c, :) = avgmTTPmap;
end
maxTTP = max(max(max(max(avgmTTPmap))));
tmp_cTTPmapAP = cTTPmapAP / maxTTP;
clear cTTPmapAP; cTTPmapAP = tmp_cTTPmapAP; clear tmp_cTTPmapAP;
tmp_cTTPmapVP = cTTPmapVP / maxTTP;
clear cTTPmapVP; cTTPmapVP = tmp_cTTPmapVP; clear tmp_cTTPmapVP;

peak_time1 = peak_phase1 * minTempRes;
%h = waitbar(0,'Generating colorTTPmap(APeak)');
%for z = 1:rSlices
%    for y = 1:number_of_slice
%        for x = 1:width
%            if avgmTTPmap(y, x, 1, z) >= 0 && avgmTTPmap(y, x, 1, z) < peak_time1 + minTempRes
%                cTTPmapAP(y, x, :, z) = cR;
%            elseif avgmTTPmap(y, x, 1, z) >= peak_time1 + minTempRes && avgmTTPmap(y, x, 1, z) < peak_time1 + 2*minTempRes
%                cTTPmapAP(y, x, :, z) = cO;
%            elseif avgmTTPmap(y, x, 1, z) >= peak_time1 + 2*minTempRes && avgmTTPmap(y, x, 1, z) < peak_time1 + 3*minTempRes
%                cTTPmapAP(y, x, :, z) = cY;
%            elseif avgmTTPmap(y, x, 1, z) >= peak_time1 + 3*minTempRes && avgmTTPmap(y, x, 1, z) < peak_time1 + 4*minTempRes
%                cTTPmapAP(y, x, :, z) = cG;
%            elseif avgmTTPmap(y, x, 1, z) >= peak_time1 + 4*minTempRes && avgmTTPmap(y, x, 1, z) < peak_time1 + 5*minTempRes
%                cTTPmapAP(y, x, :, z) = cB;
%            elseif avgmTTPmap(y, x, 1, z) >= peak_time1 + 5*minTempRes && avgmTTPmap(y, x, 1, z) < peak_time1 + 6*minTempRes
%                cTTPmapAP(y, x, :, z) = cN;
%            elseif avgmTTPmap(y, x, 1, z) >= peak_time1 + 6*minTempRes && avgmTTPmap(y, x, 1, z) < peak_time1 + 7*minTempRes
%                cTTPmapAP(y, x, :, z) = cP;
%            end
%        end
%    end
%    status_str = strcat('Generating colorTTPmap(APeak) (', num2str(z), '/', num2str(rSlices), ')');
%    waitbar( double(z) / double(rSlices), h, status_str );
%end

h = waitbar(0,'Generating colorTTPmap');
tic;
for z = 1:rSlices
    for y = 1:number_of_slice
        for x = 1:width
            if avgmTTPmap(y, x, 1, z) >= peak_time1 && avgmTTPmap(y, x, 1, z) < peak_time1 + minTempRes
                cTTPmapAP(y, x, :, z) = cR;
            elseif avgmTTPmap(y, x, 1, z) >= peak_time1 + minTempRes && avgmTTPmap(y, x, 1, z) < peak_time1 + 2*minTempRes
                cTTPmapAP(y, x, :, z) = cO;
            elseif avgmTTPmap(y, x, 1, z) >= peak_time1 + 2*minTempRes && avgmTTPmap(y, x, 1, z) < peak_time1 + 3*minTempRes
                cTTPmapAP(y, x, :, z) = cY;
            elseif avgmTTPmap(y, x, 1, z) >= peak_time1 + 3*minTempRes && avgmTTPmap(y, x, 1, z) < peak_time1 + 4*minTempRes
                cTTPmapAP(y, x, :, z) = cG;
            elseif avgmTTPmap(y, x, 1, z) >= peak_time1 + 4*minTempRes && avgmTTPmap(y, x, 1, z) < peak_time1 + 5*minTempRes
                cTTPmapAP(y, x, :, z) = cB;
            elseif avgmTTPmap(y, x, 1, z) >= peak_time1 + 5*minTempRes && avgmTTPmap(y, x, 1, z) < peak_time1 + 6*minTempRes
                cTTPmapAP(y, x, :, z) = cN;
            elseif avgmTTPmap(y, x, 1, z) >= peak_time1 + 6*minTempRes && avgmTTPmap(y, x, 1, z) < peak_time1 + 7*minTempRes
                cTTPmapAP(y, x, :, z) = cP;
            end
        end
    end
    CurrentLoop = double(z);
    TotalLoops = double(rSlices);
    ET = double(floor(toc));
    %timestr = mr_collateral_time_cal(ET, TotalLoops, CurrentLoop);
    status_str{2, 1} = mr_collateral_time_cal(ET, TotalLoops, CurrentLoop);
    status_str{1, 1} = strcat('Generating colorTTPmap (', num2str(z), '/', num2str(rSlices), ')');
    waitbar( double(z) / double(rSlices), h, status_str );
end
close(h);
clear h;
clear x y z;
clear status_str;
clear CurrentLoop TotalLoops ET;
%clear peak_time1;

%disp(strcat('start AP : ', num2str(sap), ', ', num2str(sap * minTempRes), '[sec]'));
%disp(strcat('end AP : ', num2str(eap), ', ', num2str(eap * minTempRes), '[sec]'));
%disp(strcat('start VP : ', num2str(svp), ', ', num2str(svp * minTempRes), '[sec]'));
%disp(strcat('mid VP : ', num2str(mvp), ', ', num2str(mvp * minTempRes), '[sec]'));
%disp(strcat('end VP : ', num2str(evp), ', ', num2str(evp * minTempRes), '[sec]'));

peak_time2 = peak_phase2 * minTempRes;
%peak_time2 = peak_phase1 * minTempRes;

% numimg loading
load('mr_collateral_barTXT.mat');

h = waitbar(0,'Generating colorTTPDelay');
tic;
for z = 1:rSlices
    for y = 1:number_of_slice
        for x = 1:width
            if avgmTTPmap(y, x, 1, z) == 0
                cTTPmapVP(y, x, :, z) = avgmTTPmap(y, x, 1, z) / maxTTP;
            elseif avgmTTPmap(y, x, 1, z) > 0 && avgmTTPmap(y, x, 1, z) <= eap * minTempRes
                cTTPmapVP(y, x, :, z) = cR * (avgmTTPmap(y, x, 1, z) / maxTTP) * 2; % * 1.5; % cR; % * (avgmTTPmap(y, x, 1, z) / maxTTP) * 2;
            elseif avgmTTPmap(y, x, 1, z) > eap * minTempRes && avgmTTPmap(y, x, 1, z) < svp * minTempRes
                cTTPmapVP(y, x, :, z) = cY * (avgmTTPmap(y, x, 1, z) / maxTTP) * 2; % * 1.5; % cO;
            elseif avgmTTPmap(y, x, 1, z) >= svp * minTempRes && avgmTTPmap(y, x, 1, z) <= mvp * minTempRes
                cTTPmapVP(y, x, :, z) = cG * (avgmTTPmap(y, x, 1, z) / maxTTP) * 2; % * 1.5;
            elseif avgmTTPmap(y, x, 1, z) > mvp * minTempRes && avgmTTPmap(y, x, 1, z) <= evp * minTempRes
                cTTPmapVP(y, x, :, z) = cB * (avgmTTPmap(y, x, 1, z) / maxTTP) * 2; % * 1.5;
            %elseif avgmTTPmap(y, x, 1, z) > mvp * minTempRes && avgmTTPmap(y, x, 1, z) <= evp * minTempRes
            %    cTTPmapVP(y, x, :, z) = cB * (avgmTTPmap(y, x, 1, z) / maxTTP) * 2;
            elseif avgmTTPmap(y, x, 1, z) > evp * minTempRes
                cTTPmapVP(y, x, :, z) = cP * (avgmTTPmap(y, x, 1, z) / maxTTP) * 2; %; % cN;
            %elseif avgmTTPmap(y, x, 1, z) > peak_time1 + 15 % && avgmTTPmap(y, x, 1, z) < peak_time1 + 18
            %    cTTPmapVP(y, x, :, z) = cP; % * (avgmTTPmap(y, x, 1, z) / maxTTP);
            %elseif avgmTTPmap(y, x, 1, z) >= peak_time1 + 18 % && avgmTTPmap(y, x, 1, z) < peak_time1 + 18
            %    cTTPmapVP(y, x, :, z) = cP;
            end
        end
    end
    
    cTTPmapVP((end-3-39+1):(end-3), (end-3-24+1):(end-3), :, z) = barTXT;
    
    CurrentLoop = double(z);
    TotalLoops = double(rSlices);
    ET = double(floor(toc));
    %timestr = mr_collateral_time_cal(ET, TotalLoops, CurrentLoop);
    status_str{2, 1} = mr_collateral_time_cal(ET, TotalLoops, CurrentLoop);
    status_str{1, 1} = strcat('Generating colorTTPDelay (', num2str(z), '/', num2str(rSlices), ')');
    waitbar( double(z) / double(rSlices), h, status_str );
end

%h = waitbar(0,'Generating colorTTPmap(VPeak)');
%for z = 1:rSlices
%    for y = 1:number_of_slice
%        for x = 1:width
%            if avgmTTPmap(y, x, 1, z) == 0
%                cTTPmapVP(y, x, :, z) = avgmTTPmap(y, x, 1, z) / maxTTP;
%            elseif avgmTTPmap(y, x, 1, z) > 0 && avgmTTPmap(y, x, 1, z) <= peak_time1
%                cTTPmapVP(y, x, :, z) = cR; % * avgmTTPmap(y, x, 1, z) / maxTTP; % cR; % * (avgmTTPmap(y, x, 1, z) / maxTTP) * 2;
%            elseif avgmTTPmap(y, x, 1, z) > peak_time1 && avgmTTPmap(y, x, 1, z) <= peak_time1 + 3
%                cTTPmapVP(y, x, :, z) = cO; % * (avgmTTPmap(y, x, 1, z) / maxTTP); % cO;
%            elseif avgmTTPmap(y, x, 1, z) > peak_time1 + 3 && avgmTTPmap(y, x, 1, z) <= peak_time1 + 6
%                cTTPmapVP(y, x, :, z) = cY; % * (avgmTTPmap(y, x, 1, z) / maxTTP);
%            elseif avgmTTPmap(y, x, 1, z) > peak_time1 + 6 && avgmTTPmap(y, x, 1, z) <= peak_time1 + 9
%                cTTPmapVP(y, x, :, z) = cG; % * (avgmTTPmap(y, x, 1, z) / maxTTP);
%            elseif avgmTTPmap(y, x, 1, z) > peak_time1 + 9 && avgmTTPmap(y, x, 1, z) <= peak_time1 + 12
%                cTTPmapVP(y, x, :, z) = cB; % * (avgmTTPmap(y, x, 1, z) / maxTTP);
%            elseif avgmTTPmap(y, x, 1, z) > peak_time1 + 12 && avgmTTPmap(y, x, 1, z) <= peak_time1 + 15
%                cTTPmapVP(y, x, :, z) = cN; % * (avgmTTPmap(y, x, 1, z) / maxTTP); % cN;
%            elseif avgmTTPmap(y, x, 1, z) > peak_time1 + 15 % && avgmTTPmap(y, x, 1, z) < peak_time1 + 18
%                cTTPmapVP(y, x, :, z) = cP; % * (avgmTTPmap(y, x, 1, z) / maxTTP);
%            %elseif avgmTTPmap(y, x, 1, z) >= peak_time1 + 18 % && avgmTTPmap(y, x, 1, z) < peak_time1 + 18
%            %    cTTPmapVP(y, x, :, z) = cP;
%            end
%        end
%    end
%    status_str = strcat('Generating colorTTPmap(VPeak) (', num2str(z), '/', num2str(rSlices), ')');
%    waitbar( double(z) / double(rSlices), h, status_str );
%end

%for z = 1:rSlices
%    for y = 1:number_of_slice
%        for x = 1:width
%            if avgmTTPmap(y, x, 1, z) >= 0 && avgmTTPmap(y, x, 1, z) < peak_time2 + minTempRes
%                cTTPmapVP(y, x, :, z) = cR * (avgmTTPmap(y, x, 1, z) / maxTTP) * 2;
%            elseif avgmTTPmap(y, x, 1, z) >= peak_time2 + minTempRes && avgmTTPmap(y, x, 1, z) < peak_time2 + 2*minTempRes
%                cTTPmapVP(y, x, :, z) = cO;
%            elseif avgmTTPmap(y, x, 1, z) >= peak_time2 + 2*minTempRes && avgmTTPmap(y, x, 1, z) < peak_time2 + 3*minTempRes
%                cTTPmapVP(y, x, :, z) = cY;
%            elseif avgmTTPmap(y, x, 1, z) >= peak_time2 + 3*minTempRes && avgmTTPmap(y, x, 1, z) < peak_time2 + 4*minTempRes
%                cTTPmapVP(y, x, :, z) = cG;
%            elseif avgmTTPmap(y, x, 1, z) >= peak_time2 + 4*minTempRes && avgmTTPmap(y, x, 1, z) < peak_time2 + 5*minTempRes
%                cTTPmapVP(y, x, :, z) = cB;
%            elseif avgmTTPmap(y, x, 1, z) >= peak_time2 + 5*minTempRes && avgmTTPmap(y, x, 1, z) < peak_time2 + 6*minTempRes
%                cTTPmapVP(y, x, :, z) = cN;
%            elseif avgmTTPmap(y, x, 1, z) >= peak_time2 + 6*minTempRes && avgmTTPmap(y, x, 1, z) < peak_time2 + 7*minTempRes
%                cTTPmapVP(y, x, :, z) = cP;
%            end
%        end
%    end
%    status_str = strcat('Generating colorTTPmap(VPeak) (', num2str(z), '/', num2str(rSlices), ')');
%    waitbar( double(z) / double(rSlices), h, status_str );
%end
%for z = 1:rSlices
%    for y = 1:number_of_slice
%        for x = 1:width
%            if avgmTTPmap(y, x, 1, z) >= peak_time2 && avgmTTPmap(y, x, 1, z) < peak_time2 + minTempRes
%                cTTPmapVP(y, x, :, z) = cR;
%            elseif avgmTTPmap(y, x, 1, z) >= peak_time2 + minTempRes && avgmTTPmap(y, x, 1, z) < peak_time2 + 2*minTempRes
%                cTTPmapVP(y, x, :, z) = cO;
%            elseif avgmTTPmap(y, x, 1, z) >= peak_time2 + 2*minTempRes && avgmTTPmap(y, x, 1, z) < peak_time2 + 3*minTempRes
%                cTTPmapVP(y, x, :, z) = cY;
%            elseif avgmTTPmap(y, x, 1, z) >= peak_time2 + 3*minTempRes && avgmTTPmap(y, x, 1, z) < peak_time2 + 4*minTempRes
%                cTTPmapVP(y, x, :, z) = cG;
%            elseif avgmTTPmap(y, x, 1, z) >= peak_time2 + 4*minTempRes && avgmTTPmap(y, x, 1, z) < peak_time2 + 5*minTempRes
%                cTTPmapVP(y, x, :, z) = cB;
%            elseif avgmTTPmap(y, x, 1, z) >= peak_time2 + 5*minTempRes && avgmTTPmap(y, x, 1, z) < peak_time2 + 6*minTempRes
%                cTTPmapVP(y, x, :, z) = cN;
%            elseif avgmTTPmap(y, x, 1, z) >= peak_time2 + 6*minTempRes && avgmTTPmap(y, x, 1, z) < peak_time2 + 7*minTempRes
%                cTTPmapVP(y, x, :, z) = cP;
%            end
%        end
%    end
%    status_str = strcat('Generating colorTTPmap(VPeak) (', num2str(z), '/', num2str(rSlices), ')');
%    waitbar( double(z) / double(rSlices), h, status_str );
%end
close(h);
clear h;
clear x y z;
clear status_str;
clear peak_time1;
clear CurrentLoop TotalLoops ET;

%disp(strcat('Color TTPmap : ', num2str(toc)));


%% Changing Dicom Header information...
% changed for GE
%ICD = HDR.InstanceCreationDate;
ICD = HDR.StudyDate;
SN = HDR.SeriesNumber;
SD = HDR.SeriesDescription;
new_SN = 11000 + 1; % 10000 + SN

%% Change NaN to Zero
avgmpColArt(isnan(avgmpColArt)) = 0;
avgmpColCap(isnan(avgmpColCap)) = 0;
avgmpColEVen(isnan(avgmpColEVen)) = 0;
avgmpColLVen(isnan(avgmpColLVen)) = 0;
avgmpColDel(isnan(avgmpColDel)) = 0;

%% Save Dicom Files
ColSF = 10;
pColSF = 10;
zero_length = 3;
zero_str = '_';
oriWidth = HDR.Width;
oriHeight = HDR.Height;
oriRows = HDR.Rows;
oriColumns = HDR.Columns;
oriSliceThickness = HDR.SliceThickness;
oriPixelSpacing = HDR.PixelSpacing;

%ImageComments
HDR.ImageComments = '';
% ImageOrientationPatient
row_dircos(1) = HDR.ImageOrientationPatient(1);
row_dircos(2) = HDR.ImageOrientationPatient(2);
row_dircos(3) = HDR.ImageOrientationPatient(3);
col_dircos(1) = HDR.ImageOrientationPatient(4);
col_dircos(2) = HDR.ImageOrientationPatient(5);
col_dircos(3) = HDR.ImageOrientationPatient(6);
DirCos = [row_dircos(1), row_dircos(3), row_dircos(2), col_dircos(1), col_dircos(3), col_dircos(2)]; % Transform to Transverse
HDR.ImageOrientationPatient = DirCos;
clear row_dircos;
clear col_dircos;
clear DirCos;
% SliceThickness
HDR.SliceThickness = rThickness; % * oriPixelSpacing(2);
% Rows
HDR.Rows = number_of_slice;
% Column
HDR.Column = width; %oriWidth;
% Width
HDR.Width = width; %oriWidth;
% Height
HDR.Height = number_of_slice;
% PixelSpacing
HDR.PixelSpacing = [1, 1]; %[oriPixelSpacing(1), oriSliceThickness];
tmpImagePositionPatient = HDR.ImagePositionPatient; % For calculation of ImagePositionPatient
h = waitbar(0,'Saving Dicom Files...');
color = 'jet'; % jet, hot...
tic;
for slice_loop = 1 : rSlices
    % ImagePositionPatient
    %HDR.ImagePositionPatient = [tmpImagePositionPatient(1), tmpImagePositionPatient(2), (tmpImagePositionPatient(3) - (rSkipDistance + (slice_loop - 1)*rDistance)*oriPixelSpacing(2))];
    HDR.ImagePositionPatient = [tmpImagePositionPatient(1), tmpImagePositionPatient(2), (tmpImagePositionPatient(3) - (rSkipDistance + (slice_loop - 1)*rDistance))];
    % SliceLocation
    HDR.SliceLocation = HDR.ImagePositionPatient(3);
    avgmColArt_header = HDR;
    avgmColCap_header = HDR;
    %avgmColVen_header = HDR;
    avgmColEVen_header = HDR;
    avgmColLVen_header = HDR;
    avgmColDel_header = HDR;
    avgmpColArt_header = HDR;
    avgmpColCap_header = HDR;
    %avgmpColVen_header = HDR;
    avgmpColEVen_header = HDR;
    avgmpColLVen_header = HDR;
    avgmpColDel_header = HDR;
    % color
    color_avgmColArt_header = HDR;
    color_avgmColCap_header = HDR;
    %color_avgmColVen_header = HDR;
    color_avgmColEVen_header = HDR;
    color_avgmColLVen_header = HDR;
    color_avgmColDel_header = HDR;
    color_avgmpColArt_header = HDR;
    color_avgmpColCap_header = HDR;
    %color_avgmpColVen_header = HDR;
    color_avgmpColEVen_header = HDR;
    color_avgmpColLVen_header = HDR;
    color_avgmpColDel_header = HDR;
    % collateral MIP
    mColArtMIP_header = HDR;
    mColCapMIP_header = HDR;
    %mColVenMIP_header = HDR;
    mColEVenMIP_header = HDR;
    mColLVenMIP_header = HDR;
    mColDelMIP_header = HDR;
    imColArtMIP_header = HDR;
    imColCapMIP_header = HDR;
    %imColVenMIP_header = HDR;
    imColEVenMIP_header = HDR;
    imColLVenMIP_header = HDR;
    imColDelMIP_header = HDR;
    TTPmap_header = HDR;
    cTTPmapAP_header = HDR;
    cTTPmapVP_header = HDR;
    mixedACV_header = HDR;
    mixedCVD_header = HDR;
    MIPACV_header = HDR;
    MIPCVD_header = HDR;
    for zero_loop = 1 : zero_length - length(num2str(slice_loop))
        zero_str = strcat(zero_str, '0');
    end
    
    [MIN, MAX, WW, WL] = mr_collateral_find_WWL(squeeze(avgmColArt(:, :, 1, slice_loop)*ColSF), 2);
    avgmColArt_header.SeriesDescription = 'DCE_Collateral_Arterial'; %strcat(SD,'_PWI');
    avgmColArt_header.SeriesInstanceUID = num2str(new_SN);
    avgmColArt_header.SeriesNumber = new_SN;
    avgmColArt_header.SmallestImagePixelValue = MIN; % min(min(squeeze(avgmColArt(:, :, 1, slice_loop)*ColSF)));
    avgmColArt_header.LargestImagePixelValue = MAX; % max(max(squeeze(avgmColArt(:, :, 1, slice_loop)*ColSF)));
    avgmColArt_header.WindowCenter = WL; % max(max(max(avgmColArt*ColSF)))/3/3;
    avgmColArt_header.WindowWidth = WW; % max(max(max(avgmColArt*ColSF)))/3;
    avgmColArt_header.AcquisitionNumber = slice_loop;
    avgmColArt_header.InstanceNumber = slice_loop;
    avgmColArt_path = strcat(ColDir_path, '/', avgmColArt_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(int16(avgmColArt(:, :, 1, slice_loop)*ColSF), avgmColArt_path, avgmColArt_header);
    
    [MIN, MAX, WW, WL] = mr_collateral_find_WWL(squeeze(avgmColCap(:, :, 1, slice_loop)*ColSF), 4);
    avgmColCap_header.SeriesDescription = 'DCE_Collateral_Capillary'; %strcat(SD,'_PWI');
    avgmColCap_header.SeriesInstanceUID = num2str(new_SN + 1);
    avgmColCap_header.SeriesNumber = new_SN + 1;
    avgmColCap_header.SmallestImagePixelValue = MIN; % min(min(squeeze(avgmColCap(:, :, 1, slice_loop)*ColSF)));
    avgmColCap_header.LargestImagePixelValue = MAX; % max(max(squeeze(avgmColCap(:, :, 1, slice_loop)*ColSF)));
    avgmColCap_header.WindowCenter = WL; % max(max(max(avgmColCap*ColSF)))/3/3;
    avgmColCap_header.WindowWidth = WW; % max(max(max(avgmColCap*ColSF)))/3;
    avgmColCap_header.AcquisitionNumber = slice_loop;
    avgmColCap_header.InstanceNumber = slice_loop;
    avgmColCap_path = strcat(ColDir_path, '/', avgmColCap_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(int16(avgmColCap(:, :, 1, slice_loop)*ColSF), avgmColCap_path, avgmColCap_header);
    
    %[MIN, MAX, WW, WL] = mr_collateral_find_WWL(squeeze(avgmColVen(:, :, 1, slice_loop)*ColSF));
    %avgmColVen_header.SeriesDescription = 'DCE_Collateral_Venous'; %strcat(SD,'_PWI');
    %avgmColVen_header.SeriesInstanceUID = num2str(new_SN + 2);
    %avgmColVen_header.SeriesNumber = new_SN + 2;
    %avgmColVen_header.SmallestImagePixelValue = MIN; % min(min(squeeze(avgmColVen(:, :, 1, slice_loop)*ColSF)));
    %avgmColVen_header.LargestImagePixelValue = MAX; % max(max(squeeze(avgmColVen(:, :, 1, slice_loop)*ColSF)));
    %avgmColVen_header.WindowCenter = WL; % max(max(max(avgmColVen*ColSF)))/3/3;
    %avgmColVen_header.WindowWidth = WW; % max(max(max(avgmColVen*ColSF)))/3;
    %avgmColVen_header.AcquisitionNumber = slice_loop;
    %avgmColVen_header.InstanceNumber = slice_loop;
    %avgmColVen_path = strcat(ColDir_path, '/', avgmColVen_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    %dicomwrite(int16(avgmColVen(:, :, 1, slice_loop)*ColSF), avgmColVen_path, avgmColVen_header);
    
    [MIN, MAX, WW, WL] = mr_collateral_find_WWL(squeeze(avgmColEVen(:, :, 1, slice_loop)*ColSF), 8);
    avgmColEVen_header.SeriesDescription = 'DCE_Collateral_Early_Venous'; %strcat(SD,'_PWI');
    avgmColEVen_header.SeriesInstanceUID = num2str(new_SN + 2);
    avgmColEVen_header.SeriesNumber = new_SN + 2;
    avgmColEVen_header.SmallestImagePixelValue = MIN; % min(min(squeeze(avgmColVen(:, :, 1, slice_loop)*ColSF)));
    avgmColEVen_header.LargestImagePixelValue = MAX; % max(max(squeeze(avgmColVen(:, :, 1, slice_loop)*ColSF)));
    avgmColEVen_header.WindowCenter = WL; % max(max(max(avgmColVen*ColSF)))/3/3;
    avgmColEVen_header.WindowWidth = WW; % max(max(max(avgmColVen*ColSF)))/3;
    avgmColEVen_header.AcquisitionNumber = slice_loop;
    avgmColEVen_header.InstanceNumber = slice_loop;
    avgmColEVen_path = strcat(ColDir_path, '/', avgmColEVen_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(int16(avgmColEVen(:, :, 1, slice_loop)*ColSF), avgmColEVen_path, avgmColEVen_header);
    
    [MIN, MAX, WW, WL] = mr_collateral_find_WWL(squeeze(avgmColLVen(:, :, 1, slice_loop)*ColSF), 10);
    avgmColLVen_header.SeriesDescription = 'DCE_Collateral_Late_Venous'; %strcat(SD,'_PWI');
    avgmColLVen_header.SeriesInstanceUID = num2str(new_SN + 3);
    avgmColLVen_header.SeriesNumber = new_SN + 3;
    avgmColLVen_header.SmallestImagePixelValue = MIN; % min(min(squeeze(avgmColVen(:, :, 1, slice_loop)*ColSF)));
    avgmColLVen_header.LargestImagePixelValue = MAX; % max(max(squeeze(avgmColVen(:, :, 1, slice_loop)*ColSF)));
    avgmColLVen_header.WindowCenter = WL; % max(max(max(avgmColVen*ColSF)))/3/3;
    avgmColLVen_header.WindowWidth = WW; % max(max(max(avgmColVen*ColSF)))/3;
    avgmColLVen_header.AcquisitionNumber = slice_loop;
    avgmColLVen_header.InstanceNumber = slice_loop;
    avgmColLVen_path = strcat(ColDir_path, '/', avgmColLVen_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(int16(avgmColLVen(:, :, 1, slice_loop)*ColSF), avgmColLVen_path, avgmColLVen_header);
    
    [MIN, MAX, WW, WL] = mr_collateral_find_WWL(squeeze(avgmColDel(:, :, 1, slice_loop)*ColSF), 10);
    avgmColDel_header.SeriesDescription = 'DCE_Collateral_Delay'; %strcat(SD,'_PWI');
    avgmColDel_header.SeriesInstanceUID = num2str(new_SN + 4);
    avgmColDel_header.SeriesNumber = new_SN + 4;
    avgmColDel_header.SmallestImagePixelValue = MIN; % min(min(squeeze(avgmColDel(:, :, 1, slice_loop)*ColSF)));
    avgmColDel_header.LargestImagePixelValue = MAX; % max(max(squeeze(avgmColDel(:, :, 1, slice_loop)*ColSF)));
    avgmColDel_header.WindowCenter = WL; % max(max(max(avgmColDel*ColSF)))/3/3;
    avgmColDel_header.WindowWidth = WW; % max(max(max(avgmColDel*ColSF)))/3;
    avgmColDel_header.AcquisitionNumber = slice_loop;
    avgmColDel_header.InstanceNumber = slice_loop;
    avgmColDel_path = strcat(ColDir_path, '/', avgmColDel_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(int16(avgmColDel(:, :, 1, slice_loop)*ColSF), avgmColDel_path, avgmColDel_header);
    
    % color Collateral
    color_avgmColArt_img = mr_collateral_gen_color_image(squeeze(avgmColArt(:, :, 1, slice_loop)*ColSF), HDR.Width, HDR.Height, 2);
    color_avgmColArt_header.SeriesDescription = 'DCE_color_Collateral_Arterial'; %strcat(SD,'_PWI');
    color_avgmColArt_header.SeriesInstanceUID = num2str(new_SN + 5);
    color_avgmColArt_header.SeriesNumber = new_SN + 5;
    color_avgmColArt_header.SmallestImagePixelValue = 0;
    color_avgmColArt_header.LargestImagePixelValue = 255;
    color_avgmColArt_header.WindowCenter = 128;
    color_avgmColArt_header.WindowWidth = 255;
    color_avgmColArt_header.AcquisitionNumber = slice_loop;
    color_avgmColArt_header.InstanceNumber = slice_loop;
    color_avgmColArt_header.SamplesPerPixel = 3;
    color_avgmColArt_header.PhotometricInterpretation = 'RGB';
    color_avgmColArt_header.BitsAllocated = 8;
    color_avgmColArt_header.BitsStored = 8;
    color_avgmColArt_header.HighBit = 7;
    color_avgmColArt_header.PixelRepresentation = 0;
    color_avgmColArt_header.AcquisitionNumber = slice_loop;
    color_avgmColArt_header.InstanceNumber = slice_loop;
    color_avgmColArt_path = strcat(ColDir_path, '/', color_avgmColArt_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(uint8(color_avgmColArt_img), color_avgmColArt_path, color_avgmColArt_header);
    
    color_avgmColCap_img = mr_collateral_gen_color_image(squeeze(avgmColCap(:, :, 1, slice_loop)*ColSF), HDR.Width, HDR.Height, 4);
    color_avgmColCap_header.SeriesDescription = 'DCE_color_Collateral_Capillary'; %strcat(SD,'_PWI');
    color_avgmColCap_header.SeriesInstanceUID = num2str(new_SN + 6);
    color_avgmColCap_header.SeriesNumber = new_SN + 6;
    color_avgmColCap_header.SmallestImagePixelValue = 0;
    color_avgmColCap_header.LargestImagePixelValue = 255;
    color_avgmColCap_header.WindowCenter = 128;
    color_avgmColCap_header.WindowWidth = 255;
    color_avgmColCap_header.AcquisitionNumber = slice_loop;
    color_avgmColCap_header.InstanceNumber = slice_loop;
    color_avgmColCap_header.SamplesPerPixel = 3;
    color_avgmColCap_header.PhotometricInterpretation = 'RGB';
    color_avgmColCap_header.BitsAllocated = 8;
    color_avgmColCap_header.BitsStored = 8;
    color_avgmColCap_header.HighBit = 7;
    color_avgmColCap_header.PixelRepresentation = 0;
    color_avgmColCap_header.AcquisitionNumber = slice_loop;
    color_avgmColCap_header.InstanceNumber = slice_loop;
    color_avgmColCap_path = strcat(ColDir_path, '/', color_avgmColCap_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(uint8(color_avgmColCap_img), color_avgmColCap_path, color_avgmColCap_header);
    
    %color_avgmColVen_img = mr_collateral_gen_color_image(squeeze(avgmColVen(:, :, 1, slice_loop)*ColSF), HDR.Width, HDR.Height);
    %color_avgmColVen_header.SeriesDescription = 'DCE_color_Collateral_Venous'; %strcat(SD,'_PWI');
    %color_avgmColVen_header.SeriesInstanceUID = num2str(new_SN + 6);
    %color_avgmColVen_header.SeriesNumber = new_SN + 6;
    %color_avgmColVen_header.SmallestImagePixelValue = 0;
    %color_avgmColVen_header.LargestImagePixelValue = 255;
    %color_avgmColVen_header.WindowCenter = 128;
    %color_avgmColVen_header.WindowWidth = 255;
    %color_avgmColVen_header.AcquisitionNumber = slice_loop;
    %color_avgmColVen_header.InstanceNumber = slice_loop;
    %color_avgmColVen_header.SamplesPerPixel = 3;
    %color_avgmColVen_header.PhotometricInterpretation = 'RGB';
    %color_avgmColVen_header.BitsAllocated = 8;
    %color_avgmColVen_header.BitsStored = 8;
    %color_avgmColVen_header.HighBit = 7;
    %color_avgmColVen_header.PixelRepresentation = 0;
    %color_avgmColVen_header.AcquisitionNumber = slice_loop;
    %color_avgmColVen_header.InstanceNumber = slice_loop;
    %color_avgmColVen_path = strcat(ColDir_path, '/', color_avgmColVen_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    %dicomwrite(uint8(color_avgmColVen_img), color_avgmColVen_path, color_avgmColVen_header);
    
    color_avgmColEVen_img = mr_collateral_gen_color_image(squeeze(avgmColEVen(:, :, 1, slice_loop)*ColSF), HDR.Width, HDR.Height, 8);
    color_avgmColEVen_header.SeriesDescription = 'DCE_color_Collateral_Early_Venous'; %strcat(SD,'_PWI');
    color_avgmColEVen_header.SeriesInstanceUID = num2str(new_SN + 7);
    color_avgmColEVen_header.SeriesNumber = new_SN + 7;
    color_avgmColEVen_header.SmallestImagePixelValue = 0;
    color_avgmColEVen_header.LargestImagePixelValue = 255;
    color_avgmColEVen_header.WindowCenter = 128;
    color_avgmColEVen_header.WindowWidth = 255;
    color_avgmColEVen_header.AcquisitionNumber = slice_loop;
    color_avgmColEVen_header.InstanceNumber = slice_loop;
    color_avgmColEVen_header.SamplesPerPixel = 3;
    color_avgmColEVen_header.PhotometricInterpretation = 'RGB';
    color_avgmColEVen_header.BitsAllocated = 8;
    color_avgmColEVen_header.BitsStored = 8;
    color_avgmColEVen_header.HighBit = 7;
    color_avgmColEVen_header.PixelRepresentation = 0;
    color_avgmColEVen_header.AcquisitionNumber = slice_loop;
    color_avgmColEVen_header.InstanceNumber = slice_loop;
    color_avgmColEVen_path = strcat(ColDir_path, '/', color_avgmColEVen_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(uint8(color_avgmColEVen_img), color_avgmColEVen_path, color_avgmColEVen_header);
    
    color_avgmColLVen_img = mr_collateral_gen_color_image(squeeze(avgmColLVen(:, :, 1, slice_loop)*ColSF), HDR.Width, HDR.Height, 10);
    color_avgmColLVen_header.SeriesDescription = 'DCE_color_Collateral_Late_Venous'; %strcat(SD,'_PWI');
    color_avgmColLVen_header.SeriesInstanceUID = num2str(new_SN + 8);
    color_avgmColLVen_header.SeriesNumber = new_SN + 8;
    color_avgmColLVen_header.SmallestImagePixelValue = 0;
    color_avgmColLVen_header.LargestImagePixelValue = 255;
    color_avgmColLVen_header.WindowCenter = 128;
    color_avgmColLVen_header.WindowWidth = 255;
    color_avgmColLVen_header.AcquisitionNumber = slice_loop;
    color_avgmColLVen_header.InstanceNumber = slice_loop;
    color_avgmColLVen_header.SamplesPerPixel = 3;
    color_avgmColLVen_header.PhotometricInterpretation = 'RGB';
    color_avgmColLVen_header.BitsAllocated = 8;
    color_avgmColLVen_header.BitsStored = 8;
    color_avgmColLVen_header.HighBit = 7;
    color_avgmColLVen_header.PixelRepresentation = 0;
    color_avgmColLVen_header.AcquisitionNumber = slice_loop;
    color_avgmColLVen_header.InstanceNumber = slice_loop;
    color_avgmColLVen_path = strcat(ColDir_path, '/', color_avgmColLVen_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(uint8(color_avgmColLVen_img), color_avgmColLVen_path, color_avgmColLVen_header);
    
    color_avgmColDel_img = mr_collateral_gen_color_image(squeeze(avgmColDel(:, :, 1, slice_loop)*ColSF), HDR.Width, HDR.Height, 10);
    color_avgmColDel_header.SeriesDescription = 'DCE_color_Collateral_Delay'; %strcat(SD,'_PWI');
    color_avgmColDel_header.SeriesInstanceUID = num2str(new_SN + 9);
    color_avgmColDel_header.SeriesNumber = new_SN + 9;
    color_avgmColDel_header.SmallestImagePixelValue = 0;
    color_avgmColDel_header.LargestImagePixelValue = 255;
    color_avgmColDel_header.WindowCenter = 128;
    color_avgmColDel_header.WindowWidth = 255;
    color_avgmColDel_header.AcquisitionNumber = slice_loop;
    color_avgmColDel_header.InstanceNumber = slice_loop;
    color_avgmColDel_header.SamplesPerPixel = 3;
    color_avgmColDel_header.PhotometricInterpretation = 'RGB';
    color_avgmColDel_header.BitsAllocated = 8;
    color_avgmColDel_header.BitsStored = 8;
    color_avgmColDel_header.HighBit = 7;
    color_avgmColDel_header.PixelRepresentation = 0;
    color_avgmColDel_header.AcquisitionNumber = slice_loop;
    color_avgmColDel_header.InstanceNumber = slice_loop;
    color_avgmColDel_path = strcat(ColDir_path, '/', color_avgmColDel_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(uint8(color_avgmColDel_img), color_avgmColDel_path, color_avgmColDel_header);
    
    % percentage Collateral
    [MIN, MAX, WW, WL] = mr_collateral_find_WWL(squeeze(avgmpColArt(:, :, 1, slice_loop)*pColSF), 2);
    avgmpColArt_header.SeriesDescription = 'DCE_percentage_Collateral_Arterial'; %strcat(SD, '_pPWI');
    avgmpColArt_header.SeriesInstanceUID = num2str(new_SN + 10);
    avgmpColArt_header.SeriesNumber = new_SN + 10;
    avgmpColArt_header.SmallestImagePixelValue = MIN; % min(min(squeeze(avgmpColArt(:, :, 1, slice_loop)*pColSF)));
    avgmpColArt_header.LargestImagePixelValue = MAX; % max(max(squeeze(avgmpColArt(:, :, 1, slice_loop)*pColSF)));
    avgmpColArt_header.WindowCenter = WL; % 50*pColSF; %max(max(max(avgmpColArt*pColSF)))/2/2;
    avgmpColArt_header.WindowWidth = WW; % 100*pColSF; %max(max(max(avgmpColArt*pColSF)))/2;
    avgmpColArt_header.AcquisitionNumber = slice_loop;
    avgmpColArt_header.InstanceNumber = slice_loop;
    avgmpColArt_path = strcat(ColDir_path, '/', avgmpColArt_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(int16(avgmpColArt(:, :, 1, slice_loop)*pColSF), avgmpColArt_path, avgmpColArt_header);
    
    [MIN, MAX, WW, WL] = mr_collateral_find_WWL(squeeze(avgmpColCap(:, :, 1, slice_loop)*pColSF), 4);
    avgmpColCap_header.SeriesDescription = 'DCE_percentage_Collateral_Capillary'; %strcat(SD, '_pPWI');
    avgmpColCap_header.SeriesInstanceUID = num2str(new_SN + 11);
    avgmpColCap_header.SeriesNumber = new_SN + 11;
    avgmpColCap_header.SmallestImagePixelValue = MIN; % min(min(squeeze(avgmpColCap(:, :, 1, slice_loop)*pColSF)));
    avgmpColCap_header.LargestImagePixelValue = MAX; % max(max(squeeze(avgmpColCap(:, :, 1, slice_loop)*pColSF)));
    avgmpColCap_header.WindowCenter = WL; % 50*pColSF; %max(max(max(avgmpColCap*pColSF)))/2/2;
    avgmpColCap_header.WindowWidth = WW; % 100*pColSF; %max(max(max(avgmpColCap*pColSF)))/2;
    avgmpColCap_header.AcquisitionNumber = slice_loop;
    avgmpColCap_header.InstanceNumber = slice_loop;
    avgmpColCap_path = strcat(ColDir_path, '/', avgmpColCap_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(int16(avgmpColCap(:, :, 1, slice_loop)*pColSF), avgmpColCap_path, avgmpColCap_header);
    
    %[MIN, MAX, WW, WL] = mr_collateral_find_WWL(squeeze(avgmpColVen(:, :, 1, slice_loop)*pColSF));
    %avgmpColVen_header.SeriesDescription = 'DCE_percentage_Collateral_Venous'; %strcat(SD, '_pPWI');
    %avgmpColVen_header.SeriesInstanceUID = num2str(new_SN + 10);
    %avgmpColVen_header.SeriesNumber = new_SN + 10;
    %avgmpColVen_header.SmallestImagePixelValue = MIN; % min(min(squeeze(avgmpColVen(:, :, 1, slice_loop)*pColSF)));
    %avgmpColVen_header.LargestImagePixelValue = MAX; % max(max(squeeze(avgmpColVen(:, :, 1, slice_loop)*pColSF)));
    %avgmpColVen_header.WindowCenter = WL; % 50*pColSF; %max(max(max(avgmpColVen*pColSF)))/2/2;
    %avgmpColVen_header.WindowWidth = WW; % 100*pColSF; %max(max(max(avgmpColVen*pColSF)))/2;
    %avgmpColVen_header.AcquisitionNumber = slice_loop;
    %avgmpColVen_header.InstanceNumber = slice_loop;
    %avgmpColVen_path = strcat(ColDir_path, '/', avgmpColVen_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    %dicomwrite(int16(avgmpColVen(:, :, 1, slice_loop)*pColSF), avgmpColVen_path, avgmpColVen_header);
    
    [MIN, MAX, WW, WL] = mr_collateral_find_WWL(squeeze(avgmpColEVen(:, :, 1, slice_loop)*pColSF), 8);
    avgmpColEVen_header.SeriesDescription = 'DCE_percentage_Collateral_Early_Venous'; %strcat(SD, '_pPWI');
    avgmpColEVen_header.SeriesInstanceUID = num2str(new_SN + 12);
    avgmpColEVen_header.SeriesNumber = new_SN + 12;
    avgmpColEVen_header.SmallestImagePixelValue = MIN; % min(min(squeeze(avgmpColVen(:, :, 1, slice_loop)*pColSF)));
    avgmpColEVen_header.LargestImagePixelValue = MAX; % max(max(squeeze(avgmpColVen(:, :, 1, slice_loop)*pColSF)));
    avgmpColEVen_header.WindowCenter = WL; % 50*pColSF; %max(max(max(avgmpColVen*pColSF)))/2/2;
    avgmpColEVen_header.WindowWidth = WW; % 100*pColSF; %max(max(max(avgmpColVen*pColSF)))/2;
    avgmpColEVen_header.AcquisitionNumber = slice_loop;
    avgmpColEVen_header.InstanceNumber = slice_loop;
    avgmpColEVen_path = strcat(ColDir_path, '/', avgmpColEVen_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(int16(avgmpColEVen(:, :, 1, slice_loop)*pColSF), avgmpColEVen_path, avgmpColEVen_header);
    
    [MIN, MAX, WW, WL] = mr_collateral_find_WWL(squeeze(avgmpColLVen(:, :, 1, slice_loop)*pColSF), 10);
    avgmpColLVen_header.SeriesDescription = 'DCE_percentage_Collateral_Late_Venous'; %strcat(SD, '_pPWI');
    avgmpColLVen_header.SeriesInstanceUID = num2str(new_SN + 13);
    avgmpColLVen_header.SeriesNumber = new_SN + 13;
    avgmpColLVen_header.SmallestImagePixelValue = MIN; % min(min(squeeze(avgmpColVen(:, :, 1, slice_loop)*pColSF)));
    avgmpColLVen_header.LargestImagePixelValue = MAX; % max(max(squeeze(avgmpColVen(:, :, 1, slice_loop)*pColSF)));
    avgmpColLVen_header.WindowCenter = WL; % 50*pColSF; %max(max(max(avgmpColVen*pColSF)))/2/2;
    avgmpColLVen_header.WindowWidth = WW; % 100*pColSF; %max(max(max(avgmpColVen*pColSF)))/2;
    avgmpColLVen_header.AcquisitionNumber = slice_loop;
    avgmpColLVen_header.InstanceNumber = slice_loop;
    avgmpColLVen_path = strcat(ColDir_path, '/', avgmpColLVen_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(int16(avgmpColLVen(:, :, 1, slice_loop)*pColSF), avgmpColLVen_path, avgmpColLVen_header);
    
    [MIN, MAX, WW, WL] = mr_collateral_find_WWL(squeeze(avgmpColDel(:, :, 1, slice_loop)*pColSF), 10);
    avgmpColDel_header.SeriesDescription = 'DCE_percentage_Collateral_Delay'; %strcat(SD, '_pPWI');
    avgmpColDel_header.SeriesInstanceUID = num2str(new_SN + 14);
    avgmpColDel_header.SeriesNumber = new_SN + 14;
    avgmpColDel_header.SmallestImagePixelValue = MIN; % min(min(squeeze(avgmpColDel(:, :, 1, slice_loop)*pColSF)));
    avgmpColDel_header.LargestImagePixelValue = MAX; % max(max(squeeze(avgmpColDel(:, :, 1, slice_loop)*pColSF)));
    avgmpColDel_header.WindowCenter = WL; % 50*pColSF; %max(max(max(avgmpColDel*pColSF)))/2/2;
    avgmpColDel_header.WindowWidth = WW; % 100*pColSF; %max(max(max(avgmpColDel*pColSF)))/2;
    avgmpColDel_header.AcquisitionNumber = slice_loop;
    avgmpColDel_header.InstanceNumber = slice_loop;
    avgmpColDel_path = strcat(ColDir_path, '/', avgmpColDel_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(int16(avgmpColDel(:, :, 1, slice_loop)*pColSF), avgmpColDel_path, avgmpColDel_header);
    
    % color Collateral
    color_avgmpColArt_img = mr_collateral_gen_color_image(squeeze(avgmpColArt(:, :, 1, slice_loop)*pColSF), HDR.Width, HDR.Height, 2);
    color_avgmpColArt_header.SeriesDescription = 'DCE_color_percentage_Collateral_Arterial'; %strcat(SD,'_PWI');
    color_avgmpColArt_header.SeriesInstanceUID = num2str(new_SN + 15);
    color_avgmpColArt_header.SeriesNumber = new_SN + 15;
    color_avgmpColArt_header.SmallestImagePixelValue = 0;
    color_avgmpColArt_header.LargestImagePixelValue = 255;
    color_avgmpColArt_header.WindowCenter = 128;
    color_avgmpColArt_header.WindowWidth = 255;
    color_avgmpColArt_header.AcquisitionNumber = slice_loop;
    color_avgmpColArt_header.InstanceNumber = slice_loop;
    color_avgmpColArt_header.SamplesPerPixel = 3;
    color_avgmpColArt_header.PhotometricInterpretation = 'RGB';
    color_avgmpColArt_header.BitsAllocated = 8;
    color_avgmpColArt_header.BitsStored = 8;
    color_avgmpColArt_header.HighBit = 7;
    color_avgmpColArt_header.PixelRepresentation = 0;
    color_avgmpColArt_header.AcquisitionNumber = slice_loop;
    color_avgmpColArt_header.InstanceNumber = slice_loop;
    color_avgmpColArt_path = strcat(ColDir_path, '/', color_avgmpColArt_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(uint8(color_avgmpColArt_img), color_avgmpColArt_path, color_avgmpColArt_header);
    
    color_avgmpColCap_img = mr_collateral_gen_color_image(squeeze(avgmpColCap(:, :, 1, slice_loop)*pColSF), HDR.Width, HDR.Height, 4);
    color_avgmpColCap_header.SeriesDescription = 'DCE_color_percentage_Collateral_Capillary'; %strcat(SD,'_PWI');
    color_avgmpColCap_header.SeriesInstanceUID = num2str(new_SN + 16);
    color_avgmpColCap_header.SeriesNumber = new_SN + 16;
    color_avgmpColCap_header.SmallestImagePixelValue = 0;
    color_avgmpColCap_header.LargestImagePixelValue = 255;
    color_avgmpColCap_header.WindowCenter = 128;
    color_avgmpColCap_header.WindowWidth = 255;
    color_avgmpColCap_header.AcquisitionNumber = slice_loop;
    color_avgmpColCap_header.InstanceNumber = slice_loop;
    color_avgmpColCap_header.SamplesPerPixel = 3;
    color_avgmpColCap_header.PhotometricInterpretation = 'RGB';
    color_avgmpColCap_header.BitsAllocated = 8;
    color_avgmpColCap_header.BitsStored = 8;
    color_avgmpColCap_header.HighBit = 7;
    color_avgmpColCap_header.PixelRepresentation = 0;
    color_avgmpColCap_header.AcquisitionNumber = slice_loop;
    color_avgmpColCap_header.InstanceNumber = slice_loop;
    color_avgmpColCap_path = strcat(ColDir_path, '/', color_avgmpColCap_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(uint8(color_avgmpColCap_img), color_avgmpColCap_path, color_avgmpColCap_header);
    
    %color_avgmpColVen_img = mr_collateral_gen_color_image(squeeze(avgmpColVen(:, :, 1, slice_loop)*pColSF), HDR.Width, HDR.Height);
    %color_avgmpColVen_header.SeriesDescription = 'DCE_color_percentage_Collateral_Venous'; %strcat(SD,'_PWI');
    %color_avgmpColVen_header.SeriesInstanceUID = num2str(new_SN + 14);
    %color_avgmpColVen_header.SeriesNumber = new_SN + 14;
    %color_avgmpColVen_header.SmallestImagePixelValue = 0;
    %color_avgmpColVen_header.LargestImagePixelValue = 255;
    %color_avgmpColVen_header.WindowCenter = 128;
    %color_avgmpColVen_header.WindowWidth = 255;
    %color_avgmpColVen_header.AcquisitionNumber = slice_loop;
    %color_avgmpColVen_header.InstanceNumber = slice_loop;
    %color_avgmpColVen_header.SamplesPerPixel = 3;
    %color_avgmpColVen_header.PhotometricInterpretation = 'RGB';
    %color_avgmpColVen_header.BitsAllocated = 8;
    %color_avgmpColVen_header.BitsStored = 8;
    %color_avgmpColVen_header.HighBit = 7;
    %color_avgmpColVen_header.PixelRepresentation = 0;
    %color_avgmpColVen_header.AcquisitionNumber = slice_loop;
    %color_avgmpColVen_header.InstanceNumber = slice_loop;
    %color_avgmpColVen_path = strcat(ColDir_path, '/', color_avgmpColVen_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    %dicomwrite(uint8(color_avgmpColVen_img), color_avgmpColVen_path, color_avgmpColVen_header);
    
    color_avgmpColEVen_img = mr_collateral_gen_color_image(squeeze(avgmpColEVen(:, :, 1, slice_loop)*pColSF), HDR.Width, HDR.Height, 8);
    color_avgmpColEVen_header.SeriesDescription = 'DCE_color_percentage_Collateral_Early_Venous'; %strcat(SD,'_PWI');
    color_avgmpColEVen_header.SeriesInstanceUID = num2str(new_SN + 17);
    color_avgmpColEVen_header.SeriesNumber = new_SN + 17;
    color_avgmpColEVen_header.SmallestImagePixelValue = 0;
    color_avgmpColEVen_header.LargestImagePixelValue = 255;
    color_avgmpColEVen_header.WindowCenter = 128;
    color_avgmpColEVen_header.WindowWidth = 255;
    color_avgmpColEVen_header.AcquisitionNumber = slice_loop;
    color_avgmpColEVen_header.InstanceNumber = slice_loop;
    color_avgmpColEVen_header.SamplesPerPixel = 3;
    color_avgmpColEVen_header.PhotometricInterpretation = 'RGB';
    color_avgmpColEVen_header.BitsAllocated = 8;
    color_avgmpColEVen_header.BitsStored = 8;
    color_avgmpColEVen_header.HighBit = 7;
    color_avgmpColEVen_header.PixelRepresentation = 0;
    color_avgmpColEVen_header.AcquisitionNumber = slice_loop;
    color_avgmpColEVen_header.InstanceNumber = slice_loop;
    color_avgmpColEVen_path = strcat(ColDir_path, '/', color_avgmpColEVen_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(uint8(color_avgmpColEVen_img), color_avgmpColEVen_path, color_avgmpColEVen_header);
    
    color_avgmpColLVen_img = mr_collateral_gen_color_image(squeeze(avgmpColLVen(:, :, 1, slice_loop)*pColSF), HDR.Width, HDR.Height, 10);
    color_avgmpColLVen_header.SeriesDescription = 'DCE_color_percentage_Collateral_Late_Venous'; %strcat(SD,'_PWI');
    color_avgmpColLVen_header.SeriesInstanceUID = num2str(new_SN + 18);
    color_avgmpColLVen_header.SeriesNumber = new_SN + 18;
    color_avgmpColLVen_header.SmallestImagePixelValue = 0;
    color_avgmpColLVen_header.LargestImagePixelValue = 255;
    color_avgmpColLVen_header.WindowCenter = 128;
    color_avgmpColLVen_header.WindowWidth = 255;
    color_avgmpColLVen_header.AcquisitionNumber = slice_loop;
    color_avgmpColLVen_header.InstanceNumber = slice_loop;
    color_avgmpColLVen_header.SamplesPerPixel = 3;
    color_avgmpColLVen_header.PhotometricInterpretation = 'RGB';
    color_avgmpColLVen_header.BitsAllocated = 8;
    color_avgmpColLVen_header.BitsStored = 8;
    color_avgmpColLVen_header.HighBit = 7;
    color_avgmpColLVen_header.PixelRepresentation = 0;
    color_avgmpColLVen_header.AcquisitionNumber = slice_loop;
    color_avgmpColLVen_header.InstanceNumber = slice_loop;
    color_avgmpColLVen_path = strcat(ColDir_path, '/', color_avgmpColLVen_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(uint8(color_avgmpColLVen_img), color_avgmpColLVen_path, color_avgmpColLVen_header);
    
    color_avgmpColDel_img = mr_collateral_gen_color_image(squeeze(avgmpColDel(:, :, 1, slice_loop)*pColSF), HDR.Width, HDR.Height, 10);
    color_avgmpColDel_header.SeriesDescription = 'DCE_color_percentage_Collateral_Delay'; %strcat(SD,'_PWI');
    color_avgmpColDel_header.SeriesInstanceUID = num2str(new_SN + 19);
    color_avgmpColDel_header.SeriesNumber = new_SN + 19;
    color_avgmpColDel_header.SmallestImagePixelValue = 0;
    color_avgmpColDel_header.LargestImagePixelValue = 255;
    color_avgmpColDel_header.WindowCenter = 128;
    color_avgmpColDel_header.WindowWidth = 255;
    color_avgmpColDel_header.AcquisitionNumber = slice_loop;
    color_avgmpColDel_header.InstanceNumber = slice_loop;
    color_avgmpColDel_header.SamplesPerPixel = 3;
    color_avgmpColDel_header.PhotometricInterpretation = 'RGB';
    color_avgmpColDel_header.BitsAllocated = 8;
    color_avgmpColDel_header.BitsStored = 8;
    color_avgmpColDel_header.HighBit = 7;
    color_avgmpColDel_header.PixelRepresentation = 0;
    color_avgmpColDel_header.AcquisitionNumber = slice_loop;
    color_avgmpColDel_header.InstanceNumber = slice_loop;
    color_avgmpColDel_path = strcat(ColDir_path, '/', color_avgmpColDel_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(uint8(color_avgmpColDel_img), color_avgmpColDel_path, color_avgmpColDel_header);
    
    % collateral MIP
    [MIN, MAX, WW, WL] = mr_collateral_find_WWL(squeeze(mColArtMIP(:, :, 1, slice_loop)*ColSF), 2);
    mColArtMIP_header.SeriesDescription = 'DCE_Collateral_Arterial_MIP'; %strcat(SD,'_PWI');
    mColArtMIP_header.SeriesInstanceUID = num2str(new_SN + 20);
    mColArtMIP_header.SeriesNumber = new_SN + 20;
    mColArtMIP_header.SmallestImagePixelValue = MIN; % min(min(squeeze(mColArtMIP(:, :, 1, slice_loop)*ColSF)));
    mColArtMIP_header.LargestImagePixelValue = MAX; % max(max(squeeze(mColArtMIP(:, :, 1, slice_loop)*ColSF)));
    mColArtMIP_header.WindowCenter = WL; % max(max(max(mColArtMIP*ColSF)))/2/3;
    mColArtMIP_header.WindowWidth = WW; % max(max(max(mColArtMIP*ColSF)))/2;
    mColArtMIP_header.AcquisitionNumber = slice_loop;
    mColArtMIP_header.InstanceNumber = slice_loop;
    mColArtMIP_path = strcat(ColDir_path, '/', mColArtMIP_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(int16(mColArtMIP(:, :, 1, slice_loop)*ColSF), mColArtMIP_path, mColArtMIP_header);
    
    [MIN, MAX, WW, WL] = mr_collateral_find_WWL(squeeze(mColCapMIP(:, :, 1, slice_loop)*ColSF), 4);
    mColCapMIP_header.SeriesDescription = 'DCE_Collateral_Capillary_MIP'; %strcat(SD,'_PWI');
    mColCapMIP_header.SeriesInstanceUID = num2str(new_SN + 21);
    mColCapMIP_header.SeriesNumber = new_SN + 21;
    mColCapMIP_header.SmallestImagePixelValue = MIN; % min(min(squeeze(mColCapMIP(:, :, 1, slice_loop)*ColSF)));
    mColCapMIP_header.LargestImagePixelValue = MAX; % max(max(squeeze(mColCapMIP(:, :, 1, slice_loop)*ColSF)));
    mColCapMIP_header.WindowCenter = WL; % max(max(max(mColCapMIP*ColSF)))/2/3;
    mColCapMIP_header.WindowWidth = WW; % max(max(max(mColCapMIP*ColSF)))/2;
    mColCapMIP_header.AcquisitionNumber = slice_loop;
    mColCapMIP_header.InstanceNumber = slice_loop;
    mColCapMIP_path = strcat(ColDir_path, '/', mColCapMIP_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(int16(mColCapMIP(:, :, 1, slice_loop)*ColSF), mColCapMIP_path, mColCapMIP_header);
    
    %[MIN, MAX, WW, WL] = mr_collateral_find_WWL(squeeze(mColVenMIP(:, :, 1, slice_loop)*ColSF));
    %mColVenMIP_header.SeriesDescription = 'DCE_Collateral_Venous_MIP'; %strcat(SD,'_PWI');
    %mColVenMIP_header.SeriesInstanceUID = num2str(new_SN + 18);
    %mColVenMIP_header.SeriesNumber = new_SN + 18;
    %mColVenMIP_header.SmallestImagePixelValue = MIN; % min(min(squeeze(mColVenMIP(:, :, 1, slice_loop)*ColSF)));
    %mColVenMIP_header.LargestImagePixelValue = MAX; % max(max(squeeze(mColVenMIP(:, :, 1, slice_loop)*ColSF)));
    %mColVenMIP_header.WindowCenter = WL; % max(max(max(mColVenMIP*ColSF)))/2/3;
    %mColVenMIP_header.WindowWidth = WW; % max(max(max(mColVenMIP*ColSF)))/2;
    %mColVenMIP_header.AcquisitionNumber = slice_loop;
    %mColVenMIP_header.InstanceNumber = slice_loop;
    %mColVenMIP_path = strcat(ColDir_path, '/', mColVenMIP_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    %dicomwrite(int16(mColVenMIP(:, :, 1, slice_loop)*ColSF), mColVenMIP_path, mColVenMIP_header);
    
    [MIN, MAX, WW, WL] = mr_collateral_find_WWL(squeeze(mColEVenMIP(:, :, 1, slice_loop)*ColSF), 8);
    mColLVenMIP_header.SeriesDescription = 'DCE_Collateral_Early_Venous_MIP'; %strcat(SD,'_PWI');
    mColLVenMIP_header.SeriesInstanceUID = num2str(new_SN + 22);
    mColLVenMIP_header.SeriesNumber = new_SN + 22;
    mColLVenMIP_header.SmallestImagePixelValue = MIN; % min(min(squeeze(mColVenMIP(:, :, 1, slice_loop)*ColSF)));
    mColLVenMIP_header.LargestImagePixelValue = MAX; % max(max(squeeze(mColVenMIP(:, :, 1, slice_loop)*ColSF)));
    mColLVenMIP_header.WindowCenter = WL; % max(max(max(mColVenMIP*ColSF)))/2/3;
    mColLVenMIP_header.WindowWidth = WW; % max(max(max(mColVenMIP*ColSF)))/2;
    mColLVenMIP_header.AcquisitionNumber = slice_loop;
    mColLVenMIP_header.InstanceNumber = slice_loop;
    mColLVenMIP_path = strcat(ColDir_path, '/', mColLVenMIP_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(int16(mColLVenMIP(:, :, 1, slice_loop)*ColSF), mColLVenMIP_path, mColLVenMIP_header);
    
    [MIN, MAX, WW, WL] = mr_collateral_find_WWL(squeeze(mColLVenMIP(:, :, 1, slice_loop)*ColSF), 10);
    mColLVenMIP_header.SeriesDescription = 'DCE_Collateral_Late_Venous_MIP'; %strcat(SD,'_PWI');
    mColLVenMIP_header.SeriesInstanceUID = num2str(new_SN + 23);
    mColLVenMIP_header.SeriesNumber = new_SN + 23;
    mColLVenMIP_header.SmallestImagePixelValue = MIN; % min(min(squeeze(mColVenMIP(:, :, 1, slice_loop)*ColSF)));
    mColLVenMIP_header.LargestImagePixelValue = MAX; % max(max(squeeze(mColVenMIP(:, :, 1, slice_loop)*ColSF)));
    mColLVenMIP_header.WindowCenter = WL; % max(max(max(mColVenMIP*ColSF)))/2/3;
    mColLVenMIP_header.WindowWidth = WW; % max(max(max(mColVenMIP*ColSF)))/2;
    mColLVenMIP_header.AcquisitionNumber = slice_loop;
    mColLVenMIP_header.InstanceNumber = slice_loop;
    mColLVenMIP_path = strcat(ColDir_path, '/', mColLVenMIP_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(int16(mColLVenMIP(:, :, 1, slice_loop)*ColSF), mColLVenMIP_path, mColLVenMIP_header);
    
    [MIN, MAX, WW, WL] = mr_collateral_find_WWL(squeeze(mColDelMIP(:, :, 1, slice_loop)*ColSF), 10);
    mColDelMIP_header.SeriesDescription = 'DCE_Collateral_Delay_MIP'; %strcat(SD,'_PWI');
    mColDelMIP_header.SeriesInstanceUID = num2str(new_SN + 24);
    mColDelMIP_header.SeriesNumber = new_SN + 24;
    mColDenMIP_header.SmallestImagePixelValue = MIN; % min(min(squeeze(mColDelMIP(:, :, 1, slice_loop)*ColSF)));
    mColDelMIP_header.LargestImagePixelValue = MAX; % max(max(squeeze(mColDelMIP(:, :, 1, slice_loop)*ColSF)));
    mColDelMIP_header.WindowCenter = WL; % max(max(max(mColDelMIP*ColSF)))/2/3;
    mColDelMIP_header.WindowWidth = WW; % max(max(max(mColDelMIP*ColSF)))/2;
    mColDelMIP_header.AcquisitionNumber = slice_loop;
    mColDelMIP_header.InstanceNumber = slice_loop;
    mColDelMIP_path = strcat(ColDir_path, '/', mColDelMIP_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(int16(mColDelMIP(:, :, 1, slice_loop)*ColSF), mColDelMIP_path, mColDelMIP_header);
    
    % invert MIP
    [MIN, MAX, WW, WL] = mr_collateral_find_WWL(squeeze(imColArtMIP(:, :, 1, slice_loop)*ColSF), 2);
    imColArtMIP_header.SeriesDescription = 'DCE_Collateral_Arterial_invertMIP'; %strcat(SD,'_PWI');
    imColArtMIP_header.SeriesInstanceUID = num2str(new_SN + 25);
    imColArtMIP_header.SeriesNumber = new_SN + 25;
    imColArtMIP_header.SmallestImagePixelValue = MIN; % min(min(squeeze(imColArtMIP(:, :, 1, slice_loop)*ColSF)));
    imColArtMIP_header.LargestImagePixelValue = MAX; % max(max(squeeze(imColArtMIP(:, :, 1, slice_loop)*ColSF)));
    imColArtMIP_header.WindowCenter = WL; % max(max(max(imColArtMIP*ColSF))) - max(max(max(imColArtMIP*ColSF)))/2/3;
    imColArtMIP_header.WindowWidth = WW; % max(max(max(imColArtMIP*ColSF)))/2;
    imColArtMIP_header.AcquisitionNumber = slice_loop;
    imColArtMIP_header.InstanceNumber = slice_loop;
    imColArtMIP_path = strcat(ColDir_path, '/', imColArtMIP_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(int16(imColArtMIP(:, :, 1, slice_loop)*ColSF), imColArtMIP_path, imColArtMIP_header);
    
    [MIN, MAX, WW, WL] = mr_collateral_find_WWL(squeeze(imColCapMIP(:, :, 1, slice_loop)*ColSF), 4);
    imColCapMIP_header.SeriesDescription = 'DCE_Collateral_Capillary_invertMIP'; %strcat(SD,'_PWI');
    imColCapMIP_header.SeriesInstanceUID = num2str(new_SN + 26);
    imColCapMIP_header.SeriesNumber = new_SN + 26;
    imColCapMIP_header.SmallestImagePixelValue = MIN; % min(min(squeeze(imColCapMIP(:, :, 1, slice_loop)*ColSF)));
    imColCapMIP_header.LargestImagePixelValue = MAX; % max(max(squeeze(imColCapMIP(:, :, 1, slice_loop)*ColSF)));
    imColCapMIP_header.WindowCenter = WL; % max(max(max(imColCapMIP*ColSF))) - max(max(max(imColCapMIP*ColSF)))/2/2;
    imColCapMIP_header.WindowWidth = WW; % max(max(max(imColCapMIP*ColSF)))/2;
    imColCapMIP_header.AcquisitionNumber = slice_loop;
    imColCapMIP_header.InstanceNumber = slice_loop;
    imColCapMIP_path = strcat(ColDir_path, '/', imColCapMIP_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(int16(imColCapMIP(:, :, 1, slice_loop)*ColSF), imColCapMIP_path, imColCapMIP_header);
    
    %[MIN, MAX, WW, WL] = mr_collateral_find_WWL(squeeze(imColVenMIP(:, :, 1, slice_loop)*ColSF));
    %imColVenMIP_header.SeriesDescription = 'DCE_Collateral_Venous_invertMIP'; %strcat(SD,'_PWI');
    %imColVenMIP_header.SeriesInstanceUID = num2str(new_SN + 22);
    %imColVenMIP_header.SeriesNumber = new_SN + 22;
    %imColVenMIP_header.SmallestImagePixelValue = MIN; % min(min(squeeze(imColVenMIP(:, :, 1, slice_loop)*ColSF)));
    %imColVenMIP_header.LargestImagePixelValue = MAX; % max(max(squeeze(imColVenMIP(:, :, 1, slice_loop)*ColSF)));
    %imColVenMIP_header.WindowCenter = MAX - WL; % max(max(max(imColVenMIP*ColSF))) - max(max(max(imColVenMIP*ColSF)))/2/2;
    %imColVenMIP_header.WindowWidth = WW; % max(max(max(imColVenMIP*ColSF)))/2;
    %imColVenMIP_header.AcquisitionNumber = slice_loop;
    %imColVenMIP_header.InstanceNumber = slice_loop;
    %imColVenMIP_path = strcat(ColDir_path, '/', imColVenMIP_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    %dicomwrite(int16(imColVenMIP(:, :, 1, slice_loop)*ColSF), imColVenMIP_path, imColVenMIP_header);
    
    [MIN, MAX, WW, WL] = mr_collateral_find_WWL(squeeze(imColEVenMIP(:, :, 1, slice_loop)*ColSF), 8);
    imColEVenMIP_header.SeriesDescription = 'DCE_Collateral_Early_Venous_invertMIP'; %strcat(SD,'_PWI');
    imColEVenMIP_header.SeriesInstanceUID = num2str(new_SN + 27);
    imColEVenMIP_header.SeriesNumber = new_SN + 27;
    imColEVenMIP_header.SmallestImagePixelValue = MIN; % min(min(squeeze(imColVenMIP(:, :, 1, slice_loop)*ColSF)));
    imColEVenMIP_header.LargestImagePixelValue = MAX; % max(max(squeeze(imColVenMIP(:, :, 1, slice_loop)*ColSF)));
    imColEVenMIP_header.WindowCenter = WL; % max(max(max(imColVenMIP*ColSF))) - max(max(max(imColVenMIP*ColSF)))/2/2;
    imColEVenMIP_header.WindowWidth = WW; % max(max(max(imColVenMIP*ColSF)))/2;
    imColEVenMIP_header.AcquisitionNumber = slice_loop;
    imColEVenMIP_header.InstanceNumber = slice_loop;
    imColEVenMIP_path = strcat(ColDir_path, '/', imColEVenMIP_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(int16(imColEVenMIP(:, :, 1, slice_loop)*ColSF), imColEVenMIP_path, imColEVenMIP_header);
    
    [MIN, MAX, WW, WL] = mr_collateral_find_WWL(squeeze(imColLVenMIP(:, :, 1, slice_loop)*ColSF), 10);
    imColLVenMIP_header.SeriesDescription = 'DCE_Collateral_Late_Venous_invertMIP'; %strcat(SD,'_PWI');
    imColLVenMIP_header.SeriesInstanceUID = num2str(new_SN + 28);
    imColLVenMIP_header.SeriesNumber = new_SN + 28;
    imColLVenMIP_header.SmallestImagePixelValue = MIN; % min(min(squeeze(imColVenMIP(:, :, 1, slice_loop)*ColSF)));
    imColLVenMIP_header.LargestImagePixelValue = MAX; % max(max(squeeze(imColVenMIP(:, :, 1, slice_loop)*ColSF)));
    imColLVenMIP_header.WindowCenter = WL; % max(max(max(imColVenMIP*ColSF))) - max(max(max(imColVenMIP*ColSF)))/2/2;
    imColLVenMIP_header.WindowWidth = WW; % max(max(max(imColVenMIP*ColSF)))/2;
    imColLVenMIP_header.AcquisitionNumber = slice_loop;
    imColLVenMIP_header.InstanceNumber = slice_loop;
    imColLVenMIP_path = strcat(ColDir_path, '/', imColLVenMIP_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(int16(imColLVenMIP(:, :, 1, slice_loop)*ColSF), imColLVenMIP_path, imColLVenMIP_header);
    
    [MIN, MAX, WW, WL] = mr_collateral_find_WWL(squeeze(imColDelMIP(:, :, 1, slice_loop)*ColSF), 10);
    imColDelMIP_header.SeriesDescription = 'DCE_Collateral_Delay_invertMIP'; %strcat(SD,'_PWI');
    imColDelMIP_header.SeriesInstanceUID = num2str(new_SN + 29);
    imColDelMIP_header.SeriesNumber = new_SN + 29;
    imColDelMIP_header.SmallestImagePixelValue = MIN; % min(min(squeeze(imColDelMIP(:, :, 1, slice_loop)*ColSF)));
    imColDelMIP_header.LargestImagePixelValue = MAX; % max(max(squeeze(imColDelMIP(:, :, 1, slice_loop)*ColSF)));
    imColDelMIP_header.WindowCenter = WL; % max(max(max(imColDelMIP*ColSF))) - max(max(max(imColDelMIP*ColSF)))/2/2;
    imColDelMIP_header.WindowWidth = WW; % max(max(max(imColDelMIP*ColSF)))/2;
    imColDelMIP_header.AcquisitionNumber = slice_loop;
    imColDelMIP_header.InstanceNumber = slice_loop;
    imColDelMIP_path = strcat(ColDir_path, '/', imColDelMIP_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(int16(imColDelMIP(:, :, 1, slice_loop)*ColSF), imColDelMIP_path, imColDelMIP_header);
    
    TTPmap_header.SeriesDescription = 'DCE_TTPmap'; %strcat(SD,'_PWI');
    TTPmap_header.SeriesInstanceUID = num2str(new_SN + 30);
    TTPmap_header.SeriesNumber = new_SN + 30;
    TTPmap_header.SmallestImagePixelValue = min(min(squeeze(avgmTTPmap(:, :, 1, slice_loop))));
    TTPmap_header.LargestImagePixelValue = max(max(squeeze(avgmTTPmap(:, :, 1, slice_loop))));
    TTPmap_header.WindowCenter = max(max(max(avgmTTPmap)))/2 * 10;
    TTPmap_header.WindowWidth = max(max(max(avgmTTPmap)))/2 * 10;
    TTPmap_header.AcquisitionNumber = slice_loop;
    TTPmap_header.InstanceNumber = slice_loop;
    TTPmap_path = strcat(ColDir_path, '/', TTPmap_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(int16(avgmTTPmap(:, :, 1, slice_loop)*10), TTPmap_path, TTPmap_header);
    
    cTTP_img = mr_collateral_gen_color_image(squeeze(avgmTTPmap(:, :, 1, slice_loop)), HDR.Width, HDR.Height, 15);
    cTTPmapAP_header.SeriesDescription = 'DCE_colorTTPmap'; %strcat(SD,'_PWI');
    cTTPmapAP_header.SeriesInstanceUID = num2str(new_SN + 31);
    cTTPmapAP_header.SeriesNumber = new_SN + 31;
    cTTPmapAP_header.LargestImagePixelValue = 255;
    cTTPmapAP_header.WindowCenter = 128;
    cTTPmapAP_header.WindowWidth = 255;
    cTTPmapAP_header.AcquisitionNumber = slice_loop;
    cTTPmapAP_header.InstanceNumber = slice_loop;
    cTTPmapAP_header.SamplesPerPixel = 3;
    cTTPmapAP_header.PhotometricInterpretation = 'RGB';
    cTTPmapAP_header.BitsAllocated = 8;
    cTTPmapAP_header.BitsStored = 8;
    cTTPmapAP_header.HighBit = 7;
    cTTPmapAP_header.PixelRepresentation = 0;
    cTTPmapAP_header.AcquisitionNumber = slice_loop;
    cTTPmapAP_header.InstanceNumber = slice_loop;
    cTTPmapAP_path = strcat(ColDir_path, '/', cTTPmapAP_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(uint8(cTTP_img), cTTPmapAP_path, cTTPmapAP_header);
    
    %cTTPmapVP_header.SeriesDescription = 'DCE_Collateral_colorTTPmap(VPeak)'; %strcat(SD,'_PWI');
    cTTPmapVP_header.SeriesDescription = 'DCE_colorTTPDelay'; %strcat(SD,'_PWI');
    cTTPmapVP_header.SeriesInstanceUID = num2str(new_SN + 32);
    cTTPmapVP_header.SeriesNumber = new_SN + 32;
    cTTPmapVP_header.LargestImagePixelValue = 255;
    cTTPmapVP_header.WindowCenter = 128;
    cTTPmapVP_header.WindowWidth = 255;
    cTTPmapVP_header.AcquisitionNumber = slice_loop;
    cTTPmapVP_header.InstanceNumber = slice_loop;
    cTTPmapVP_header.SamplesPerPixel = 3;
    cTTPmapVP_header.PhotometricInterpretation = 'RGB';
    cTTPmapVP_header.BitsAllocated = 8;
    cTTPmapVP_header.BitsStored = 8;
    cTTPmapVP_header.HighBit = 7;
    cTTPmapVP_header.PixelRepresentation = 0;
    cTTPmapVP_path = strcat(ColDir_path, '/', cTTPmapVP_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(uint8(cTTPmapVP(:, :, :, slice_loop)*255), cTTPmapVP_path, cTTPmapVP_header);
    
    mixedACV_header.SeriesDescription = 'DCE_Collateral_mixedACV'; %strcat(SD,'_PWI');
    mixedACV_header.SeriesInstanceUID = num2str(new_SN + 33);
    mixedACV_header.SeriesNumber = new_SN + 33;
    mixedACV_header.LargestImagePixelValue = 255;
    mixedACV_header.WindowCenter = 50;
    mixedACV_header.WindowWidth = 100;
    mixedACV_header.AcquisitionNumber = slice_loop;
    mixedACV_header.InstanceNumber = slice_loop;
    mixedACV_header.SamplesPerPixel = 3;
    mixedACV_header.PhotometricInterpretation = 'RGB';
    mixedACV_header.BitsAllocated = 8;
    mixedACV_header.BitsStored = 8;
    mixedACV_header.HighBit = 7;
    mixedACV_header.PixelRepresentation = 0;
    mixedACV_path = strcat(ColDir_path, '/', mixedACV_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(uint8(mixedACV(:, :, :, slice_loop)*1000), mixedACV_path, mixedACV_header);
    
    mixedCVD_header.SeriesDescription = 'DCE_Collateral_mixedCVD'; %strcat(SD,'_PWI');
    mixedCVD_header.SeriesInstanceUID = num2str(new_SN + 34);
    mixedCVD_header.SeriesNumber = new_SN + 34;
    mixedCVD_header.LargestImagePixelValue = 255;
    mixedCVD_header.WindowCenter = 50;
    mixedCVD_header.WindowWidth = 100;
    mixedCVD_header.AcquisitionNumber = slice_loop;
    mixedCVD_header.InstanceNumber = slice_loop;
    mixedCVD_header.SamplesPerPixel = 3;
    mixedCVD_header.PhotometricInterpretation = 'RGB';
    mixedCVD_header.BitsAllocated = 8;
    mixedCVD_header.BitsStored = 8;
    mixedCVD_header.HighBit = 7;
    mixedCVD_header.PixelRepresentation = 0;
    mixedCVD_path = strcat(ColDir_path, '/', mixedCVD_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(uint8(mixedCVD(:, :, :, slice_loop)*1000), mixedCVD_path, mixedCVD_header);
    
    MIPACV_header.SeriesDescription = 'DCE_Collateral_MIPACV'; %strcat(SD,'_PWI');
    MIPACV_header.SeriesInstanceUID = num2str(new_SN + 35);
    MIPACV_header.SeriesNumber = new_SN + 35;
    MIPACV_header.LargestImagePixelValue = 255;
    MIPACV_header.WindowCenter = 50;
    MIPACV_header.WindowWidth = 100;
    MIPACV_header.AcquisitionNumber = slice_loop;
    MIPACV_header.InstanceNumber = slice_loop;
    MIPACV_header.SamplesPerPixel = 3;
    MIPACV_header.PhotometricInterpretation = 'RGB';
    MIPACV_header.BitsAllocated = 8;
    MIPACV_header.BitsStored = 8;
    MIPACV_header.HighBit = 7;
    MIPACV_header.PixelRepresentation = 0;
    MIPACV_path = strcat(ColDir_path, '/', MIPACV_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(uint8(MIPACV(:, :, :, slice_loop)*255), MIPACV_path, MIPACV_header);
    
    MIPCVD_header.SeriesDescription = 'DCE_Collateral_MIPCVD'; %strcat(SD,'_PWI');
    MIPCVD_header.SeriesInstanceUID = num2str(new_SN + 36);
    MIPCVD_header.SeriesNumber = new_SN + 36;
    MIPCVD_header.LargestImagePixelValue = 255;
    MIPCVD_header.WindowCenter = 50;
    MIPCVD_header.WindowWidth = 100;
    MIPCVD_header.AcquisitionNumber = slice_loop;
    MIPCVD_header.InstanceNumber = slice_loop;
    MIPCVD_header.SamplesPerPixel = 3;
    MIPCVD_header.PhotometricInterpretation = 'RGB';
    MIPCVD_header.BitsAllocated = 8;
    MIPCVD_header.BitsStored = 8;
    MIPCVD_header.HighBit = 7;
    MIPCVD_header.PixelRepresentation = 0;
    MIPCVD_path = strcat(ColDir_path, '/', MIPCVD_header.SeriesDescription, zero_str, num2str(slice_loop), '.dcm');
    dicomwrite(uint8(MIPCVD(:, :, :, slice_loop)*255), MIPCVD_path, MIPCVD_header);

    zero_str = '_'; % initialize
    
    CurrentLoop = double(slice_loop);
    TotalLoops = double(rSlices);
    ET = double(floor(toc));
    %timestr = mr_collateral_time_cal(ET, TotalLoops, CurrentLoop);
    status_str{2, 1} = mr_collateral_time_cal(ET, TotalLoops, CurrentLoop);
    status_str{1, 1} = strcat('Saving Dicom Files... (', num2str(slice_loop), '/', num2str(rSlices), ')');
    waitbar( double(slice_loop) / double(rSlices), h, status_str );
end
close(h);
clear h;
clear slice_loop;
clear status_str;
clear zero_str;
clear CurrentLoop TotalLoops ET;

%disp(strcat('Dicom Saving : ', num2str(toc)));


end

