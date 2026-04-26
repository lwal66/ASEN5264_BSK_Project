close all
clear
clc


evalData = readtable("outdir\eval_outputs\ppo_eval_steps.csv");
evalData.Properties.VariableNames = {'episode', 'step', 'd_ts', 'required_retasking', ...
    'action', 'selected_target', 'imaged_target','reward', 'total_reward', 'terminated', 'truncated', ...
    'target_0_name','target_1_name','target_2_name','target_3_name','target_4_name', ...
    'battery_fraction', 'storage_fraction', ...
    'target_0_priority', 'target_0_target_angle', 'target_0_oppOpen', 'target_0_oppClose', ...
    'target_1_priority', 'target_1_target_angle', 'target_1_oppOpen', 'target_1_oppClose', ...
    'target_2_priority', 'target_2_target_angle', 'target_2_oppOpen', 'target_2_oppClose', ...
    'target_3_priority', 'target_3_target_angle', 'target_3_oppOpen', 'target_3_oppClose', ...
    'target_4_priority', 'target_4_target_angle', 'target_4_oppOpen', 'target_4_oppClose', ...
    'eclipse_start', 'eclipse_end'};

evalData.target_0_oppOpen = 300.*evalData.target_0_oppOpen;
evalData.target_0_oppClose = 300.*evalData.target_0_oppClose;
evalData.target_1_oppOpen = 300.*evalData.target_0_oppOpen;
evalData.target_1_oppClose = 300.*evalData.target_0_oppClose;
evalData.target_2_oppOpen = 300.*evalData.target_0_oppOpen;
evalData.target_2_oppClose = 300.*evalData.target_0_oppClose;
evalData.target_3_oppOpen = 300.*evalData.target_0_oppOpen;
evalData.target_3_oppClose = 300.*evalData.target_0_oppClose;
evalData.target_4_oppOpen = 300.*evalData.target_0_oppOpen;
evalData.target_4_oppClose = 300.*evalData.target_0_oppClose;

episodes = unique(evalData.episode);

for eIdx = 1:1%length(episodes)
    disp("Episode: " + num2str(eIdx))
    episodeNum = episodes(eIdx);
    thisEpisodeData = evalData(evalData.episode == episodeNum, :);
    thisEpisodeData.stepStartTime = [0; cumsum(thisEpisodeData.d_ts(1:end-1))];
    %thisEpisodeData.collectedCities = {};

    target0_doesOppOpen = thisEpisodeData.target_0_oppOpen < thisEpisodeData.d_ts;
    target1_doesOppOpen = thisEpisodeData.target_1_oppOpen < thisEpisodeData.d_ts;
    target2_doesOppOpen = thisEpisodeData.target_2_oppOpen < thisEpisodeData.d_ts;
    target3_doesOppOpen = thisEpisodeData.target_3_oppOpen < thisEpisodeData.d_ts;
    target4_doesOppOpen = thisEpisodeData.target_4_oppOpen < thisEpisodeData.d_ts;
    
    numOppsOpen = sum([target0_doesOppOpen, target1_doesOppOpen, target2_doesOppOpen, target3_doesOppOpen, target4_doesOppOpen], 2);

    [numOpps, ~, idx] = unique(numOppsOpen);   % unique values
    counts = histcounts(idx, 1:(numel(numOpps)+1));
    perc = 100*counts / length(numOppsOpen);

    selectedTargetOpenings = zeros(height(thisEpisodeData), 1);
    for p = 1:height(thisEpisodeData)
        act = thisEpisodeData.action(p);
        if (act > 0)
            selectedTargetOpenings(p) = thisEpisodeData.(['target_',num2str( act - 1),'_oppOpen'])(p);
        end
    end
    figure()

    thisEpisodeDatOnlyImaging = thisEpisodeData(thisEpisodeData.action > 0, :);
    imagedNotSelected = ~strcmp(thisEpisodeDatOnlyImaging.selected_target, thisEpisodeDatOnlyImaging.imaged_target);
    scatter(thisEpisodeDatOnlyImaging.step(imagedNotSelected), selectedTargetOpenings(imagedNotSelected), 'x')
    hold on
    scatter(thisEpisodeDatOnlyImaging.step(~imagedNotSelected), selectedTargetOpenings(~imagedNotSelected), 'p')
    hold on
    
    disp("  Num cities in view")
    for optIdx = 1:length(numOpps)
        disp("    " + num2str(numOpps(optIdx)) + " cities in view " + num2str(perc(optIdx)) + "% of the steps")
    end
    disp("    Other numbers of cities in view had 0%")
    
    %%  Opportunities Loop
    for n = 0:5
    
        oppsList = thisEpisodeData(numOppsOpen == n, :);
        numThisOpps = height(oppsList);
        disp("  " + num2str(numThisOpps) + " Steps with " + num2str(n) +" View Opportunities")
    
        chargeActions = sum(oppsList.action == 0);
        imageActions = sum(oppsList.action ~= 0);
    
  
        disp("    Charge " + num2str(100*chargeActions/numThisOpps) + "% of the time")
        disp("    Image " + num2str(100*imageActions/numThisOpps) + "% of the time")

        %% NO OPPS
        if n == 0
            imageAction = oppsList(oppsList.action ~= 0, : );
            x300SecSteps = imageAction(imageAction.d_ts > 299, :);
            disp("    Of the " + height(imageAction) + " steps where we images with no opportunity, " + height(x300SecSteps) + " of them used the max step time")
            disp("    Of those " + height(x300SecSteps) + ", " + sum(x300SecSteps.reward == 0) + " got 0 reward")
            disp("    THESE WERE WAITING ACTIONS")
            xLT300SecSteps = imageAction(imageAction.d_ts < 299, :);
            disp("    Of the " + height(imageAction) + " steps where we images with no opportunity, " + height(xLT300SecSteps) + " of them use less that max time")
            disp("    Of those " + height(xLT300SecSteps) + ", " + sum(xLT300SecSteps.reward > 0) + " got some reward")
            disp("    THEY GOT REWARD EVEN WITH NO OPPORTUNITY")
        

        elseif n == 1
            action1 = oppsList(oppsList.action == 1, :);
            disp("    " + num2str(height(action1)) + " of the " + num2str(numThisOpps) + " Steps used action of 1")
            notAction1 = oppsList(oppsList.action ~= 1, :);
            noReward = notAction1(notAction1.reward == 0, :);
            disp("    Of the " + num2str(height(notAction1)) + " stpes that didnt use action 1, " + num2str(height(noReward)) + " got no reward")
            disp("    THESE WERE WAITING ACTIONS")
            someReward = notAction1(notAction1.reward ~= 0, :);
            disp("    The other " + num2str(height(someReward)) + " stpes that didnt use action 1, got some reward")
            disp("    EITHER A MISMATCH OF ACTIONS TO OBSERVATIONS, OR THEY GOT REWARD WITH NO OPPORTUNITY")
        elseif (n == 5)
            fiveOppsList = thisEpisodeData(numOppsOpen == 5, :);
            figure()
            subplot(3, 1, 1)
            scatter(fiveOppsList.stepStartTime, fiveOppsList.action)
            ylabel("ACTION  NUMBER 0-charge, 1-5 image")
            subplot(3, 1, 2)
            scatter(fiveOppsList.stepStartTime, fiveOppsList.reward)
            ylabel("REWARD")

            scores = [fiveOppsList.target_0_priority, ...
                fiveOppsList.target_1_priority, ...
                fiveOppsList.target_2_priority, ...
                fiveOppsList.target_3_priority, ...
                fiveOppsList.target_4_priority];
            [~, order] = sort(scores, 2, 'descend');
            [~, ranks] = sort(order, 2);
            chosen_rank = zeros(height(fiveOppsList), 1);
            subplot(3, 1, 3)

            for i = 1:height(fiveOppsList)        
                a = fiveOppsList.action(i);
                if (a == 0) % Charge
                    chosen_rank(i) = -1;
                else % Image
        
                    chosen_rank(i) = ranks(i, a);
                end
            end

            scatter(fiveOppsList.stepStartTime, chosen_rank)
            ylabel("RANKED ACTION BY PRIORITY")
            xlabel("TIME [s]")
            firstWindow = fiveOppsList(fiveOppsList.stepStartTime > 1000 & fiveOppsList.stepStartTime < 2000, :);
            targetPossibleCities = [firstWindow.target_0_name, firstWindow.target_1_name, firstWindow.target_2_name, firstWindow.target_3_name, firstWindow.target_4_name];
            possibltCitiesCovered = unique(targetPossibleCities);
            achievedCities = {};
            for m = 1:height(firstWindow)
                if (fiveOppsList.reward(m) > 0)
                    achievedCities{end+1} = targetPossibleCities{m, fiveOppsList.action(m)};
                end
            end
    

            b=1;
        end
    end

    %% WHat happens where there are a lot of opps?
    


end
