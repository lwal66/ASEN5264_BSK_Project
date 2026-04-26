
evalData = readtable("outdir\cities\eval_outputs\eval_steps.csv");
evalData.Properties.VariableNames = {'episode', 'step', 'd_ts', 'required_retasking', ...
    'action', 'reward', 'total_reward', 'terminated', 'truncated', ...
    'battery_charge_fraction', 'storage_level_fraction', ...
    'target_0_priority', 'target_0_target_angle', 'target_0_oppOpen', 'target_0_oppClose', ...
    'target_1_priority', 'target_1_target_angle', 'target_1_oppOpen', 'target_1_oppClose', ...
    'target_2_priority', 'target_2_target_angle', 'target_2_oppOpen', 'target_2_oppClose', ...
    'target_3_priority', 'target_3_target_angle', 'target_3_oppOpen', 'target_3_oppClose', ...
    'target_4_priority', 'target_4_target_angle', 'target_4_oppOpen', 'target_4_oppClose', ...
    'eclipse_start', 'eclipse_end'};


episodes = unique(evalData.episode);

for eIdx = 1:length(episodes)
    disp("Episode: " + num2str(eIdx))
    episodeNum = episodes(eIdx);
    thisEpisodeData = evalData(evalData.episode == episodeNum, :);
    thisEpisodeData.stepStartTime = [0; cumsum(thisEpisodeData.d_ts(1:end-1))];
    
    figure(eIdx)
    % Rward Plot
    subplot(6, 1, 1)
    plot(thisEpisodeData.stepStartTime, thisEpisodeData.reward, 'LineWidth', 2, 'DisplayName','reward')
    hold on
    plot(thisEpisodeData.stepStartTime, thisEpisodeData.total_reward, 'LineWidth', 2, 'DisplayName','totalReward')
    grid on
    grid minor
    xlabel("Step")
    ylabel("Reward")
    legend show

    if (false)
        % Action Plot
        actionData = thisEpisodeData.action;
        actionData(actionData >= 1) = 1;
        subplot(6, 1, 2)
        scatter(thisEpisodeData.stepStartTime, actionData)
        xlabel("Step")
        ylabel("Action")
        yticks([0 1])
        yticklabels({'Charge', 'Image'})
        grid on
        grid minor
    else
         % Action Plot
        subplot(6, 1, 2)
        plot(thisEpisodeData.stepStartTime, thisEpisodeData.action, 'LineWidth', 2)
        xlabel("Step")
        ylabel("Action")
        yticks([0:1:5])
        yticklabels({'Charge', 'Tgt0', 'Tgt1', 'Tgt2', 'Tgt3', 'Tgt4'})
        grid on
        grid minor
    end

    %Action rank
    scores = [thisEpisodeData.target_0_priority, ...
        thisEpisodeData.target_1_priority, ...
        thisEpisodeData.target_2_priority, ...
        thisEpisodeData.target_3_priority, ...
        thisEpisodeData.target_4_priority];
    [~, order] = sort(scores, 2, 'descend');
    [~, ranks] = sort(order, 2);
    chosen_rank = zeros(height(thisEpisodeData), 1);
    
    for i = 1:height(thisEpisodeData)
        scores = [thisEpisodeData.target_0_priority, ...
            thisEpisodeData.target_1_priority, ...
            thisEpisodeData.target_2_priority, ...
            thisEpisodeData.target_3_priority, ...
            thisEpisodeData.target_4_priority];
        oppOpenings = [thisEpisodeData.target_0_oppOpen(i) < 0.01, ...
            thisEpisodeData.target_1_oppOpen(i) < 0.01, ...
            thisEpisodeData.target_2_oppOpen(i) < 0.01, ...
            thisEpisodeData.target_3_oppOpen(i) < 0.01, ...
            thisEpisodeData.target_4_oppOpen(i) < 0.01];

        

        a = thisEpisodeData.action(i);
        if (a == 0) % Charge
            chosen_rank(i) = -1;
        else % Image

            chosen_rank(i) = ranks(i, a);
        end
    end
    subplot(6, 1, 3)
    scatter(thisEpisodeData.stepStartTime, chosen_rank, 'LineWidth', 2)
    xlabel("Step")
    ylabel("Action Rank")
    grid on
    grid minor
    numImaging = length(chosen_rank);
    disp("    Percent Rank 1: " + num2str(100*sum(chosen_rank == 1) / numImaging))
    disp("    Percent Rank 2: " + num2str(100*sum(chosen_rank == 2) / numImaging))
    disp("    Percent Rank 3: " + num2str(100*sum(chosen_rank == 3) / numImaging))
    disp("    Percent Rank 4: " + num2str(100*sum(chosen_rank == 4) / numImaging))
    disp("    Percent Rank 5: " + num2str(100*sum(chosen_rank == 5) / numImaging))

    % Resource Fractions
    subplot(6, 1, 4)
    plot(thisEpisodeData.stepStartTime, thisEpisodeData.battery_charge_fraction, 'LineWidth', 2, 'DisplayName','BatteryFraction')
    hold on
    plot(thisEpisodeData.stepStartTime, thisEpisodeData.storage_level_fraction, 'LineWidth', 2, 'DisplayName','StorageFraction')
    hold on
    grid on
    grid minor
    xlabel("Step")
    ylabel("Fraction")
    legend show

    % In eclipse
    subplot(6, 1, 5)
    inEclipse = thisEpisodeData(thisEpisodeData.eclipse_start < thisEpisodeData.eclipse_start, :);
    scatter(inEclipse.stepStartTime, ones(length(inEclipse.step), 1))
    xlabel("Step")
    ylabel("Steps In Eclipse")
    grid on
    grid minor


    subplot(6, 1, 6)
    actions = thisEpisodeData.action;
    isOpen = zeros(height(thisEpisodeData), 1);
    for m = 1:height(thisEpisodeData)
        if actions(m) ~= 0
            chosenColOpening = ['target_', num2str(actions(m)-1), '_oppOpen'];
            isOpen(m) = thisEpisodeData.(chosenColOpening)(m) < 1e-3;
        else
            isOpen(m) = -1;
        end
    end
    plot(thisEpisodeData.stepStartTime, isOpen)

end
