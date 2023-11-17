clc; clear all; close all; 

rng('default');

% set properties for plotting
set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
addpath(genpath('src'));

%% === Initialization ================================================== %%

% Tightening scheme: exact or robust
tightening = 'robust';

% Load parameters
Parameters

% Make simulation time and storing variables for states, outputs and input.  
% The initial state is the origin (= system is in the steady-state).
t = p.T0 : p.Ts : p.Tf;

% System states
x_numeric_exact = zeros(p.nx, length(t)); 
x_numeric_exact(:, 1) = [10; 0];
x_numeric_robust = x_numeric_exact;
x_closed_loop_analytic_exact = x_numeric_exact;
x_closed_loop_analytic_robust = x_numeric_exact;

% Control input
u_numeric_exact = zeros(p.nu, length(t)-1);
u_numeric_robust = u_numeric_exact;
u_closed_loop_analytic_exact = u_numeric_exact;

u_analytic_exact = zeros(p.nu, length(t)-1);
u_analytic_robust = u_analytic_exact;
u_closed_loop_analytic_robust = u_numeric_robust;

P_closed_loop_analytic_exact = zeros(length(t)-1, p.nx);
P_closed_loop_analytic_robust = zeros(length(t)-1, p.nx);
q_closed_loop_analytic_exact = zeros(1, length(t)-1);
q_closed_loop_analytic_robust = zeros(1, length(t)-1);
gamma_closed_loop_exact = zeros(1, length(t)-1);
gamma_closed_loop_robust = zeros(1, length(t)-1);
running_sum_exact = zeros(p.nx, length(t)-1);
running_sum_robust = zeros(p.nx, length(t)-1);
psi_exact = zeros(length(t)-1, p.nx);
psi_robust = zeros(length(t)-1, p.nx);

% Compute disturbance (the same for both cases for comparability)
w = mvlaprnd(2, zeros(2,1), p.Sigma_w, length(t)-1); % w = mvnrnd([0 0], p.Sigma_w)';

% Regret
regret_total_numeric = zeros(1, length(t)-1);  
regret_total_analytic = zeros(1, length(t)-1);  
regret_open_loop_numeric = zeros(1, length(t)-1);
regret_open_loop_analytic = zeros(1, length(t)-1);
regret_closed_loop_numeric = zeros(1, length(t)-1);
regret_closed_loop_analytic = zeros(1, length(t)-1);


%% === Closed-Loop Simulation ========================================== %%

% Set flag to record when both systems enters the set Phi
notInPhiFlag = 1;
exactSystemInPhi = 0;
robustSystemInPhi = 0;

% Simulation loop
for k = 1 : 1 : length(t)-1
    
    % =================================================================== %
    % Numeric solution of QP
        
    % Initial guess of the input sequence for the optimization routine
    if k == 1
        u_seq_exact_0 = zeros(p.N, 1);
        u_seq_robust_0 = zeros(p.N, 1);
    else
        u_seq_exact_0 = [u_seq_numeric_exact(2:end); u_seq_numeric_exact(end)];
        u_seq_robust_0 = [u_seq_numeric_robust(2:end); u_seq_numeric_robust(end)];
    end
    
    % MPC using quadratic programming with exact and robust constraint tightening
    [u_seq_numeric_exact, fval_numeric_exact] = Linear_MPC(u_seq_exact_0, x_numeric_exact(:, k), p, 'exact');
    [u_seq_numeric_robust, fval_numeric_robust] = Linear_MPC(u_seq_robust_0, x_numeric_robust(:, k), p, 'robust');
    
    % Extract control input for the next sampling period
    u_numeric_exact(k) = u_seq_numeric_exact(1);
    u_numeric_robust(k) = u_seq_numeric_robust(1);
    
    % Apply optimal input to the system
    x_numeric_exact(:, k+1) = p.model.A * x_numeric_exact(:, k) + p.model.B * u_numeric_exact(k) + w(:, k);
    x_numeric_robust(:, k+1) = p.model.A * x_numeric_robust(:, k) + p.model.B * u_numeric_robust(k) + w(:, k);
    
    % Compute regret (Open Loop)
    regret_open_loop_numeric(k) = fval_numeric_robust - fval_numeric_exact;
    
    % Compute closed loop regret
    state_contribution = MatrixWeightedNorm(x_numeric_robust, Q) - MatrixWeightedNorm(x_numeric_exact, Q);
    control_contribution = MatrixWeightedNorm(u_numeric_robust, R) - MatrixWeightedNorm(u_numeric_exact, R);
    regret_closed_loop_numeric(k) = state_contribution + control_contribution;
    
    % compute infinite horizon total regret
    regret_total_numeric(k) = regret_open_loop_numeric(k) + regret_closed_loop_numeric(k);
    
    % =================================================================== %
    % Analytic solution of QP
    
    % Given the optimal solution (obtained from the numeric approach),
    % determine the set of active inequality constraints
    gu_exact = p.gu(x_numeric_exact(:, k));
    gu_exact = gu_exact - p.phi_ex .* p.z;
    
    gu_robust = p.gu(x_numeric_robust(:, k));
    gu_robust = gu_robust - p.phi_dr .* p.z;
 
    ineq_constr_vals_exact = [p.Ju; p.Fu] * u_seq_numeric_exact - [p.bu; gu_exact];
    ineq_constr_vals_robust = [p.Ju; p.Fu] * u_seq_numeric_robust - [p.bu; gu_robust];
    
    idx_active_exact = abs(ineq_constr_vals_exact) <= 1E-04;
    idx_active_robust = abs(ineq_constr_vals_robust) <= 1E-04;

    %%%% Exact tightening of state constraints
    
    % If no constraints are active, use the unconstrained optimum
    if ~any(idx_active_exact)
        exactSystemInPhi = 1;
        u_seq_analytic_exact = -p.H_inv * p.fT(x_numeric_exact(:, k))';      
        u_analytic_exact(k) = u_seq_analytic_exact(1);
        
    % Otherwise, use the set of active constraints to construct the
    % constrained solution
    else
        
        % Extract the active constraints
        M_tilde = [p.Ju; p.Fu]; M_tilde = M_tilde(idx_active_exact, :);
        b_tilde = [p.bu; gu_exact]; b_tilde = b_tilde(idx_active_exact);
        
        % Solve the Lagrangian system for the optimal input sequence
        u_seq_analytic_exact = (p.H_inv * M_tilde' / (M_tilde * p.H_inv * M_tilde' + 1E-08 * eye(size(M_tilde, 1))) * M_tilde * p.H_inv - p.H_inv) * p.fT(x_numeric_exact(:, k))' + ...
            p.H_inv * M_tilde' / (M_tilde * p.H_inv * M_tilde' + 1E-08 * eye(size(M_tilde, 1))) * b_tilde;
        
        % Extract the optimal input sequence from the solution of the
        % Lagrangian system
        u_analytic_exact(k) = u_seq_analytic_exact(1);
        
        P_analytic_exact = (p.H_inv * M_tilde' / (M_tilde * p.H_inv * M_tilde' + 1E-08 * eye(size(M_tilde, 1))) * M_tilde * p.H_inv - p.H_inv);
        q_analytic_exact = p.H_inv * M_tilde' / (M_tilde * p.H_inv * M_tilde' + 1E-08 * eye(size(M_tilde, 1))) * b_tilde;
        P_closed_loop_analytic_exact(k, :) = P_analytic_exact(1,:);
        q_closed_loop_analytic_exact(1, k) = q_analytic_exact(1);
        
        % running_sum_exact = SUM (A^i(B*gamma + w))
        if(k > 1)            
            for kk = 1:k-1
                running_sum_exact(:, k) = running_sum_exact(:, k) + (p.model.A)^(kk)*(p.model.B*gamma_closed_loop_exact(1, kk) + ...
                    w(:, kk));
            end
            gamma_closed_loop_exact(1, k) = q_closed_loop_analytic_exact(1, k) + ...
                P_closed_loop_analytic_exact(k, :)*running_sum_exact(:, k);
        end
        
        % Update the u_closed_loop_analytic_exact
        if(k == 1)
            psi_exact(:, k) = P_closed_loop_analytic_exact(k, :);
        else
            % runningStateSum = Sum(A^i*B*\Psi)
            runningStateSum = zeros(p.nx, 1);
            if(k > 1)            
                for kk = 1:k-1
                    runningStateSum = runningStateSum + (p.model.A)^(kk)*(p.model.B*psi_exact(:, kk));
                end            
            end
            psi_exact(:, k) = P_closed_loop_analytic_exact(k, :)*((p.model.A)^(k) + runningStateSum);
        end        
        u_closed_loop_analytic_exact(:, k) = psi_exact(:, k)*x_closed_loop_analytic_exact(:, 1) + gamma_closed_loop_exact(1, k);
              
        
        % Update the x_closed_loop_analytic_exact
        x_closed_loop_analytic_exact(:, k+1) = ((p.model.A)^(k) + runningStateSum)*x_closed_loop_analytic_exact(:, 1) + running_sum_exact;        
        
        
    end
    
    %%%% Distributionally robust tightening of state constraints
    
    % If no constraints are active, use the unconstrained optimum
    if ~any(idx_active_robust)
        robustSystemInPhi = 1;
        u_seq_analytic_robust = -p.H_inv * p.fT(x_numeric_robust(:, k))';      
        u_analytic_robust(k) = u_seq_analytic_robust(1);
        
    % Otherwise, use the set of active constraints to construct the
    % constrained solution
    else
        
        % Extract the active constraints
        M_tilde = [p.Ju; p.Fu]; M_tilde = M_tilde(idx_active_robust, :);
        b_tilde = [p.bu; gu_robust]; b_tilde = b_tilde(idx_active_robust);
        
        % Solve the Lagrangian system for the optimal input sequence
        u_seq_analytic_robust = (p.H_inv * M_tilde' / (M_tilde * p.H_inv * M_tilde' + 1E-08 * eye(size(M_tilde, 1))) * M_tilde * p.H_inv - p.H_inv) * p.fT(x_numeric_robust(:, k))' + ...
            p.H_inv * M_tilde' / (M_tilde * p.H_inv * M_tilde' + 1E-08 * eye(size(M_tilde, 1))) * b_tilde;
        
        % Extract the optimal input sequence from the solutino of the
        % Lagrangian system
        u_analytic_robust(k) = u_seq_analytic_robust(1);
        
        P_analytic_robust = (p.H_inv * M_tilde' / (M_tilde * p.H_inv * M_tilde' + 1E-08 * eye(size(M_tilde, 1))) * M_tilde * p.H_inv - p.H_inv);
        q_analytic_robust = p.H_inv * M_tilde' / (M_tilde * p.H_inv * M_tilde' + 1E-08 * eye(size(M_tilde, 1))) * b_tilde;
        P_closed_loop_analytic_robust(k, :) = P_analytic_robust(1,:);
        q_closed_loop_analytic_robust(1, k) = q_analytic_robust(1);
        
        if(k > 1)            
            for kk = 1:k-1
                running_sum_robust(:, k) = running_sum_robust(:, k) + (p.model.A)^(kk)*(p.model.B*gamma_closed_loop_robust(1, kk) + ...
                    w(:, kk));
            end
            gamma_closed_loop_robust(1, k) = q_closed_loop_analytic_robust(1, k) + ...
                P_closed_loop_analytic_robust(k, :)*running_sum_robust(:, k);
        end
        
        % Update the u_closed_loop_analytic_robust
        if(k == 1)
            psi_robust(:, k) = P_closed_loop_analytic_robust(k, :);
        else
            % runningStateSum = Sum(A^i*B*\Psi)
            runningStateSum = zeros(p.nx, 1);
            if(k > 1)            
                for kk = 1:k-1
                    runningStateSum = runningStateSum + (p.model.A)^(kk)*(p.model.B*psi_robust(:, kk));
                end            
            end
            psi_robust(:, k) = P_closed_loop_analytic_robust(k, :)*((p.model.A)^(k) + runningStateSum);
        end        
        u_closed_loop_analytic_robust(:, k) = psi_robust(:, k)*x_closed_loop_analytic_robust(:, 1) + gamma_closed_loop_robust(1, k);
              
        
        % Update the x_closed_loop_analytic_robust
        x_closed_loop_analytic_robust(:, k+1) = ((p.model.A)^(k) + runningStateSum)*x_closed_loop_analytic_robust(:, 1) + running_sum_robust;        
        
        
    end
    
    % Record the time when both systems enter the set Phi
    if exactSystemInPhi && robustSystemInPhi && notInPhiFlag
        notInPhiFlag = 0;
        phiTimeEnter = k;
    end
    
    % Compute regret
    fval_analytic_exact = 1/2 * u_seq_analytic_exact' * p.H * u_seq_analytic_exact + ...
        p.fT(x_numeric_exact(:, k)) * u_seq_analytic_exact + ...
        p.const(x_numeric_exact(:, k));
    fval_analytic_robust = 1/2 * u_seq_analytic_robust' * p.H * u_seq_analytic_robust + ...
        p.fT(x_numeric_robust(:, k)) * u_seq_analytic_robust + ...
        p.const(x_numeric_robust(:, k));
    
    regret_open_loop_analytic(k) = fval_analytic_robust - fval_analytic_exact;
    
end

% Compute analytic closed loop regret
for k = 2:length(t)-1
    xclExactSq = MatrixWeightedNorm(x_closed_loop_analytic_exact(:, k), Q);
    xclRobustSq = MatrixWeightedNorm(x_closed_loop_analytic_robust(:, k), Q);
    uclExactSq = MatrixWeightedNorm(u_closed_loop_analytic_exact(:, k), R);
    uclRobustSq = MatrixWeightedNorm(u_closed_loop_analytic_robust(:, k), R);
    regret_closed_loop_analytic(1, k) = regret_closed_loop_analytic(1, k-1) + (xclRobustSq + uclRobustSq) - (xclExactSq + uclExactSq);
end

% Compute analytic total regret
for k = 1:length(t)-1
    regret_total_analytic(1, k) = regret_open_loop_analytic(k) + regret_closed_loop_analytic(1, k);
end


% Display the time when both systems entered the set Phi
fprintf('Time step when both systems enter Phi: %d \n', phiTimeEnter);

%% === Plotting ======================================================== %%

figure1 = figure('Color',[1 1 1]);
hold on; grid on;
plot(t, x_numeric_exact(1, :), 'Linewidth', 5);
plot(t, x_numeric_robust(1, :), ':', 'Linewidth', 5);
% Plot the x_1_max and x_1_min
plot(t, bx(1)*ones(size(t)), '--', 'Linewidth', 5);
plot(t, -bx(2)*ones(size(t)), '--', 'Linewidth', 5);
% Plot the start of time entering into set Phi
phiEntryTime = 4.1;
yVector = -bx(2):bx(1);
plot(phiEntryTime*ones(size(yVector)), yVector, '--', 'Linewidth', 5);
xlabel('t (s)'); ylabel('$x_1$');
xlim([t(1), t(end)]);
text(4.3,-bx(2)+1,'$\tau_{\Phi}$', 'FontSize', 50);
legend('Exact', 'Dist. Robust', '$x_{\mathrm{max}}$', '$x_{\mathrm{min}}$');
a = findobj(gcf, 'type', 'axes');
h = findobj(gcf, 'type', 'line');
set(h, 'linewidth', 10);
set(a, 'linewidth', 10);
set(a, 'FontSize', 70);
gca.XAxis.TickLabelFormat = '\\textbf{%g}';
gca.YAxis.TickLabelFormat = '\\textbf{%g}';
% Convert matlab figs to tikz for pgfplots in latex document.
matlab2tikz('figurehandle',figure1,'filename','x1.tex' ,'standalone', true, 'showInfo', false);

figure2 = figure('Color',[1 1 1]);
hold on; grid on;
plot(t, x_numeric_exact(2, :), 'Linewidth', 5);
plot(t, x_numeric_robust(2, :), ':', 'Linewidth', 5);
% Plot the x_2_max and x_2_min
plot(t, bx(3)*ones(size(t)), '--', 'Linewidth', 5);
plot(t, -bx(4)*ones(size(t)), '--', 'Linewidth', 5);
% Plot the start of time entering into set Phi
phiEntryTime = 4.1;
yVector = -bx(4):bx(3);
plot(phiEntryTime*ones(size(yVector)), yVector, '--', 'Linewidth', 5);
xlabel('t (s)'); ylabel('$x_2$');
text(4.3,-bx(4)+0.5,'$\tau_{\Phi}$', 'FontSize', 50);
xlim([t(1), t(end)]);
legend('Exact', 'Dist. Robust', '$x_{\mathrm{max}}$', '$x_{\mathrm{min}}$');
a = findobj(gcf, 'type', 'axes');
h = findobj(gcf, 'type', 'line');
set(h, 'linewidth', 10);
set(a, 'linewidth', 10);
set(a, 'FontSize', 70);
gca.XAxis.TickLabelFormat = '\\textbf{%g}';
gca.YAxis.TickLabelFormat = '\\textbf{%g}';
% Convert matlab figs to tikz for pgfplots in latex document.
matlab2tikz('figurehandle',figure2,'filename','x2.tex' ,'standalone', true, 'showInfo', false);


figure3 = figure('Color',[1 1 1]);
hold on; grid on;
% First state
plot(t, x_numeric_exact(1, :), '-.', 'Linewidth', 5);
plot(t, x_numeric_robust(1, :), ':', 'Linewidth', 5);
% Second state
plot(t, x_numeric_exact(2, :), '-.', 'Linewidth', 5);
plot(t, x_numeric_robust(2, :), ':', 'Linewidth', 5);
% Plot the x_1_max and x_1_min
plot(t, bx(1)*ones(size(t)), '--', 'Linewidth', 5);
plot(t, -bx(2)*ones(size(t)), '--', 'Linewidth', 5);
% Plot the x_2_max and x_2_min
plot(t, bx(3)*ones(size(t)), '--', 'Linewidth', 5);
plot(t, -bx(4)*ones(size(t)), '--', 'Linewidth', 5);
% Plot the start of time entering into set Phi
phiEntryTime = 4.1;
yVector = -bx(4):bx(1);
plot(phiEntryTime*ones(size(yVector)), yVector, '--', 'Linewidth', 5);
xlabel('t (s)');
text(4.3,-bx(4)+0.5,'$\tau_{\Phi}$', 'FontSize', 50);
xlim([t(1), t(end)]);
legend('$[x^{\star}]_1$', '$[x^{\dagger}]_1$', '$[x^{\star}]_2$', '$[x^{\dagger}]_2$', '$x^1_{\mathrm{max}}$', '$x^1_{\mathrm{min}}$', '$x^2_{\mathrm{max}}$', '$x^2_{\mathrm{min}}$', 'NumColumns',2);
a = findobj(gcf, 'type', 'axes');
h = findobj(gcf, 'type', 'line');
set(h, 'linewidth', 10);
set(a, 'linewidth', 10);
set(a, 'FontSize', 70);
gca.XAxis.TickLabelFormat = '\\textbf{%g}';
gca.YAxis.TickLabelFormat = '\\textbf{%g}';
% Convert matlab figs to tikz for pgfplots in latex document.
matlab2tikz('figurehandle',figure3,'filename','states.tex' ,'standalone', true, 'showInfo', false);

% Total Regret (Numerically Computed)
figure4 = figure('Color',[1 1 1]);
hold on; grid on;
plot(t(1:end-1), regret_open_loop_numeric, 'Linewidth', 2);
plot(t(1:end-1), regret_closed_loop_numeric, 'Linewidth', 2);
plot(t(1:end-1), regret_total_numeric, 'Linewidth', 2);
% Plot the start of time entering into set Phi
phiEntryTime = 4.1;
yVector = min(regret_closed_loop_numeric) - 10:max(regret_total_numeric);
plot(phiEntryTime*ones(size(yVector)), yVector, '--', 'Linewidth', 5);
xlabel('t (s)'); 
legend('$R^{\mathrm{ol}}$', '$R^{\mathrm{cl}}$', '$R^{\mathrm{total}}$');
xlim([t(1), t(end)]);
ylim([min(regret_closed_loop_numeric) - 10, max(regret_total_numeric) + 20]);
text(4.3,20,'$\tau_{\Phi}$', 'FontSize', 50);
a = findobj(gcf, 'type', 'axes');
h = findobj(gcf, 'type', 'line');
set(h, 'linewidth', 10);
set(a, 'linewidth', 10);
set(a, 'FontSize', 50);
gca.XAxis.TickLabelFormat = '\\textbf{%g}';
gca.YAxis.TickLabelFormat = '\\textbf{%g}';
% Convert matlab figs to tikz for pgfplots in latex document.
matlab2tikz('figurehandle',figure4,'filename','regrets.tex' ,'standalone', true, 'showInfo', false);

% Total Regret (Analytically Computed)
figure5 = figure('Color',[1 1 1]);
hold on; grid on;
plot(t(1:end-1), regret_open_loop_analytic, 'Linewidth', 2);
plot(t(1:end-1), regret_closed_loop_analytic, 'Linewidth', 2);
plot(t(1:end-1), regret_total_analytic, 'Linewidth', 2);
% Plot the start of time entering into set Phi
phiEntryTime = 4.1;
yVector = min(regret_closed_loop_numeric) - 10:max(regret_total_numeric);
plot(phiEntryTime*ones(size(yVector)), yVector, '--', 'Linewidth', 5);
xlabel('t (s)'); 
legend('$R^{\mathrm{ol}}$', '$R^{\mathrm{cl}}$', '$R^{\mathrm{total}}$');
xlim([t(1), t(end)]);
ylim([min(regret_closed_loop_numeric) - 10, max(regret_total_numeric) + 20]);
text(4.3,20,'$\tau_{\Phi}$', 'FontSize', 50);
a = findobj(gcf, 'type', 'axes');
h = findobj(gcf, 'type', 'line');
set(h, 'linewidth', 10);
set(a, 'linewidth', 10);
set(a, 'FontSize', 50);
gca.XAxis.TickLabelFormat = '\\textbf{%g}';
gca.YAxis.TickLabelFormat = '\\textbf{%g}';
% Convert matlab figs to tikz for pgfplots in latex document.
matlab2tikz('figurehandle',figure5,'filename','analyticregrets.tex' ,'standalone', true, 'showInfo', false);


function [normResult] = MatrixWeightedNorm(zeta, Z)
    [~, zeta_n] = size(zeta);
    normResult = 0;
    for zz = 1:zeta_n
        normResult = normResult + zeta(:, zz)'*Z*zeta(:, zz);
    end
end