clc; clear all; close all; 

rng('default');

% set properties for plotting
set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

%% === Initialization ================================================== %%

% Tightening scheme: exact or robust
tightening = 'robust';

% Load parameters
Parameters

% Make simulation time and storing variables for states, outputs and input.  
% The initial state is the origin (= system is in the steady-state).
t = p.T0 : p.Ts : p.Tf;

% System states
x_numeric_exact = zeros(p.nx, length(t)); x_numeric_exact(:, 1) = [10; 0];
x_numeric_robust = x_numeric_exact;

% Control input
u_numeric_exact = zeros(p.nu, length(t)-1);
u_numeric_robust = u_numeric_exact;

u_analytic_exact = zeros(p.nu, length(t)-1);
u_analytic_robust = u_analytic_exact;

% Regret
regret_total = zeros(1, length(t)-1);  
regret_numeric = zeros(1, length(t)-1);
regret_analytic = zeros(1, length(t)-1);
regret_closed_loop = zeros(1, length(t)-1);


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
    
    % Compute disturbance (the same for both cases for comparability)
    % w = mvnrnd([0 0], p.Sigma_w)';
    w = mvlaprnd(2, zeros(2,1), p.Sigma_w, 1);
    
    
    % Apply optimal input to the system
    x_numeric_exact(:, k+1) = p.model.A * x_numeric_exact(:, k) + p.model.B * u_numeric_exact(k) + w;
    x_numeric_robust(:, k+1) = p.model.A * x_numeric_robust(:, k) + p.model.B * u_numeric_robust(k) + w;
    
    % Compute regret (Open Loop)
    regret_numeric(k) = fval_numeric_robust - fval_numeric_exact;
    
    % Compute closed loop regret
    state_contribution = MatrixWeightedNorm(x_numeric_robust, Q) - MatrixWeightedNorm(x_numeric_exact, Q);
    control_contribution = MatrixWeightedNorm(u_numeric_robust, R) - MatrixWeightedNorm(u_numeric_exact, R);
    regret_closed_loop(k) = state_contribution + control_contribution;
    
    % compute infinite horizon total regret
    regret_total(k) = regret_numeric(k) + regret_closed_loop(k);
    
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
    
    regret_analytic(k) = fval_analytic_robust - fval_analytic_exact;
    
end

% Compute the accumuated numeric and analytic regrets
cumulative_regret_numeric = cumsum(regret_numeric);
cumulative_regret_analytic = cumsum(regret_analytic);

% Display the time when both systems entered the set Phi
fprintf('Time step when both systems enter Phi: %d \n', phiTimeEnter);

%% === Plotting ======================================================== %%

% Comparision of closed-loop solutions for exact and distributionally 
% robust tightening of state constraints
figure(1); hold on; grid on;
plot(t(1:end-1), u_numeric_exact, 'Linewidth', 5);
plot(t(1:end-1), u_numeric_robust, ':', 'Linewidth', 5);
% Plot the u_max and u_min
plot(t(1:end-1), bu(1)*ones(size(t(1:end-1))), '--', 'Linewidth', 5);
plot(t(1:end-1), -bu(2)*ones(size(t(1:end-1))), '--', 'Linewidth', 5);
% Plot the start of time entering into set Phi
phiEntryTime = 4.1;
yVector = -bu(2):bu(1);
plot(phiEntryTime*ones(size(yVector)), yVector, '--', 'Linewidth', 5);
xlabel('t (s)'); ylabel('u');
xlim([t(1), t(end)]);
text(4.3,-bu(2)+1,'$\tau_{\Phi}$', 'FontSize', 50);
legend('Exact', 'Dist. Robust', '$u_{\mathrm{max}}$', '$u_{\mathrm{min}}$');
a = findobj(gcf, 'type', 'axes');
h = findobj(gcf, 'type', 'line');
set(h, 'linewidth', 10);
set(a, 'linewidth', 10);
set(a, 'FontSize', 70);
gca.XAxis.TickLabelFormat = '\\textbf{%g}';
gca.YAxis.TickLabelFormat = '\\textbf{%g}';

figure(2); hold on; grid on;
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

figure(3); hold on; grid on;
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


% Regret (numerically and analytically computed)
figure(4); hold on; grid on;
plot(t(1:end-1), regret_numeric, 'Linewidth', 2);
plot(t(1:end-1), regret_analytic, ':', 'Linewidth', 2);
% Plot the start of time entering into set Phi
phiEntryTime = 4.1;
yVector = -20:80;
plot(phiEntryTime*ones(size(yVector)), yVector, '--', 'Linewidth', 5);
xlabel('t (s)'); ylabel('Regret');
legend('Numeric', 'Analytic');
text(4.3,-15,'$\tau_{\Phi}$', 'FontSize', 50);
xlim([t(1), t(end)]);
a = findobj(gcf, 'type', 'axes');
h = findobj(gcf, 'type', 'line');
set(h, 'linewidth', 10);
set(a, 'linewidth', 10);
set(a, 'FontSize', 50);
gca.XAxis.TickLabelFormat = '\\textbf{%g}';
gca.YAxis.TickLabelFormat = '\\textbf{%g}';

% Accumuated Regret (numerically and analytically computed)
figure(5); hold on; grid on;
plot(t(1:end-1), cumulative_regret_numeric, 'Linewidth', 2);
plot(t(1:end-1), cumulative_regret_analytic, ':', 'Linewidth', 2);
% Plot the start of time entering into set Phi
phiEntryTime = 4.1;
yVector = 0:2500;
plot(phiEntryTime*ones(size(yVector)), yVector, '--', 'Linewidth', 5);
xlabel('t (s)'); ylabel('Cumulative Regret');
legend('Numeric', 'Analytic');
xlim([t(1), t(end)]);
text(4.3,200,'$\tau_{\Phi}$', 'FontSize', 50);
a = findobj(gcf, 'type', 'axes');
h = findobj(gcf, 'type', 'line');
set(h, 'linewidth', 10);
set(a, 'linewidth', 10);
set(a, 'FontSize', 50);
gca.XAxis.TickLabelFormat = '\\textbf{%g}';
gca.YAxis.TickLabelFormat = '\\textbf{%g}';

% Total Regret (Numerically Computed)
figure(6); hold on; grid on;
plot(t(1:end-1), regret_numeric, 'Linewidth', 2);
plot(t(1:end-1), regret_closed_loop, 'Linewidth', 2);
plot(t(1:end-1), regret_total, 'Linewidth', 2);
% Plot the start of time entering into set Phi
phiEntryTime = 4.1;
yVector = min(regret_closed_loop) - 10:max(regret_total);
plot(phiEntryTime*ones(size(yVector)), yVector, '--', 'Linewidth', 5);
xlabel('t (s)'); 
legend('$R^{\mathrm{ol}}$', '$R^{\mathrm{cl}}$', '$R^{\mathrm{total}}$');
xlim([t(1), t(end)]);
ylim([min(regret_closed_loop) - 10, max(regret_total) + 20]);
text(4.3,20,'$\tau_{\Phi}$', 'FontSize', 50);
a = findobj(gcf, 'type', 'axes');
h = findobj(gcf, 'type', 'line');
set(h, 'linewidth', 10);
set(a, 'linewidth', 10);
set(a, 'FontSize', 50);
gca.XAxis.TickLabelFormat = '\\textbf{%g}';
gca.YAxis.TickLabelFormat = '\\textbf{%g}';

% % Comparison of numerically and anlytically computed control inputs
% figure(2);
% 
% subplot(1, 2, 1); hold on; grid on;
% plot(t(1:end-1), u_numeric_exact, 'Linewidth', 2);
% plot(t(1:end-1), u_analytic_exact, ':', 'Linewidth', 2);
% xlabel('t (s)'); ylabel('u');
% xlim([t(1), t(end)]);
% legend('Numeric', 'Analytic');
% title('Exact Tightening');
% 
% subplot(1, 2, 2); hold on; grid on;
% plot(t(1:end-1), u_numeric_robust, 'Linewidth', 2);
% plot(t(1:end-1), u_analytic_robust, ':', 'Linewidth', 2);
% xlabel('t (s)'); ylabel('u');
% xlim([t(1), t(end)]);
% legend('Numeric', 'Analytic');
% title('Dist. Robust Tightening');
% a = findobj(gcf, 'type', 'axes');
% h = findobj(gcf, 'type', 'line');
% set(h, 'linewidth', 4);
% set(a, 'linewidth', 4);
% set(a, 'FontSize', 40);
% gca.XAxis.TickLabelFormat = '\\textbf{%g}';
% gca.YAxis.TickLabelFormat = '\\textbf{%g}';

function [normResult] = MatrixWeightedNorm(zeta, Z)
    [~, zeta_n] = size(zeta);
    normResult = 0;
    for zz = 1:zeta_n
        normResult = normResult + zeta(:, zz)'*Z*zeta(:, zz);
    end
end