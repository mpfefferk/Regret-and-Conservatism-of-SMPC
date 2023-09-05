clc; clear all; close all; 

rng('default');

%% === Initialization ================================================== %%

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
regret_numeric = zeros(1, length(t)-1);
regret_analytic = zeros(1, length(t)-1);
regret_analytic_v2 = regret_analytic;
regret_analytic_unconstrained = regret_analytic_v2;

%% === Closed-Loop Simulation ========================================== %%

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
    w = mvnrnd([0 0], p.Sigma_w)';
    
    % Apply optimal input to the system
    x_numeric_exact(:, k+1) = p.model.A * x_numeric_exact(:, k) + p.model.B * u_numeric_exact(k) + w;
    x_numeric_robust(:, k+1) = p.model.A * x_numeric_robust(:, k) + p.model.B * u_numeric_robust(k) + w;
    
    % Compute regret
    regret_numeric(k) = fval_numeric_robust - fval_numeric_exact;
    
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
    
    %%%% Regret
    
    % Compute regret
    fval_analytic_exact = 1/2 * u_seq_analytic_exact' * p.H * u_seq_analytic_exact + ...
        p.fT(x_numeric_exact(:, k)) * u_seq_analytic_exact + ...
        p.const(x_numeric_exact(:, k));
    fval_analytic_robust = 1/2 * u_seq_analytic_robust' * p.H * u_seq_analytic_robust + ...
        p.fT(x_numeric_robust(:, k)) * u_seq_analytic_robust + ...
        p.const(x_numeric_robust(:, k));
    
    regret_analytic(k) = fval_analytic_robust - fval_analytic_exact;
    
    %%%% Alternative computation of regret with more details for analytical
    %%%% insights
    
    % Split active constraints explicitly in active original input and 
    % original state constraints
    n_input_constrs = size(p.Ju, 1);
    n_state_constrs = size(p.Fu, 1);
    
    idx_active_input_exact = idx_active_exact(1:n_input_constrs);
    idx_active_input_robust = idx_active_robust(1:n_input_constrs);
    idx_active_state_exact = idx_active_exact(n_input_constrs+1:end);
    idx_active_state_robust = idx_active_robust(n_input_constrs+1:end);
    
    % Extract parts of the constraint that correspond to active constraints
    g_tilde_exact = p.bx(idx_active_state_exact);
    g_tilde_robust = p.bx(idx_active_state_robust);
    
    F_tilde_A_exact = p.Jx_M(idx_active_state_exact, :);
    F_tilde_A_robust = p.Jx_M(idx_active_state_robust, :);
    
    d_tilde_exact = p.bu(idx_active_input_exact);
    d_tilde_robust = p.bu(idx_active_input_robust);
    
    M_tilde_robust = [p.Ju; p.Fu]; M_tilde_robust = M_tilde_robust(idx_active_robust, :);
    M_tilde_exact = [p.Ju; p.Fu]; M_tilde_exact = M_tilde_exact(idx_active_exact, :);
    
    V_exact = p.H_inv * M_tilde_exact' / (M_tilde_exact * p.H_inv * M_tilde_exact' + 1E-08 * eye(size(M_tilde_exact, 1)));
    V_robust = p.H_inv * M_tilde_robust' / (M_tilde_robust * p.H_inv * M_tilde_robust' + 1E-08 * eye(size(M_tilde_robust, 1)));
    
    V1_exact = V_exact(:, 1:numel(d_tilde_exact));
    V2_exact = V_exact(:, numel(d_tilde_exact)+1:end);
    
    V1_robust = V_robust(:, 1:numel(d_tilde_robust));
    V2_robust = V_robust(:, numel(d_tilde_robust)+1:end);
    
    phi_tilde_exact = p.phi_ex(idx_active_state_exact);
    phi_tilde_robust = p.phi_dr(idx_active_state_robust);
    
    z_tilde_exact = p.z(idx_active_state_exact);
    z_tilde_robust = p.z(idx_active_state_robust);
    
    % The following was only used to validate the previous computations
%     u_seq_test_exact = (V_exact * M_tilde_exact * p.H_inv - p.H_inv) * p.fT_tilde' * x_numeric_exact(:, k) + V1_exact * d_tilde_exact ...
%         + V2_exact * (g_tilde_exact - F_tilde_A_exact * x_numeric_exact(:, k)) - V2_exact * diag(phi_tilde_exact) * z_tilde_exact;
%     
%     u_analytic_exact(k) = u_seq_test_exact(1);
%     
%     u_seq_test_robust = (V_robust * M_tilde_robust * p.H_inv - p.H_inv) * p.fT_tilde' * x_numeric_robust(:, k) + V1_robust * d_tilde_robust ...
%         + V2_robust * (g_tilde_robust - F_tilde_A_robust * x_numeric_robust(:, k)) - V2_robust * diag(phi_tilde_robust) * z_tilde_robust;
%     
%     u_analytic_robust(k) = u_seq_test_robust(1);
    
    % Define constants for short-hand notion of the optimal input sequence.
    % This is done only once for exact and robust as the following holds
    % only when the same constraints are active for both schemes
    alpha_exact = (V_exact * M_tilde_exact * p.H_inv - p.H_inv) * p.fT_tilde' - V2_exact * F_tilde_A_exact; 
    alpha_robust = (V_robust * M_tilde_robust * p.H_inv - p.H_inv) * p.fT_tilde' - V2_robust * F_tilde_A_robust;
    gamma_exact = V1_exact * d_tilde_exact + V2_exact * g_tilde_exact;
    gamma_robust = V1_robust * d_tilde_robust + V2_robust * g_tilde_robust;
    
    u_seq_test_exact = alpha_exact * x_numeric_exact(:, k) - V2_exact * diag(phi_tilde_exact) * z_tilde_exact + gamma_exact;
    u_seq_test_robust = alpha_robust * x_numeric_robust(:, k) - V2_robust * diag(phi_tilde_robust) * z_tilde_robust + gamma_robust;
    
    u_analytic_exact(k) = u_seq_test_exact(1);
    u_analytic_robust(k) = u_seq_test_robust(1);
    
    alpha = alpha_exact;
    gamma = gamma_exact;
    
    % Precompute matrix expressions
    Lambda_1 = 1/2 * alpha' * p.H * alpha + 1/2 * (p.fT_tilde * alpha + alpha' * p.fT_tilde') + p.AT_Q_A;
    Lambda_2 = 1/2 * V2_exact' * p.H * V2_exact;
    Lambda_3 = 1/2 * (p.fT_tilde * V2_exact + alpha' * p.H * V2_exact);
    Lambda_4 = p.fT_tilde * gamma + alpha' * p.H * gamma;
    Lambda_5 = V2_exact' * p.H * gamma;
    
    % Compute regret using its more detailled representation (only valid
    % when the same constraints are active for both schemes)
    if all(idx_active_exact == idx_active_robust)
       
        % Precompute quantities of interes
        dx_m = x_numeric_exact(:, k) - x_numeric_robust(:, k);
        dx_p = x_numeric_exact(:, k) + x_numeric_robust(:, k);
        
        dphi_m = diag(phi_tilde_exact - phi_tilde_robust);
        dphi_p = diag(phi_tilde_exact + phi_tilde_robust);
        
%         du_m = u_seq_test_exact - u_seq_test_robust;
%         du_p = u_seq_test_exact + u_seq_test_robust;
        
        du_m = alpha * dx_m - V2_exact * dphi_m * z_tilde_exact;
        du_p = alpha * dx_p - V2_exact * dphi_p * z_tilde_exact + 2 * gamma;
        
%         regret_analytic_v2(k) = 1/2 * du_m' * p.H * du_p ...
%             + x_numeric_exact(:, k)' * p.fT_tilde * u_seq_test_exact ...
%             - x_numeric_robust(:, k)' * p.fT_tilde * u_seq_test_robust ...
%             + dx_m' * p.AT_Q_A * dx_p;
        
%         regret_analytic_v2(k) = 1/2 * (alpha * dx_m - V2_exact * dphi_m * z_tilde_exact)' * p.H * (alpha * dx_p - V2_exact * dphi_p * z_tilde_exact + 2 * gamma) ...
%             + x_numeric_exact(:, k)' * p.fT_tilde * (alpha * x_numeric_exact(:, k) - V2_exact * diag(phi_tilde_exact) * z_tilde_exact + gamma) ...
%             - x_numeric_robust(:, k)' * p.fT_tilde * (alpha * x_numeric_robust(:, k) - V2_exact * diag(phi_tilde_robust) * z_tilde_exact + gamma) ...
%             + dx_m' * p.AT_Q_A * dx_p;
        
%         regret_analytic_v2(k) = 1/2 * dx_m' * alpha' * p.H * alpha * dx_p + 1/2 * z_tilde_exact' * dphi_m * V2_exact' * p.H * V2_exact * dphi_p * z_tilde_exact ...
%             -1/2 * dx_m' * alpha' * p.H * V2_exact * dphi_p * z_tilde_exact - 1/2 * z_tilde_exact' * dphi_m * V2_exact' * p.H * alpha * dx_p ...
%             + dx_m' * alpha' * p.H * gamma - z_tilde_exact' * dphi_m * V2_exact' * p.H * gamma ...        
%             + x_numeric_exact(:, k)' * p.fT_tilde * (alpha * x_numeric_exact(:, k) - V2_exact * diag(phi_tilde_exact) * z_tilde_exact + gamma) ...
%             - x_numeric_robust(:, k)' * p.fT_tilde * (alpha * x_numeric_robust(:, k) - V2_exact * diag(phi_tilde_robust) * z_tilde_exact + gamma) ...
%             + dx_m' * p.AT_Q_A * dx_p;
        
%         regret_analytic_v2(k) = 1/2 * dx_m' * alpha' * p.H * alpha * dx_p + 1/2 * z_tilde_exact' * dphi_m * V2_exact' * p.H * V2_exact * dphi_p * z_tilde_exact ...
%             -1/2 * dx_m' * alpha' * p.H * V2_exact * dphi_p * z_tilde_exact - 1/2 * z_tilde_exact' * dphi_m * V2_exact' * p.H * alpha * dx_p ...
%             + dx_m' * alpha' * p.H * gamma - z_tilde_exact' * dphi_m * V2_exact' * p.H * gamma ...             
%             + x_numeric_exact(:, k)' * p.fT_tilde * alpha * x_numeric_exact(:, k) - x_numeric_robust(:, k)' * p.fT_tilde * alpha * x_numeric_robust(:, k) ...
%             - x_numeric_exact(:, k)' * p.fT_tilde * V2_exact * diag(phi_tilde_exact) * z_tilde_exact + x_numeric_robust(:, k)' * p.fT_tilde * V2_exact * diag(phi_tilde_robust) * z_tilde_exact ...
%             + x_numeric_exact(:, k)' * p.fT_tilde * gamma - x_numeric_robust(:, k)' * p.fT_tilde * gamma ...           
%             + dx_m' * p.AT_Q_A * dx_p;
        
%         regret_analytic_v2(k) = dx_m' * Lambda_1 * dx_p + 1/2 * z_tilde_exact' * dphi_m * V2_exact' * p.H * V2_exact * dphi_p * z_tilde_exact ...
%             -1/2 * dx_m' * alpha' * p.H * V2_exact * dphi_p * z_tilde_exact - 1/2 * z_tilde_exact' * dphi_m * V2_exact' * p.H * alpha * dx_p ...
%             + dx_m' * alpha' * p.H * gamma - z_tilde_exact' * dphi_m * V2_exact' * p.H * gamma ...             
%             - x_numeric_exact(:, k)' * p.fT_tilde * V2_exact * diag(phi_tilde_exact) * z_tilde_exact + x_numeric_robust(:, k)' * p.fT_tilde * V2_exact * diag(phi_tilde_robust) * z_tilde_exact ...
%             + x_numeric_exact(:, k)' * p.fT_tilde * gamma - x_numeric_robust(:, k)' * p.fT_tilde * gamma ...
%             - x_numeric_exact(:, k)' * p.fT_tilde * alpha * x_numeric_robust(:, k) + x_numeric_robust(:, k)' * p.fT_tilde * alpha * x_numeric_exact(:, k);
        
%         regret_analytic_v2(k) = dx_m' * Lambda_1 * dx_p + 1/2 * z_tilde_exact' * dphi_m * V2_exact' * p.H * V2_exact * dphi_p * z_tilde_exact ...
%             -1/2 * dx_m' * alpha' * p.H * V2_exact * dphi_p * z_tilde_exact - 1/2 * z_tilde_exact' * dphi_m * V2_exact' * p.H * alpha * dx_p ...
%             + dx_m' * alpha' * p.H * gamma - z_tilde_exact' * dphi_m * V2_exact' * p.H * gamma ...             
%             - x_numeric_exact(:, k)' * p.fT_tilde * V2_exact * diag(phi_tilde_exact) * z_tilde_exact + x_numeric_robust(:, k)' * p.fT_tilde * V2_exact * diag(phi_tilde_robust) * z_tilde_exact ...
%             + x_numeric_exact(:, k)' * p.fT_tilde * gamma - x_numeric_robust(:, k)' * p.fT_tilde * gamma;
        
        regret_analytic_v2(k) = dx_m' * Lambda_1 * dx_p ...
            + z_tilde_exact' * dphi_m * Lambda_2 * dphi_p * z_tilde_exact ...
            - dx_m' * Lambda_3 * dphi_p * z_tilde_exact ...
            - dx_p' * Lambda_3 * dphi_m * z_tilde_exact ...
            + dx_m' * Lambda_4 ...
            - z_tilde_exact' * dphi_m * Lambda_5;
        
        regret_analytic_unconstrained(k) = -dx_m' * (p.AT_Q_A - 1/2 * p.fT_tilde * p.H_inv * p.fT_tilde') * dx_p;
        
        % In my notes, I defined regret as "exact minus robust", while I
        % implemented regret here as "robust minus exact"
        regret_analytic_v2(k) = -regret_analytic_v2(k);
        
    else
        
        regret_analytic_v2(k) = NaN;
        
    end
    
    
end

%% === Plotting ======================================================== %%

% Comparision of closed-loop solutions for exact and distributionally 
% robust tightening of state constraints
figure(1);

subplot(1, 3, 1); hold on; grid on;
plot(t(1:end-1), u_numeric_exact, 'Linewidth', 2);
plot(t(1:end-1), u_numeric_robust, ':', 'Linewidth', 2);
xlabel('t (s)'); ylabel('u');
xlim([t(1), t(end)]);
legend('Exact Tightening', 'Robust Tightening');

subplot(1, 3, 2); hold on; grid on;
plot(t, x_numeric_exact(1, :), 'Linewidth', 2);
plot(t, x_numeric_robust(1, :), ':', 'Linewidth', 2);
xlabel('t (s)'); ylabel('x_1');
xlim([t(1), t(end)]);

subplot(1, 3, 3); hold on; grid on;
plot(t, x_numeric_exact(2, :), 'Linewidth', 2);
plot(t, x_numeric_robust(2, :), ':', 'Linewidth', 2);
xlabel('t (s)'); ylabel('x_2');
xlim([t(1), t(end)]);

% Comparison of numerically and anlytically computed control inputs
figure(2);

subplot(1, 2, 1); hold on; grid on;
plot(t(1:end-1), u_numeric_exact, 'Linewidth', 2);
plot(t(1:end-1), u_analytic_exact, ':', 'Linewidth', 2);
xlabel('t (s)'); ylabel('u');
xlim([t(1), t(end)]);
legend('Numeric', 'Analytic');
title('Exact Tightening');

subplot(1, 2, 2); hold on; grid on;
plot(t(1:end-1), u_numeric_robust, 'Linewidth', 2);
plot(t(1:end-1), u_analytic_robust, ':', 'Linewidth', 2);
xlabel('t (s)'); ylabel('u');
xlim([t(1), t(end)]);
legend('Numeric', 'Analytic');
title('Dist. Robust Tightening');

% Regret (numerically and analytically computed)
figure(3); hold on; grid on;
plot(t(1:end-1), regret_numeric, 'Linewidth', 2);
plot(t(1:end-1), regret_analytic, ':', 'Linewidth', 2);
plot(t(1:end-1), regret_analytic_v2, 'gx', 'Linewidth', 2, 'Markersize', 3, 'Linestyle', 'none');
plot(t(1:end-1), regret_analytic_unconstrained, 'r-');
xlabel('t (s)'); ylabel('Regret');
legend('Numeric', 'Analytic');
xlim([t(1), t(end)]);