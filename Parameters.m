%% === Parameters of the linear quarter-car model ====================== %%

%% === Parameters for the controller (and the simulation) ============== %%

% Initial, sampling and final time
p.T0 = 0;
p.Ts = 0.05;
p.Tf = 10;

% Control / Prediction horizon
p.N = 5;

% Matrices of the time-discrete system
A = [1, p.Ts; 0, 1];
B = [0; p.Ts]; 
C = eye(2); 
D = zeros(2, 1);

model = ss(A, B, C, D, p.Ts);

p.model = model;

% Number of states and inputs
p.nx = size(model.A, 2);
p.nu = size(model.B, 2);

% Covariance matrix of the process noise
p.Sigma_w = 0.01^2 * eye(p.nx);
% p.Sigma_w = 0.01^2 * [0.7, 1.3] * [0.7; 1.3] + 1E-04 * eye(p.nx);

% ======================================================================= %
% Cost-function-related computations

% Parameters of a quadratic stage cost function in the states and inputs
R = 0.1;              % Input penalty: Penalize large input values
Q = diag([1, 0.1]);   % State penalty: Penalize large state values

% Compute terminal penalty from unconstrained, infinite horizon LQR
[~, P] = lqr(model, Q, R);

% Vectorized weight matrices
p.Lu = repmat({R}, 1, p.N);
p.Lu = blkdiag(p.Lu{:});

p.Lx = [repmat({Q}, 1, p.N), {P}];
p.Lx = blkdiag(p.Lx{:});
                                
% ======================================================================= %
% Quadratic programming reformulation: Cost function

% Initialize cell arrays, representing the block matrices M and W for the
% nominal dynamics and V for the influence of the uncertainty
M = cell(p.N+1, 1);
W = cell(p.N+1, p.N); 
V = cell(p.N+1, p.N);

% Build up block matrices M and W to directly express the state sequence in
% terms of the initial state and the input sequence
for i = 1 : 1 : p.N+1
   
    if i == 1
       
        % Initialize first row of the block matrices
        M{i} = eye(p.nx);
        W(i, :) = {zeros(p.nx, p.nu)};
        V(i, :) = {zeros(p.nx, p.nx)};
        
    else
        
        % Add appropriate powers of A to M
        M{i} = model.A^(i-1);

        % Exploit structure of W: first element of each line needs to be
        % computed while all follow-up elements are the previous line
        % shifted by one to the right
        W(i, :) = [{model.A^(i-2) * model.B}, W(i-1, 1:end-1)];
        
        % Compute block martix V analogously to W
        V(i, :) = [{model.A^(i-2)}, V(i-1, 1:end-1)];
        
    end
    
end

% Convert the cell arrays representing the block matrices into numerical
% arrays (matrices)
M = cell2mat(M);
W = cell2mat(W);
V = cell2mat(V);

% Compute the QP cost matrix H
p.H = 2 * (p.Lu + W' * p.Lx * W);

% Due to numeric errors, the matrix is only almost symmetric (deviations in
% late decimal places). As QP requires H to be symmetric, it is symmetrized
% to compensate for those numerical errors
p.H = (p.H + p.H') / 2;

% Precompute inverse Hessian matrix
p.H_inv = eye(size(p.H)) / p.H;

% Compute the QP cost matrix f^T as a function of the initial condition
p.fT = @(x)(2 * x' * M' * p.Lx * W);

% Compute offset as a function of the initial condition
p.const = @(x)(x' * M' * p.Lx * M * x);

% ======================================================================= %
% Constraint-related computations and QP reformulation

% Box constraints on each input along the horizon Ju u <= bu
Ju = [1; -1]; 
bu = [2; 20];

% Concatenate input constraints over the entire horizon
Ju = repmat({Ju}, 1, p.N);
p.Ju = blkdiag(Ju{:});

p.bu = repmat(bu, p.N, 1);

% Box constraints on each state along the horizon Jx x <= bx
Jx = [1, 0; -1, 0; 0, 1; 0, -1];
bx = [11; 4; 1.5; 4];

% Concatenate state constraints over the entire horizon
Jx = repmat({Jx}, 1, p.N+1);
p.Jx = blkdiag(Jx{:});

p.bx = repmat(bx, p.N+1, 1);

% Compute coefficient matrix of inequality constraints
p.Fu = p.Jx * W;

% Compute inhomogeneity for inequality constraints (dependent on initial
% condition of the horizon)
p.gu = @(x)(p.bx - p.Jx * M * x);

% Compute covaraince matrix of state sequence
Sigma_x = repmat({p.Sigma_w}, 1, p.N);
Sigma_x = blkdiag(Sigma_x{:});
Sigma_x = V * Sigma_x * V';

% Lower Cholesky factor of state covariance matrix
Sigma_x_sqrt = chol(Sigma_x + 1E-08 * eye(size(Sigma_x)), 'Lower');

% Build vector for tightening the state constraints
z = zeros(size(p.Jx, 1), 1);

for i = 1 : 1 : length(z)
    
    z(i) = norm(Sigma_x_sqrt * p.Jx(i, :)');
    
end

p.z = z;

% Risk allocation
Delta = 0.1;                                             % Joint risk
delta = Delta / size(p.Jx, 1) * ones(size(p.Jx, 1), 1);  % Individual risks

% Coefficients for distributionally robust tightening
p.phi_dr = sqrt((1 - delta) ./ delta);

% Coefficients for exact tightening, assuming a Gaussian distribution
p.phi_ex = norminv(1 - delta);