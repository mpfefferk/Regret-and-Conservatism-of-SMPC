function [u_seq, f_val] = Linear_MPC(u_seq_0, x_0, p, tightening)

% Evaluate matrix f^T for the current horizon's initial state
fT = p.fT(x_0);

% Evaluate inhomogeneity for (nominal) constraints
gu = p.gu(x_0);

% Compute tightening of the state constraints
if strcmp(tightening, 'exact')
    gu = gu - p.phi_ex .* p.z;
elseif strcmp(tightening, 'robust')
    gu = gu - p.phi_dr .* p.z;
end

% Compute optimization to solve the OCP
% [u_seq, fval] = quadprog(p.H, ...   Cost function parameter H
%                  fT, ...            Cost function parameter f^T
%                  [p.Ju; p.Fu], ...  Coefficient matrix for linear inequality constraints
%                  [p.bu; gu], ...    Inhomogeneity for linear inequality constraints                                   
%                  [], ...            Coefficient matrix for linear equality constraints
%                  [], ...            Inhomogeneity for linear equality constraints
%                  [], ...            Lower bound on u_seq
%                  [], ...            Upper bound on u_seq
%                  u_seq_0, ...       Initial guess
%                  [] ...             Solver options
%                  );  

[u_seq, f_val] = fmincon(@(u_seq)(0.5 * u_seq' * p.H * u_seq + fT * u_seq), ...    
                 u_seq_0, ...     
                 [p.Ju; p.Fu], ...      
                 [p.bu; gu] ...       
                 );

% Add off-set to cost function, which is independent of u
f_val = f_val + p.const(x_0);

% !!! fval is not the actual cost value, as the term that is independent of
% u and x_0 (but only dependent on the system dynamics, noise covariance
% and weighting marix Q) is not computed. It will have no effect on the
% regret.
             
end