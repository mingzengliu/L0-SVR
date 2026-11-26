% function  alg_l0_svr_admm
function [w, b, tau, history] = alg_l0_svr_admm_time(X,y, C, epsilon, lambda_scale, rho, tol)

% l0_SVR_ADMM  Solve SVR with l0-regularization problem via ADMM
%
% [w, b, tau, history] = alg_l0_svr_admm(X,y, C, epsilon, lambda_scale, tol,  rho)
%
% Solves the following problem via ADMM:
%
%   minimize 1/2*|| w||_2^2 + C e_m^T (y - Xw - b e_m - tol e_m)_+
%   Ce_m^T(Xw+be_m-y-tol e_m)_++ \lambda || w ||_0
%
% The solution is returned in the vector tau.
% Input:  X: m x n; y: m x 1; C, epsilon, tol, lambda_scale, rho: 1 x 1;
%
% history is a structure that contains the objective value, the primal and
% dual residual norms, and the tolerances for the primal and dual residual
% norms at each iteration.
%
% rho is the augmented Lagrangian parameter.
%
% alpha is the over-relaxation parameter (typical values for alpha are
% between 1.0 and 1.8).
%
%
% More information can be found in the paper linked at:
% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
%

%%  Parameters:  rho, C, epsilon(varepsilon for SVR), lambda_scale(for lambda) and tol (for stoppting )
% C = 1; 
% epsilon = 0.01; % epsilon for svr
% lam_scale = 1E-9;
% rho = 1;
% tol = 0.01;


if nargin < 2
	error("The number of arguments too less! \n");
end

if nargin < 3
	C  = 1;
	epsilon = 0.01;
	lambda_scale = 1E-9;
	rho = 1.0;
	tol = 0.01;
elseif nargin < 4
	C  = C;
	epsilon = 0.01;
	lambda_scale = 1E-9;
	rho = 1.0;
	tol = 0.01;
elseif nargin < 5
	C  = C;
	epsilon = epsilon;
	lambda_scale = 1E-9;
	rho = 1.0;
	tol = 0.01;
elseif nargin < 6
	C  = C;
	epsilon = epsilon;
	lambda_scale = lambda_scale;
	rho = 1.0;
	tol = 0.01;
elseif nargin < 7
	C  = C;
	epsilon = epsilon;
	lambda_scale = lambda_scale;
	rho = rho;
	tol = 0.01;
elseif nargin == 7
	C  = C;
	epsilon = epsilon;
	lambda_scale = lambda_scale;
	rho = rho;
	tol = tol;
end



%% computation of max_lambda and lambda
p_norm = norm(sum(X),'inf')^2;
beta = 1.0/rho;
lambda_max = 0.5 * C * beta * p_norm;
lambda = lambda_scale* lambda_max;
%% ==========================================
%% begin ADMM 
%% ==========================================
t_start = tic;

QUIET    = 1;
MAX_ITER = 1000;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;

[m, n] = size(X);

%% Initialize: w;b; beta; gamma; tau; u1; u2; u3;
w = zeros(n,1);
b = 0.0;
beta = zeros(m,1);
gamma = zeros(m,1);
tau = zeros(n,1);
u1 = ones(m, 1);
u2 = ones(m, 1);
u3 = ones(n, 1);
em = ones(m,1);


%% prepare A, B, and c
B = sparse( [X, em;
	                -X, em; 
		           - speye(n), zeros(n,1)] );
A = speye(2*m + n) ;
c = sparse( [y - epsilon; 
	                 -y - epsilon; 
	                zeros(n, 1)] );

% cache the factorization
[L U] = factor(X, rho);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective','theta1','theta2','theta3','theta4','theta5','theta6');
end

for k = 1:MAX_ITER

	%%  Update  \beta
	beta_old = beta;
	
    Xw = X * w;
	Z = y - Xw - b*em - epsilon* em - u1/rho;
	beta = prox_relu(C, rho, Z);
	
	%% Update \gamma
	gamma_old = gamma;
	Z = -y + Xw + b * em - epsilon * em - u2/rho;
	gamma = prox_relu(C, rho, Z);
	
	%% Update \tau
	tau_old = tau;
	Z = w - u3/rho;
	tau = prox_norm_zero(lambda, rho , Z);
	
	
	%% Update w by solving Hw = d;
    %% H = (rho + 1.0)* eye(n) + 2.0 * rho * X' * X;
	v1 = beta - y + b*em + epsilon*em + u1/rho;
	v2 = gamma + y - b *em + epsilon*em + u2/rho;
	v3 = tau + u3/rho;
	d = ( rho * (X'*(v2 - v1) + v3));
	w_old = w;
   
    if( m >= n )    % if skinny
       w = U \ (L \ (d));
    else            % if fat
       w = d/(rho+1) - (2.*rho.*X'*(U \ ( L \ (X*d) )))/(rho+1)^2;
	end
  
	
	%% Update b
    Xw = X * w;
	v1 = beta - y + Xw + epsilon*em + u1/rho;
	v2 = gamma + y - Xw + epsilon*em + u2/rho;
	b_old = b;
	% b = 0.5 * mean(v2 - v1);
    b = 0.5 * dot(v2-v1, em)/m;
	
	% Update u1, u2, u3
	eta = 0.5 * (sqrt(5) - 1); %1.0; % (1.0/k); 
	eta = 1.0;
	u1_old = u1;
	u2_old = u2;
	u3_old = u3;
	u1 = u1 + eta * rho * (beta - y + Xw + b*em + epsilon*em);
	u2 = u2 + eta * rho * (gamma + y - Xw - b*em + epsilon*em);
	u3 = u3 + eta * rho * (tau - w);


	% diagnostics, reporting, termination checks
	% history.objval(k)  = objective(X, y, C, epsilon, lambda, w, b, tau);	
	% history.r_norm(k)  = norm([(beta - y + X*w + b + epsilon); (gamma + y - X*w -b -epsilon); (tau - w)]);
	% history.s_norm(k)  = norm(rho* B' * [beta - beta_old; gamma - gamma_old; tau - tau_old;]);
	% history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max([norm(B* [w; b]), norm(A * [beta; gamma; tau]), norm(c)]);
	% history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*B'* [u1; u2; u3]);
	

	Xu1 = X' * u1; Xu2 = X' * u2;
	% history.theta1(k) = norm(w + X'* u1 - X' * u2 - u3).^2/( 1 +  norm(X'* u1 - X' * u2 - u3).^2);
    history.theta1(k) = norm(w + Xu1 - Xu2 - u3).^2/( 1 +  norm(Xu1 - Xu2 - u3).^2);
	history.theta2(k) = abs(sum(u1) - sum(u2));
	% history.theta3(k) = norm(beta - y + X * w + b + epsilon).^2 / sqrt(m);
	% history.theta4(k) = norm(gamma - X * w - b + y + epsilon).^2/sqrt(m);
    history.theta3(k) = norm(beta - y + Xw + b + epsilon).^2 / sqrt(m);
	history.theta4(k) = norm(gamma - Xw - b + y + epsilon).^2/sqrt(m);
	history.theta5(k) = norm(tau - w).^2/sqrt(n);
	history.theta6(k) = norm( tau - prox_norm_zero(lambda, 1.0/rho, tau - u3/rho)).^2 / (1.0 + norm(tau).^2);
	
	history.eps_opt(k) = max([history.theta1(k), history.theta2(k), history.theta3(k), history.theta4(k), history.theta5(k), history.theta6(k)]);
	
    if ~QUIET
%         fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\n', k, ...
%             history.r_norm(k), history.eps_pri(k), ...
%             history.s_norm(k), history.eps_dual(k), history.objval(k), ...
% 			history.theta1(k), history.theta2(k), history.theta3(k), ...
% 			history.theta4(k), history.theta5(k), history.theta6(k));
    end

%     if (history.r_norm(k) < history.eps_pri(k) && ...
%        history.s_norm(k) < history.eps_dual(k))
% %            break;
% 	end

	if (history.eps_opt(k) < tol)		
		break;
	end
end

if ~QUIET
    toc(t_start);
end
end %% function


function p = objective(X, y, C, tol, lambda, w, b, tau)
	p = 0.5 * sum(w.^2) + C*sum(max(y - X*w-b -tol, 0) ) + C* sum(max(X*w + b - y - tol, 0)) + lambda*nnz(tau);
end

function ret = prox_relu(C, rho, v)
%% this function solve the problem: 
%% min Ce_m^T max{0, x} + rho/2 \|x-v\|_2^2
	ret = zeros(size(v));
	indices = find(v < 0);
	ret(indices) = v(indices);
	indices = find(v > C/rho);
	ret(indices) = v(indices) - C/rho;
end

function ret = prox_norm_zero(lam, rho, v)
%% solve the problem as follows:
%% prox_{\lambda*\rho}(v) = min \lambda \|x\|_0 + rho/2 \|x - v\|_2^2

	ret = zeros(size(v));
	inds = find(abs(v) > sqrt(2.0 * lam / rho));
	ret(inds) = v(inds);
end

function [L U] = factor(A, rho)
    [m, n] = size(A);
    if ( m >= n )    % if skinny
       L = chol( (2*rho)*A'*A + (rho+1)*speye(n), 'lower' );
    else            % if fat
       L = chol( speye(m) + (2*rho)/(rho+1)*(A*A'), 'lower' );
    end

    % force matlab to recognize the upper / lower triangular structure
    L = sparse(L);
    U = sparse(L');
end


function plt_history(history)

	K = length(history.objval);

	h = figure;
	plot(1:K, history.objval, 'k', 'MarkerSize', 10, 'LineWidth', 2);
	ylabel('f(x^k) + g(z^k)'); xlabel('iter (k)');

	g = figure;
	subplot(3,1,1);
	semilogy(1:K, max(1e-8, history.r_norm), 'k', ...
		1:K, history.eps_pri, 'k--',  'LineWidth', 2);
	ylabel('||r||_2');

	subplot(3,1,2);
	semilogy(1:K, max(1e-8, history.s_norm), 'k', ...
		1:K, history.eps_dual, 'k--', 'LineWidth', 2);
	ylabel('||s||_2'); xlabel('iter (k)');
	
	
    subplot(3,1,3);
	semilogy(1:K, max(1e-8, history.eps_opt), 'k', ...
		1:K, min(history.eps_opt)*ones(K,1), 'k--', 'LineWidth', 2);
	ylabel('error'); xlabel('iter (k)');

end

