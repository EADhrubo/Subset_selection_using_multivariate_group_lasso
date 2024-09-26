clc
clear 
close all


K = 100;
N = 20;
T = 10000;
lambda = 100;
no_iteration = 1;
P = N*K;
Q = N*K;

X = randn(T,P);
Y = randn(T,Q);

X_norm = sum(X.^2,1);

Beta = randn(P,Q);


%grp_Norm = zeros(K,1);

lambda_g = lambda * sqrt(N);



% E = zeros(T,Q);
% E = Y - X*Beta;
% Beta_old = Beta;

% Sjk = X'*E;
% X_norm_rep = repmat(X_norm,Q,1);
% grad_L_beta = Sjk + X_norm_rep*Beta_old;
% softShrk = abs(grad_L_beta/T);
% 
% softShrk_sq = sum(softShrk.^2,1);
% softShrk_sq = softShrk_sq';
% softShrkL2norm = zeros(K,1);
% for k=1:K
%     softShrkL2norm(k,1) = sqrt(sum(softShrk_sq(((k-1)*N + 1):k*N,1)));
% end

flag = 100;
n_iter = 1;
while (flag > 1e-2) && (n_iter <= no_iteration)
    Beta_last = Beta;
    Beta_old = Beta;
    for k=1:K
        initial_index = (k-1)*N + 1;
        final_index = k*N;
        Beta_old = Beta;
        Beta = UpdateBeta(initial_index,final_index,P,Q,N,T,K,lambda_g,X,X_norm,Beta_old);
        Beta(1:5,1:5)
        E = E + X*(Beta_old - Beta);
        rss = sqrt(sum(E.^2,"all"))
        flag = max(abs(Beta_old - Beta),[],"all")
    end
    n_iter = n_iter + 1
    condition_check = sum((Beta > Beta_last),'all')
    flag = max(abs(Beta_last - Beta),[],"all")
end

rss = sqrt(sum(E.^2,"all"));







function Beta = UpdateBeta(initial_index,final_index,P,Q,N,T,K,lambda_g,X,X_norm,Beta_old)
    Beta = zeros(P,Q);
    Beta_temp = Beta_old;
    Sjk_sq_sum = 0;
    for j = initial_index:final_index
        Beta_temp(j,:) = 0;
        error = Y - X*Beta_temp;
        for k = 1:Q
            Sjk(j,k) = X(:,j)'*error(:,k);
            Sjk_sq_sum = Sjk_sq_sum + Sjk(j,k)^2;
            grad_L_beta(j,k) = -Sjk(j,k) + X_norm(1,j)*Beta_old(j,k);
        end
    end
    softshrk = sqrt(Sjk_sq_sum)/ T;
    if softshrk <= lambda_g
        Beta(initial_index:final_index,:) = 0;
    else
        Beta_g_norm = sqrt(sum(Beta_old(initial_index:final_index,:).^2));
        for j = initial_index:final_index
            for k = 1:Q
                if (Beta_g_norm^2 - Beta_g_norm(j,k)^2) > 0.01
                    denominator = X_norm(1,j) + (T*lambda_g/Beta_g_norm);
                    Beta(j,k) = Sjk(j,k) / denominator;
                else
                    Beta(j,k) = sign(Sjk(j,k))*(abs(Sjk(j,k)) - T*lambda_g)/X_norm(1,j);
                end
            end
        end
    end
end



            

%     Sjk = X'*E;
%     X_norm_rep = repmat(X_norm,Q,1);
%     grad_L_beta = Sjk + X_norm_rep*Beta_old;
%     size(grad_L_beta)
%     softShrk = abs(grad_L_beta/T);
%     softShrk_sq = softShrk.^2;
%     
%         Beta_sq = Beta_old.^2;
%         grp_Norm = sqrt(sum(Beta_sq(initial_index:final_index,:),"all"))
%         if grp_Norm == 0
%             grp_Norm = 0.01;
%         end
% 
%         sum_sjk = sqrt(sum(softShrk_sq(initial_index:final_index,:),"all"))
%         if sum_sjk < lambda_g
%             Beta(initial_index:final_index,:) = 0;
%         else
%             normalizing = X_norm(1,initial_index:final_index) + T*(lambda_g/grp_Norm);
%             Beta(initial_index:final_index,:) = grad_L_beta(initial_index:final_index,:)./normalizing';
%         end
%     end
% 
% 
% 
% end
