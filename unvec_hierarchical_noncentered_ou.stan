// Hierarchical model with equal time points, non-centered
// Declare data variables 
data{
  int<lower=0> N;          //Number of time series
  int<lower=0> T;          //Number of time points, same in each
  real<lower=0> Y[N,T];      //observations matrix as 2D array
  vector[T] time;           //observation times
}
// Make some necessary data transformation  
transformed data{
  vector[T] time_vec = to_vector(time);
}
// Parameter declared in parameters and transformed paramters blocks will be estimated
parameters{
  real lambda_log; 
  real kappa_log;  
  real mu; 
  real <lower=2> student_df; 
  matrix[N,T] error_terms;   //Error terms
}
// Parameter transformations
transformed parameters{
  // Define some helper terms 
  real sigma_log = 0.5*(kappa_log + lambda_log + log2());
  real<lower=0> sigma = exp(sigma_log);
  real<lower=0> lambda = exp(lambda_log);
  real<lower=0> kappa = exp(kappa_log);
  real<lower=0> kappa_inv = exp(-kappa_log);
  real<lower=0> kappa_sqrt = exp(0.5*kappa_log);
  
  // Store the latent OUP values 
  matrix[N, T] X_latent; 
  
  // Relation between error terms and latent values
  for(i in 1:N){
    vector [T-1] delta_t = segment(time_vec, 2, T - 1) - segment(time_vec,1,T-1);
    real X;
    
    for(k in 1:T){
      real epsilon = error_terms[i,k];
      if(k == 1){
        //For the first latent value use the stationary distribution.
        X = mu + epsilon * kappa_sqrt;
      }else{
        real t = delta_t[k-1];
        real exp_neg_lambda_t = exp(-t*lambda);
        real sd_scale = kappa_sqrt .* sqrt(1-square(exp_neg_lambda_t));
        X = mu - (mu - X) .* exp_neg_lambda_t + epsilon .* sd_scale;
      }
      X_latent[i, k] = X;
    }
  }
  
}
model{
  // Add log lambda and kappa to target log density
  target += lambda_log;
  target += kappa_log;
  
  // Add log densities of error terms to target log density
  for(i in 1:N){
    
    
    vector[T] error_terms2 = to_vector(square(segment(error_terms[i],1,T)));
    vector[T] cum_error_terms2 = cumulative_sum(append_row(0,error_terms2[1:T-1]));    
    
    
    
    for(k in 1:T){
      target +=  (lgamma((student_df + k) * 0.5) - lgamma((student_df+ k - 1 )* 0.5));     
      target += -0.5 * (student_df + k) * log1p(error_terms2[k] / (student_df + cum_error_terms2[k] - 2));
      target += -0.5 * log(student_df + cum_error_terms2[k] - 2);
    }
    
  }
  
  // Log probability for the observations given the latent values
  for(i in 1:N) {
    Y[i] ~ normal(X_latent[i], 0.1);
  }
  
  // Prior probabilities.
  // lambda ~ gamma(2,2);
  // kappa ~ gamma(2,2);
  lambda ~ normal(0.1,5);
  kappa ~ normal(0.1,5);
  mu ~ normal(8.4 , 1.3);
  student_df ~ gamma(2,.1);
}
