// Student t OUP

functions {
  
  // compute shape matrix 
  matrix shape(real student_df, real sigma, real lambda, real[] time, int T){
    matrix[T, T] covariance;
    matrix[T, T] shape;
    
    for(i in 1:T) {
      for(j in 1:T) {
        covariance[i, j] = ((sigma^2)./(2*lambda))*exp(-lambda*fabs(time[i]-time[j]));
      }
    }
    
    // shape = ((student_df-2)/student_df)*covariance;
    shape = covariance;
    
    return shape;
  }
  
  // make vector with equal inputs
  vector vectorize(real mu, int T) {
    vector[T] vec;
    for(i in 1:T) {
      vec[i] = mu;
    }
    return(vec);
  }
  
}


data {
  int<lower=0> T;           //number of time points
  int<lower=0> N;           //number of series
  matrix[N, T] Y;           //observations
  real time[T];             //observation times
  real<lower=2> student_df;   // degrees of freedom
}

parameters {
  
  // vector [N] lambda_raw;
  vector<lower=0, upper=1> [N] lambda;
  // vector [N] sigma_raw;
  vector<lower=0> [N] sigma;
  // vector [N] mu_raw;
  vector [N] mu;
  
  // real<lower=0> student_df;
  
  // hyperparameters 
  // real<lower=0> lambda_alpha;
  // real<lower=0> lambda_beta;
  // real mu_mu;
  // real<lower=0> mu_sigma;
  // real<lower=0> sigma_mu;
  // real<lower=0> sigma_sigma;
}

transformed parameters {
  
  // non-center mu and sigma
  // vector [N] mu = mu_mu + mu_sigma*mu_raw;
  // vector<lower=0> [N] sigma = sigma_mu + sigma_sigma*sigma_raw;
  // // vector<lower=0> [N] lambda = lambda_mu + lambda_sigma*lambda_raw;
  
  // mode and variance of gamma prior
//   real<lower=0> gamma_mode = lambda_beta/(lambda_alpha + 1);
//   real<lower=0> gamma_variance = (lambda_beta^2)/((lambda_alpha-1)^2*(lambda_alpha));

}

model {
  
  for(i in 1:N) {
    Y[i] ~ multi_student_t(student_df, vectorize(mu[i], T), ((student_df-2)/student_df)*shape(student_df, sigma[i], lambda[i], time, T));
  }
  
  // priors
  lambda ~ inv_gamma(.1, .5);
  // lambda_raw ~ normal(0, 1);
  // lambda ~ normal(.5, 1);
  // lambda ~ gamma(2, 3);
  
  
  mu ~ normal(0, 10);
  // mu_raw ~ normal(0, 1);
  
  sigma ~ inv_gamma(.1, .5);
  // sigma_raw ~ normal(0, 1);
  
  // student_df ~ normal(0, 100);
  
  //hyper priors
  // lambda_alpha ~ normal(0, 5);
  // lambda_beta ~ normal(0, 5);
  // 
  // mu_mu ~ normal(5, 5);
  // mu_sigma ~ normal(0, 5);
  // 
  // sigma_mu ~ normal(5, 5);
  // sigma_sigma ~ normal(0, 5);
}