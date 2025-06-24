import tensorflow as tf
import tensorflow_probability as tfp

# custom loss function foo diag model (with 8 outputs)
def custom_diag_loss(y, p_base, ):
    mu = p_base[:, 0:4]
    
    minval=1e-9
    maxval=1e9
    
    log_sigma   = p_base[:, 4:8]
    sigma_diag  = tf.nn.softplus(log_sigma) + 1e-6
    
    dist = tfp.distributions.MultivariateNormalDiag(
        loc=mu,
        scale_diag=sigma_diag
    )
    NLL = -dist.log_prob(y)

    return tf.reduce_sum(NLL) 
