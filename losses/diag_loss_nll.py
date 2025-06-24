import tensorflow as tf
import tensorflow_probability as tfp

# custom loss function foo diag model (with 8 outputs)
def custom_diag_loss(y, p_base, ):
    mu = p_base[:, 0:4]
    raw_diag = p_base[:, 4:8]
    
    minval=1e-9
    maxval=1e9
    scale = 512
    
    # creating each matrix element in 4x4
    Mdia = tf.maximum(minval + tf.math.maximum(raw_diag, 0.0), maxval)
    
    
    # placeholder zero element
    zeros = tf.zeros_like(Mdia[:,0])
    
    # assembles scale_tril matrix
    row1 = tf.stack([Mdia[:,0],zeros,zeros,zeros])
    row2 = tf.stack([zeros,Mdia[:,1],zeros,zeros])
    row3 = tf.stack([zeros,zeros,Mdia[:,2],zeros])
    row4 = tf.stack([zeros,zeros,zeros,Mdia[:,3]])

    scale_tril = tf.transpose(tf.stack([row1,row2,row3,row4]),perm=[2,0,1])

    dist = tfp.distributions.MultivariateNormalTriL(
        loc = mu, 
        scale_tril = scale_tril
    ) 
    NLL = -dist.log_prob(y)

    return tf.reduce_mean(NLL) 
