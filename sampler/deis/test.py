import jax
import jax.numpy as jnp

def quad_root(a, b, c):
    num = -b + jnp.sqrt(b**2 - 4 * a * c)
    return num / 2 / a
def get_linear_alpha_fns(beta_0, beta_1):
    def log_alpha_fn(t):
        log_mean_coef = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
        return 2 * log_mean_coef

    def t2alpha_fn(t):
        return jnp.exp(log_alpha_fn(t))

    def alpha2t_fn(alpha):
        log_mean_coef_from_alpha = jnp.log(alpha) / 2
        return quad_root(0.25 * (beta_1 - beta_0), 0.5 * beta_0, log_mean_coef_from_alpha)

    return t2alpha_fn, alpha2t_fn

t2alpha_fn, alpha2t_fn = get_linear_alpha_fns(0.1,2)
l = [1000,800,600,400,200,0]
l = list(l)
a = t2alpha_fn(l)
print(a)