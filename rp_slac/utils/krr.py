import jax.random as jr
import jax.numpy as jnp
from jax import Array
from tqdm import tqdm


class GPUKernelRidge:

    def __init__(
        self,
        X,
        Y, 
        params: dict, 
        key: jr.PRNGKey, 
        batch_frac: float | None
    ):

        if not all(k in params.keys() for k in ["sigma", "lambda"]):
            raise KeyError("GPUKernelRidge Params requires keys 'sigma' and 'lambda'.")

        self.key = key
        self.x = self._reshape(X)
        self.y = self._reshape(Y)
        self.params = params

        if batch_frac:

            assert 0 < batch_frac < 1
            n = self.x.shape[0]
            m = int(n * batch_frac)
            random_idx = jr.choice(self.key, a=n, shape=(m, ), replace=False)
            self.key, _ = jr.split(self.key)
            
            self.x = self.x[random_idx]
            self.y = self.y[random_idx]

        self.y_shape = Y.shape
        
        self.k = self.kernel(self.x, self.x)
        self.alpha = self.fit(self.y, self.k)

    def fit(self, Y, K):

        lam = self.params.get("lambda")
        N = K.shape[0]
        return jnp.linalg.solve(K + lam * jnp.eye(N), Y)
    
    def predict(self, test_X, batch_size: int = 500):
        """
        Due to large memory requirements we batch over test_X and concat predictions at the end
        """

        M = test_X.shape[0]
        test_X = self._reshape(test_X)
        MT = test_X.shape[0]

        all_preds = []
        for i in range(0, MT, batch_size):

            batch_X = test_X[i: i + batch_size]
            K = self.kernel(batch_X, self.x)
            preds = K @ self.alpha
            all_preds.append(preds)

        preds = jnp.concatenate(all_preds, axis=0)
        
        shape = (M, *self.y_shape[1:])
        return jnp.reshape(preds, shape)
    
    def kernel(self, X1, X2):
        """ Gaussian Kernel """
        sigma = self.params.get("sigma")
        
        if sigma <= 0:
            raise ValueError("Sigma must be a positive value != 0")
        
        X1_sq = jnp.sum(X1**2, axis=1, keepdims=True)
        X2_sq = jnp.sum(X2**2, axis=1, keepdims=True)
        dist_sq = X1_sq - 2 * X1 @ X2.T + X2_sq.T
        
        return jnp.exp(-0.5 * dist_sq / sigma**2)
    
    def r2(self, test_Y, preds):
        assert test_Y.shape == preds.shape
        
        test_Y = self._reshape(test_Y)
        preds = self._reshape(preds,)

        RSS = jnp.sum((test_Y - preds)**2, axis=0)
        CTSS = jnp.sum((test_Y - jnp.mean(test_Y, axis=0))**2, axis=0)
        return 1.0 - (RSS / CTSS)
    
    def _reshape(self, V: Array):
        """ 
        1. Kernel Ridge needs matrices of shape N, D 
        2. Add a time dimension such that N, T, K becomes N, T, K + 1 
            with the K + 1 th element being the index in time
        """
        shape = V.shape
        return jnp.reshape(V, (shape[0] * shape[1], shape[2]))

    def update(self, params: dict):
        self.params.update(params)
        self.k = self.kernel(self.x, self.x)
        self.alpha = self.fit(self.y, self.k)


def standardize(
    train_data: Array,
    val_data: Array
):
    mean = jnp.mean(train_data, axis=(0, 1), keepdims=True)
    std = jnp.std(train_data, axis=(0, 1), keepdims=True) + 1e-8  # avoid div by zero

    scaled_train = (train_data - mean) / std
    scaled_val = (val_data - mean) / std
    return scaled_train, scaled_val



def sweep_krr(
    train_X, 
    train_Y,
    val_X,
    val_Y,
    sigma_range,
    lambda_range,
    batch_frac: float | None = None
):
    """
    Given sequences for sigma and lambda, find the best Kernel Ridge fit. 
    Report the fit, the R^2 value and return the best sigma, lambda pair that produced it

    """

    train_X = train_X[None, ...] if train_X.ndim == 2 else train_X
    train_Y = train_Y[None, ...] if train_Y.ndim == 2 else train_Y
    
    # standardize the posterior means output
    train_X, val_X = standardize(train_X, val_X)
    train_Y, val_Y = standardize(train_Y, val_Y)
    
    # grid search best hyper parameters
    params = {"sigma": 25, "lambda": 0.001}
    key = jr.PRNGKey(0)
    gpu_krr = GPUKernelRidge(train_X, train_Y, params, key, batch_frac=batch_frac)
    
    curr_r2 = 0
    best_lambda = None
    best_sigma = None
    best_preds = None
    
    i = 0
    for lam in tqdm(lambda_range, desc="Fitting KRR"):
        for j, sigma in enumerate(sigma_range):
            
            params = {"sigma": sigma, "lambda": lam}
            gpu_krr.update(params)
            preds = gpu_krr.predict(val_X)
            avg_r2 = jnp.mean(gpu_krr.r2(val_Y, preds))

            if avg_r2 > curr_r2:
                curr_r2 = avg_r2
                best_sigma = sigma
                best_lambda = lam
                best_preds = preds
        # print(curr_r2, avg_r2)
        i += 1
        
    if best_sigma is None or best_lambda is None:
        return gpu_krr, {"sigma": best_sigma, "lambda": best_lambda, "r2": curr_r2}

    
    params = {"sigma": best_sigma, "lambda": best_lambda}
    gpu_krr.update(params)
    return gpu_krr, {"sigma": best_sigma, "lambda": best_lambda, "r2": curr_r2}