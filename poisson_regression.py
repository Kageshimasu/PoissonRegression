import pandas as pd 
import matplotlib.pylab as plt
import statsmodels.formula.api as smf
import numpy as np 
import scipy.optimize


# 今回使用するデータ
data = pd.read_csv('./psn_data.csv')


## 答え ##################################################
results = smf.poisson('y ~ x', data=data).fit()
print(results.summary())

# パラメータの推定値を取得
a, b = results.params

## 自作 ####################################################
params = [np.random.rand(), np.random.rand()]

def likelihood(params, y_vector, x_vector):
    ret = 0
    # ポアソン分布のパラメータ
    theta_reg = lambda params, x: np.exp(params[0] + params[1]*x)
    for i in range(y_vector.shape[0]):
        ret += y_vector[i] * np.log(theta_reg(params, x_vector[i])) - theta_reg(params, x_vector[i])
    return -ret

new_params = scipy.optimize.minimize(likelihood, params, args=(data['y'], data['x']))

print("自作で推定した結果")
print(list(new_params.x))
print("statsmodelsで推定した結果")
print(list(results.params))

# プロットを表示
plt.plot(data['x'], data['y'], 'o')
plt.plot(data['x'], np.exp(a + b*data['x']))
plt.text(0, 0, "a={:8.3f}, b={:8.3f}".format(a, b))
plt.show()
