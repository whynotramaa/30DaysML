## Lasso vs Ridge Regreession
Both are linear regression methods that add a penalty to control overfitting. Ridge shrinks coefficients smoothly. Lasso can shrink some coefficients exactly to zero, which makes it useful for feature selection.

Start with ordinary linear regression and add a penalty term to the loss function so large coefficients are discouraged.
- This reduces overfitting.
- This helps when features are correlated or when data is high dimensional.

The difference is how they penalize coefficients.

Ridge - Penalty -> sum of squared coefficients.
Lasso - Penalty -> sum of coefficients.

```
When features are highly correlated:
Ridge spreads the weight across them. Lasso tends to keep one and zero out the rest (which one can be unstable).
This is why Elastic Net exists: it combines both penalties. => MultiCollinearity
```
---
***RIDGE*** --->
Loss = MSE + Squared Penalty

Expanded form:
$$\mathcal{L}_{ridge}(\beta)=(y - X\beta)^T (y - X\beta)+\lambda \sum_{j=1}^{p} \beta_j^2$$

Closed-form solution:
$$\hat{\beta}_{ridge}=(X^TX + \lambda I)^{-1} X^T y$$

---
***LASSO*** ---> Loss = MSE + Penalty

Expanded form:
$$\mathcal{L}_{lasso}(\beta)=(y - X\beta)^T (y - X\beta)+\lambda \sum_{j=1}^{p} |\beta_j|$$


Closed-form solution:
$$\text{No closed-form solution}$$


### Bias Variance Tradeoff
As λ increases, model complexity decreases. This increases bias and decreases variance, reducing overfitting. The optimal λ is chosen by minimizing test error (bias² + variance + noise), not by intersecting bias and variance curves. The test error curve is smooth and U-shaped, with a minimum at an intermediate λ.

### Sparsity ?
Sparsity means many coefficients are exactly zero. This is usefule for multiple reasons, feature selection, interpretability, high dimensional settings ( higher dim data are easier to overfit, so we prefer sparse data ehich are easier to generalize ), high efficiency.

### How is this possible that ridge can never go to zero but lasso can, & stops right there without being negative ?

Rdige Penalty = λ * ∑(β)^2
This means -> smooth, differentiable everywhere and gradient is proportional to ​​β. As long as β != 0, there is a force pulling it toward zero. But nothing special happens at zero. So coefficients get smaller and smaller, but do not snap to zero.

Lasso penalty: 𝜆*∑∣𝛽𝑗∣ which is Not differentiable at zero. Gradient Behavior :
1. 𝛽𝑗 > 0 means gradient = +𝜆
2. 𝛽𝑗 < 0 means gradient = -𝜆
1. 𝛽𝑗 = 0 means gradient is in a range, not a point.

This creates a dead zone around zero. What happens during optimization:
- Small coefficients are pulled toward zero
- Once a coefficient hits zero, the penalty exactly balances the loss gradient
- There is no incentive to cross zero

L1 penalty creates a kink at zero. To move from:
zero → positive, requires overcoming a constant penalty +λ
zero → negative, requires overcoming −λ

If the data signal is weak:
Neither side wins, Zero is the optimal point. So the coefficient stays exactly at zero.


